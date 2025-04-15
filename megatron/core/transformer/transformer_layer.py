# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import warnings
from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import torch
import torch.distributed

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.transformer.cuda_graphs import CudaGraphManager
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_viewless_tensor
from contextlib import contextmanager


class AsyncCommBucket:
    """
    Store aynchronous communication operations.
    """

    def __init__(self):
        self._async_op: Dict[int, "dist.Work"] = {}
        self._copy_async_op: Dict[int, "dist.Work"] = {}

    def add(self, op_name: int, work: "dist.Work"):
        assert op_name not in self._async_op, f"Operation with name: {op_name} already exists"
        assert work is not None
        self._async_op[op_name] = work
        self._copy_async_op[op_name] = work
    def get(self, op_name: str) -> "dist.Work":
        if op_name not in self._async_op:
            raise KeyError(f"Operation with name: {op_name} doesn't exist")

        return self._async_op.get(op_name)

    def pop(self, op_name: str) -> "dist.Work":
        if op_name not in self._async_op:
            raise KeyError(f"Operation with name: {op_name} doesn't exist")

        return self._async_op.pop(op_name)

    def wait(self, op_name: str):
        """Wait and remove the operation from the bucket"""
        work = self.pop(op_name)
        work.wait()

    def is_all_completed(self) -> bool:
        if not len(self._async_op) == 0:
            return False

        not_finished = []
        for k, v in self._copy_async_op.items():
            if v.is_completed() is not True:
                not_finished.append((k, v))
        return len(not_finished) == 0

    def clear_all(self):
        self._async_op.clear()
        self._copy_async_op.clear()
CUDA_STREAM_COMM_NAME = "comm_stream_{}"

class CudaStreamManager:
    def __init__(self):
        self._streams: Dict[str, "torch.cuda.Stream"] = {}
        self.comm_bucket = AsyncCommBucket()

    def init_default_comm_stream(self):
        """
        Initialize the default communication stream for the current cuda device.
        """
        self.create(CUDA_STREAM_COMM_NAME.format(torch.cuda.current_device()), torch.cuda.current_device())

    def create(self, name: str, device: torch.device):
        assert name not in self._streams
        self._streams[name] = torch.cuda.Stream(device=device)

    def get(self, name: str):
        if name not in self._streams:
            self.create(name, torch.cuda.current_device())
        return self._streams.get(name)

    def get_default_comm_stream(self) -> torch.cuda.Stream:
        """
        Return the default communication stream for the current cuda device.
        """
        return self.get(CUDA_STREAM_COMM_NAME.format(torch.cuda.current_device()))

    @contextmanager
    def run_on_stream(self, name: str):
        stream = self.get(name)
        with torch.cuda.stream(stream):
            yield stream
            
def is_sync_stream():
    return True

class WaitComm(torch.autograd.Function):
    """
    Enforce a tensor to wait for the communication operation to finish
    in torch's autograd graph.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, op_name: str, comm_stream: torch.cuda.Stream, comm_bucket: AsyncCommBucket):
        assert isinstance(comm_stream, torch.cuda.Stream)
        ctx.op_name = op_name
        ctx.comm_stream = comm_stream
        ctx.comm_bucket = comm_bucket
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        NOTE: because the communication operation is already being executed
        so the communication stream don't have to wait for the compute stream here
        but the compute stream waits for the communication stream
        before proceeding
        """
        if is_sync_stream(ctx.op_name):
            handle = ctx.comm_bucket.pop(ctx.op_name)
            handle.wait()

            ctx.comm_stream.synchronize()
            torch.cuda.default_stream().wait_stream(ctx.comm_stream)

        return grad_output, None, None, None

def insert_backward_sync_to_tensor(
    tensor: torch.Tensor, op_name: str, stream_manager: CudaStreamManager
) -> torch.Tensor:
    """
    Insert a wait communication operation of a given op_name to the autograd graph
    of a tensor.
    """

    assert isinstance(stream_manager, CudaStreamManager)
    comm_stream = stream_manager.get(CUDA_STREAM_COMM_NAME.format(torch.cuda.current_device()))
    return WaitComm.apply(tensor, op_name, comm_stream, stream_manager.comm_bucket)


ALLTOALL_B0_HANDLE=None
ALLTOALL_B1_HANDLE=None

def get_transformer_layer_offset(config: TransformerConfig):
    """Get the index offset of current pipeline stage, given the level of pipelining."""
    pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
    if not parallel_state.is_inside_encoder():
        pp_decoder_start = parallel_state.get_pipeline_model_parallel_decoder_start()
        if pp_decoder_start is not None:
            pipeline_rank = pipeline_rank - pp_decoder_start

    if config.pipeline_model_parallel_size > 1:

        if (
            config.num_layers_in_first_pipeline_stage is not None
            or config.num_layers_in_last_pipeline_stage is not None
        ):
            # Calculate number of pipeline stages to distribute the remaining Transformer
            # layers after deducting the Transformer layers in the first or the last stages
            middle_pipeline_stages = config.pipeline_model_parallel_size
            middle_pipeline_stages -= sum(
                [
                    1 if x is not None else 0
                    for x in (
                        config.num_layers_in_first_pipeline_stage,
                        config.num_layers_in_last_pipeline_stage,
                    )
                ]
            )

            # Calculate layers to distribute in each pipeline stage. If the
            # num_layers_in_first_pipeline_stage and num_layers_in_last_pipeline_stage
            # are not set, we will not enable uneven pipeline. All layers will be treated
            # as middle layers.
            num_layers_in_first_pipeline_stage = (
                0
                if config.num_layers_in_first_pipeline_stage is None
                else config.num_layers_in_first_pipeline_stage
            )
            num_layers_in_last_pipeline_stage = (
                0
                if config.num_layers_in_last_pipeline_stage is None
                else config.num_layers_in_last_pipeline_stage
            )

            middle_num_layers = (
                config.num_layers
                - num_layers_in_first_pipeline_stage
                - num_layers_in_last_pipeline_stage
            )

            if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
                vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

                # Calculate number of layers in each virtual model chunk
                # If the num_layers_in_first_pipeline_stage and
                # num_layers_in_last_pipeline_stage are not set, all pipeline stages
                # will be treated as middle pipeline stages in the calculation
                num_layers_per_virtual_model_chunk_in_first_pipeline_stage = (
                    0
                    if config.num_layers_in_first_pipeline_stage is None
                    else config.num_layers_in_first_pipeline_stage // vp_size
                )

                num_layers_per_virtual_model_chunk_in_last_pipeline_stage = (
                    0
                    if config.num_layers_in_last_pipeline_stage is None
                    else config.num_layers_in_last_pipeline_stage // vp_size
                )

                num_layers_per_vritual_model_chunk_in_middle_pipeline_stage = (
                    middle_num_layers // vp_size
                )

                # First stage + middle stage + last stage
                total_virtual_chunks = (
                    num_layers_per_virtual_model_chunk_in_first_pipeline_stage
                    + num_layers_per_vritual_model_chunk_in_middle_pipeline_stage
                    + num_layers_per_virtual_model_chunk_in_last_pipeline_stage
                )

                # Calculate the layer offset with interleaved uneven pipeline parallelism
                if pipeline_rank == 0:
                    offset = vp_rank * total_virtual_chunks
                else:
                    offset = (
                        vp_rank * total_virtual_chunks
                        + num_layers_per_virtual_model_chunk_in_first_pipeline_stage
                        + (pipeline_rank - 1)
                        * (
                            num_layers_per_vritual_model_chunk_in_middle_pipeline_stage
                            // middle_pipeline_stages
                        )
                    )
            else:
                if middle_pipeline_stages > 0:
                    num_layers_per_pipeline_rank = middle_num_layers // middle_pipeline_stages
                else:
                    num_layers_per_pipeline_rank = 0

                middle_pipeline_rank = (
                    pipeline_rank
                    if config.num_layers_in_first_pipeline_stage is None
                    else pipeline_rank - 1
                )

                if pipeline_rank == 0:
                    offset = 0
                else:
                    offset = (
                        middle_pipeline_rank * num_layers_per_pipeline_rank
                    ) + num_layers_in_first_pipeline_stage
        else:
            num_layers = config.num_layers

            # Increase the number of layers by one if we include the embedding (loss)
            # layer into pipeline parallelism partition and placement
            if config.account_for_embedding_in_pipeline_split:
                num_layers += 1

            if config.account_for_loss_in_pipeline_split:
                num_layers += 1

            num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

            if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
                vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

                num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
                total_virtual_chunks = num_layers // vp_size
                offset = vp_rank * total_virtual_chunks + (
                    pipeline_rank * num_layers_per_virtual_rank
                )

                # Reduce the offset of embedding layer from the total layer number
                if (
                    config.account_for_embedding_in_pipeline_split
                    and not parallel_state.is_pipeline_first_stage()
                ):
                    offset -= 1
            else:
                offset = pipeline_rank * num_layers_per_pipeline_rank

                # Reduce the offset of embedding layer from the total layer number
                if (
                    config.account_for_embedding_in_pipeline_split
                    and not parallel_state.is_pipeline_first_stage()
                ):
                    offset -= 1
    else:
        offset = 0
    return offset


@dataclass
class TransformerLayerSubmodules:
    """
    Configuration class for specifying the submodules of a transformer layer.

    This class defines the structure and default implementations for various
    components of a transformer layer, allowing for flexible customization
    of the layer's architecture.

    Args:
        input_layernorm (Union[ModuleSpec, type]): Specification for the input layer normalization.
        self_attention (Union[ModuleSpec, type]): Specification for the self-attention mechanism.
        self_attn_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after self-attention.
        pre_cross_attn_layernorm (Union[ModuleSpec, type]): Specification for the layer
            normalization before cross-attention.
        cross_attention (Union[ModuleSpec, type]): Specification for the cross-attention mechanism.
        cross_attn_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after cross-attention.
        pre_mlp_layernorm (Union[ModuleSpec, type]): Specification for the layer normalization
            before the MLP.
        mlp (Union[ModuleSpec, type]): Specification for the MLP in Dense layer.
        mlp_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after the MLP.
        sharded_state_dict_keys_map (Dict[str, str]): Mapping for sharded tensor keys to be applied
            in the `sharded_state_dict` method.
    """

    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class BaseTransformerLayer(ABC):
    """A common parent class for `TransformerLayer` like implementations.

    A dummy class that is subclassed by similar `TransformerLayer`s e.g. the
    `TransformerLayer` in this file and possibly other `TransformerLayer`
    implementations that aim to use `TransformerBlock` as the base module.
    The main purpose is to check if any layer (or module) provided in the spec
    is a subclass of this class to allow fanning-out of that spec for all the
    layers in the `TransformerBlock`. See `_get_block_submodules` method
    implementation in `transformer_block.py` file for more details.
    """

    def __init__(self):
        pass



class CommStreamMananger():
    def __init__(self, name):
        self.stream=torch.cuda.stream()
        self.stream_name=name
    
    def get_stream(self):
        return self.stream
    
    def get_name(self):
        return self.stream_name
    
    def sync_stream(self):
        torch.cuda.default_stream().wait(self.stream)
    
        
class TransformerLayer(MegatronModule, BaseTransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
    ):
        super().__init__(config=config)
        self.commstream_mananger=CudaStreamManager()
        # self.commstream_mananger_new.
        # self.commstream_mananger_old=CommStreamMananger("fwdall2all")
        if config.enable_cuda_graph:
            if not self.training:
                # Cudagraphs for inference are only enabled with the flash decoding kernel
                assert (
                    self.config.flash_decode
                ), "--flash-decode is required to use CUDA graphs during inference"
            self.cudagraph_manager = CudaGraphManager(config)

        self.submodules_config = submodules
        self.layer_number = layer_number + get_transformer_layer_offset(self.config)
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout

        # [Module 1: Input Layernorm] Optional Layernorm on the input data
        # TODO: add pytorch only layernorm
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        attention_optional_kwargs = {}
        if config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[self.layer_number]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type

        # [Module 2: SelfAttention]
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=layer_number,
            **attention_optional_kwargs,
        )

        # [Module 3: BiasDropoutFusion]
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        # [Module 4: Post SelfAttention] Optional Layernorm after self-attn
        self.pre_cross_attn_layernorm = build_module(
            submodules.pre_cross_attn_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # [Module 5: CrossAttention]
        self.cross_attention = build_module(
            submodules.cross_attention,
            config=self.config,
            layer_number=layer_number,
            **attention_optional_kwargs,
        )

        # [Module 6: BiasDropoutFusion]
        self.cross_attn_bda = build_module(submodules.cross_attn_bda, config=self.config)

        # [Module 7: Pre MLP] Optional Layernorm before MLP
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        # [Module 8: MLP block]
        self.mlp = build_module(submodules.mlp, config=self.config)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        # [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

    @staticmethod
    def _get_layer_offset(config: TransformerConfig):
        """
        Get the layer offset for the current pipeline stage.

        Deprecated: please use `get_transformer_layer_offset` instead.
        """

        warnings.warn(
            "TransformerLayer._get_layer_offset is deprecated."
            "Please use get_transformer_layer_offset instead."
        )
        return get_transformer_layer_offset(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
    ):
        """
        Perform a forward pass through the transformer layer.

        This method implements the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask tensor for cross-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor, optional): Bias tensor for Q * K.T.
            inference_params (object, optional): Parameters for inference-time optimizations.
            packed_seq_params (object, optional): Parameters for packed sequence processing.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                output (Tensor): Transformed hidden states of shape [s, b, h].
                context (Tensor): Updated context tensor if cross-attention is used,
                otherwise None.
        """
        #hidden_states_shape [s,b,h]
        # Residual connection.[s, b,h]
        
        residual0=hidden_states[:,0:1,:]
        residual1=hidden_states[:,1:2,:]
        hidden_states0=hidden_states[:,0:1,:]
        hidden_states1=hidden_states[:,1:2,:]
        attention_mask0=attention_mask[0:1]
        attention_mask1=attention_mask[1:2]

        ###!!! batch0 norm attn
        # residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output0 = self.input_layernorm(hidden_states0)

        # Self attention.
        attention_output_with_bias0 = self.self_attention(
            input_layernorm_output0,
            attention_mask=attention_mask0,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states0 = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias0, residual0, self.hidden_dropout
            )

        # Residual connection.
        residual0 = hidden_states0

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output0 = self.pre_cross_attn_layernorm(hidden_states0)

        # Cross attention.
        attention_output_with_bias0 = self.cross_attention(
            pre_cross_attn_layernorm_output0,
            attention_mask=context_mask,
            key_value_states=context,
            inference_params=inference_params,
        )

        if isinstance(attention_output_with_bias0, dict) and "context" in attention_output_with_bias0:
            context = attention_output_with_bias0["context"]
            b=0

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states0 = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias0, residual0, self.hidden_dropout
            )

        # Residual connection.
        residual0 = hidden_states0

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output0 = self.pre_mlp_layernorm(hidden_states0)

        
        
        
        
        
        
        #batch0 mlp
        # MLP.
        # mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)
        
        
        probs0, routing_map0 = self.mlp.router(pre_mlp_layernorm_output0)
            
        #all2all dispatch
        (dispatched_input0, tokens_per_expert0 , handle0) = self.mlp.token_dispatcher.token_permutation(
            hidden_states0, probs0, routing_map0
        )
        
        
        
             
        ###!!! batch1 norm attn
        # residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output1 = self.input_layernorm(hidden_states1)

        # Self attention.
        attention_output_with_bias1 = self.self_attention(
            input_layernorm_output1,
            attention_mask=attention_mask1,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states1 = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias1, residual1, self.hidden_dropout
            )

        # Residual connection.
        residual1 = hidden_states1

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output1 = self.pre_cross_attn_layernorm(hidden_states1)

        # Cross attention.
        attention_output_with_bias1 = self.cross_attention(
            pre_cross_attn_layernorm_output1,
            attention_mask=context_mask,
            key_value_states=context,
            inference_params=inference_params,
        )

        if isinstance(attention_output_with_bias1, dict) and "context" in attention_output_with_bias1:
            context = attention_output_with_bias1["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states1 = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias1, residual1, self.hidden_dropout
            )

        # Residual connection.
        residual1 = hidden_states1

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output1 = self.pre_mlp_layernorm(hidden_states1)

        
        
        
        
        
        
        #batch0 mlp
        # MLP.
        # mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)
        
        
        probs1, routing_map1 = self.mlp.router(pre_mlp_layernorm_output1)
            
        #all2all dispatch
        (dispatched_input1, tokens_per_expert1 , handle1) = self.mlp.token_dispatcher.token_permutation(
            hidden_states1, probs1, routing_map1
        )
        
        
        
        
        
        with  torch.cuda.stream(self.commstream_mananger.get_default_comm_stream()):
            handle0.wait()
        torch.cuda.default_stream().wait_stream(self.commstream_mananger.get_default_comm_stream())
        
        
        dispatched_input0=self.mlp.token_dispatcher.post_all2all_token_permutation(dispatched_input0)
        
        expert_output0, mlp_bias0 = self.mlp.experts(dispatched_input0, tokens_per_expert0)
            
        #all2all combine
        output0, mlp_bias0, handle0 = self.mlp.token_dispatcher.token_unpermutation(expert_output0, mlp_bias0)

        
        
        with  torch.cuda.stream(self.commstream_mananger.get_default_comm_stream()):
            handle1.wait()
        torch.cuda.default_stream().wait_stream(self.commstream_mananger.get_default_comm_stream())
   
        
        
        
        
        dispatched_input1=self.mlp.token_dispatcher.post_all2all_token_permutation(dispatched_input1)
        
        
        
        expert_output1, mlp_bias1 = self.mlp.experts(dispatched_input1, tokens_per_expert1)
            
        #all2all combine
        output1, mlp_bias1, handle1 = self.mlp.token_dispatcher.token_unpermutation(expert_output1, mlp_bias1)
        
        
        
        with  torch.cuda.stream(self.commstream_mananger.get_default_comm_stream()):
            handle0.wait()
        torch.cuda.default_stream().wait_stream(self.commstream_mananger.get_default_comm_stream())
      

        output0 = self.mlp.token_dispatcher.post_all2all_token_unpermutation(output0)
        
        
        
        
        
        
        
        
        if self.mlp.use_shared_expert and not self.mlp.shared_expert_overlap:
                # if shared_expert_overlap is True, the expert calculation happens in
                # the token_dispatcher to overlap communications and computations
            output0 = output0 + self.mlp.shared_experts(hidden_states0)
        
        mlp_output_with_bias0=output0, mlp_bias0
        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias0, residual0, self.hidden_dropout
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output0 = make_viewless_tensor(
            inp=hidden_states0, requires_grad=hidden_states0.requires_grad, keep_graph=True
        )

        # CUDA graph requires returned values to be Tensors
    
        
        
        
        
        
        
        
        

        with  torch.cuda.stream(self.commstream_mananger.get_default_comm_stream()):
            handle1.wait()
        torch.cuda.default_stream().wait_stream(self.commstream_mananger.get_default_comm_stream())
     
        
        output1 = self.mlp.token_dispatcher.post_all2all_token_unpermutation(output1)
        
        
        if self.mlp.use_shared_expert and not self.mlp.shared_expert_overlap:
                # if shared_expert_overlap is True, the expert calculation happens in
                # the token_dispatcher to overlap communications and computations
            output1 = output1 + self.mlp.shared_experts(hidden_states1)
        
        mlp_output_with_bias1=output1, mlp_bias1
        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias1, residual1, self.hidden_dropout
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output1 = make_viewless_tensor(
            inp=hidden_states1, requires_grad=hidden_states1.requires_grad, keep_graph=True
        )

        # CUDA graph requires returned values to be Tensors
        output =torch.cat([output0,output1],dim=1)
        if self.config.external_cuda_graph and self.training:
            return output1 #+ output0
        
        
        return output, context

    def forward_bk(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
    ):
        """
        Perform a forward pass through the transformer layer.

        This method implements the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask tensor for cross-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor, optional): Bias tensor for Q * K.T.
            inference_params (object, optional): Parameters for inference-time optimizations.
            packed_seq_params (object, optional): Parameters for packed sequence processing.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                output (Tensor): Transformed hidden states of shape [s, b, h].
                context (Tensor): Updated context tensor if cross-attention is used,
                otherwise None.
        """
        #hidden_states_shape [s,b,h]
        # Residual connection.[s, b,h]
        
        # residual=hidden_states[:,0,:]
        # residual1=hidden_states[:,1,:]
        
        ###!!! batch0 norm attn
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_params=inference_params,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        
        
        
        
        
        
        #batch0 mlp
        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)
        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        # CUDA graph requires returned values to be Tensors
        if self.config.external_cuda_graph and self.training:
            return output
        return output, context

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the transformer layer.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the transformer layer.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict

    def __call__(self, *args, **kwargs):
        # Training and validation mode CUDA graphs
        if hasattr(self, 'cudagraph_manager') and kwargs.get('inference_params') is None:
            return self.cudagraph_manager(self, args, kwargs)
        # Inference mode. CUDA graphs are used in the decode phase only, when attn mask is None
        elif (
            not self.training
            and hasattr(self, 'cudagraph_manager')
            and kwargs.get('inference_params') is not None
            and kwargs['inference_params'].decode_mode
        ):
            assert (
                kwargs.get('attention_mask') is None
            ), f"Attention mask must not be set when using CUDA graphs for decode"
            return self.cudagraph_manager(self, args, kwargs)
        return super(MegatronModule, self).__call__(*args, **kwargs)
