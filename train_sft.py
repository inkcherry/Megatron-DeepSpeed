# coding=utf-8
# copyright (c) 2024 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

from functools import partial
import os
import sys

# adapt from `tasks/main.py`
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

#import multi_device.platform

import torch
from torch import Tensor

from megatron.training import get_args
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core import mpu
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.training.training import pretrain
from megatron.training.utils import get_ltor_masks_and_position_ids, average_losses_across_data_parallel_group

from megatron.core.datasets.supervised_dataset import SupervisedDataset


def model_provider(pre_process=True, post_process=True):
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"
    config = core_transformer_config_from_args(args)
    #assert args.use_mcore_models
    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        if args.num_experts:
             transformer_layer_spec = get_gpt_decoder_block_spec(
                config,
                use_transformer_engine=use_te
            )
        else:
            enable_fused_sdpa = args.use_fused_sdpa or args.use_fused_sdpa_with_recompute
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                                                                args.num_experts,
                                                                args.moe_grouped_gemm,
                                                                args.qk_layernorm,
                                                                args.moe_shared_expert_gate)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(
                                                                args.num_experts,
                                                                args.moe_grouped_gemm,
                                                                args.qk_layernorm,
                                                                args.multi_latent_attention,
                                                                args.normalization,
                                                                enable_fused_sdpa,
                                                                args.moe_shared_expert_gate)

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor, # welm-19b
    )
    return model

def get_batch(data_iterator):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()
    # Items and their type.
    keys = ['input_ids', 'labels']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['input_ids'].long()
    labels_ = data_b['labels'].long()
    labels = labels_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # TODO: handle eod correctly.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        labels, -100, args.reset_position_ids, args.reset_attention_mask, args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    tokenizer = get_tokenizer()

    print_rank_0('> building train, validation, and test datasets ...')

    train_ds = SupervisedDataset(
        args.data_path[0],
        tokenizer,
        args.seq_length)
    valid_ds, test_ds = None, None

    print_rank_0(f"> finished creating datasets ...")
    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step)
