import torch
import re
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron import print_rank_0,get_tokenizer,get_args
from megatron.core import  mpu
from megatron.core.utils import divide
from megatron.model import  GPTModelPipe,Float16Module
from megatron.utils import  unwrap_model
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.arguments import core_transformer_config_from_args
from megatron.initialize import initialize_megatron
from megatron.optimizer import get_megatron_optimizer
from megatron.checkpointing import save_checkpoint
from megatron.training import get_optimizer_param_scheduler
from deepspeed.runtime.utils import see_memory_usage
import deepspeed

def add_extra_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='hf2mega')
    group.add_argument("--hf-ckpt-num-shards",
                       type=int,
                       help='num of llama ckpt.')
    group.add_argument("--origin-hf-ckpt-dir", 
                        type=str,
                        default="",
                        help="the original path of the llama-hf ckpt")
    return parser

def compute_partition_range(hidden_size, local_rank, tp_size):
    partition_size = divide(hidden_size,tp_size)
    start_index = local_rank * partition_size
    end_index = start_index + partition_size
    return partition_size, start_index, end_index

def load_and_print_hf_weight(hf_ckpt_dir,hf_ckpt_num_of_shards):
    # Optimization point: We can selectively load specific 'shared' data to reduce CPU memory usage.
    loaded = {}
    print_rank_0(f"----------------------------hf weight list----------------------------")

    for wid in range(1, hf_ckpt_num_of_shards + 1):
        d = torch.load(
            f"{hf_ckpt_dir}/pytorch_model-{wid:05d}-of-{hf_ckpt_num_of_shards:05d}.bin",
            map_location=torch.device('cpu')
        )
        for k in d:
            print_rank_0(k)
            assert k not in loaded
            loaded[k] = d[k].clone()
    del d
    return loaded
    
def print_distinct_weights(model):
    print_rank_0(f"----------------------------mega-ds weight list----------------------------")
    for pipe_rank in range(mpu.get_pipeline_model_parallel_world_size()):        
        if mpu.get_pipeline_model_parallel_rank()==pipe_rank:
            if mpu.get_data_parallel_rank()==0 and mpu.get_tensor_model_parallel_rank()==0:
                for pname, p in model.named_parameters():
                    print(pname)
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
def refactor_and_record_weight(model,loaded,args,config):
    tokenizer = get_tokenizer()
    
    # align layer number
    offset_num = 2
    mega_emb_wnum = 1
    mega_norm_wnum = args.num_layers + 2
    mega_lm_head_wnum = mega_norm_wnum + 1
    token_vocab = tokenizer.vocab_size
    padded_vocab_size = args.padded_vocab_size
    more_padded = padded_vocab_size - token_vocab
    tp_size = mpu.get_tensor_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    decoder_pat = re.compile("(\d+)\.(.+)")
    refactor_weight_list = []
    for pname, p in model.named_parameters():
        # refactor weight
        if pname == f"{mega_emb_wnum}.word_embeddings.weight":
            hf_name = "model.embed_tokens.weight"
            hf_w = loaded[hf_name]
            assert hf_w.shape[0] == token_vocab
            per_partition_vocab_size, start_index, end_index = compute_partition_range(
                padded_vocab_size, tp_rank, tp_size)
            end_index = min(end_index, token_vocab)
            real_partition_vocab_size = end_index - start_index

            new_w = torch.zeros((per_partition_vocab_size, hf_w.shape[1]),
                                dtype=hf_w.dtype)
            new_w[:real_partition_vocab_size, :] = hf_w[
                start_index:end_index, :]
            if tp_rank == tp_size - 1:
                new_w[-more_padded:] = hf_w[:token_vocab].mean(dim=0,
                                                            keepdim=True)
            
            refactor_weight_list.append(f"mega-ds:{pname,p.data.shape}<--hf{hf_name,}  [{start_index}:{end_index},:]  of {hf_w.shape}")
            p.data.copy_(new_w)

        elif pname == f"{mega_norm_wnum}.weight":
            hf_w = loaded["model.norm.weight"]
            refactor_weight_list.append(f"mega-ds:{pname,p.data.shape}<--hf{hf_name,}  of {hf_w.shape}")
            p.data.copy_(hf_w)

        elif pname == f"{mega_lm_head_wnum}.lm_head.weight":
            hf_name = "lm_head.weight"
            hf_w = loaded[hf_name]
            assert hf_w.shape[0] == token_vocab

            per_partition_vocab_size, start_index, end_index = compute_partition_range(
                padded_vocab_size, tp_rank, tp_size)
            end_index = min(end_index, token_vocab)
            real_partition_vocab_size = end_index - start_index

            new_w = torch.zeros((per_partition_vocab_size, hf_w.shape[1]),
                                dtype=hf_w.dtype)
            new_w[:real_partition_vocab_size, :] = hf_w[
                start_index:end_index, :]
            if tp_rank == tp_size - 1:
                new_w[-more_padded:] = hf_w[:token_vocab].mean(dim=0,
                                                            keepdim=True)

            refactor_weight_list.append(f"mega-ds:{pname,p.data.shape}<--hf{hf_name,}  [{start_index}:{end_index},:]  of {hf_w.shape}")

            p.data.copy_(new_w)
        else:
            mobj = decoder_pat.match(pname)
            layer_i = int(mobj.group(1))
            subname = mobj.group(2)

            if subname in ["self_attention.query_key_value.weight"]:
                hf_wq_name =  f"model.layers.{layer_i - offset_num}.self_attn.q_proj.weight"
                hf_wk_name = f"model.layers.{layer_i - offset_num}.self_attn.k_proj.weight"
                hf_wv_name =f"model.layers.{layer_i - offset_num}.self_attn.v_proj.weight"
                wq = loaded[hf_wq_name]
                wk = loaded[hf_wk_name]
                wv = loaded[hf_wv_name]

                hidden_size = wq.shape[0]
                per_partition_size, start_index, end_index = compute_partition_range(
                    hidden_size, tp_rank, tp_size)
                hidden_size_per_attention_head = divide(
                    hidden_size, config.num_attention_heads)
                num_attention_heads_per_partition = divide(
                    config.num_attention_heads, tp_size)

                new_w = torch.zeros((per_partition_size * 3, wq.shape[1]),
                                    dtype=wq.dtype)

                for i in range(num_attention_heads_per_partition):
                    current_index = start_index + i * hidden_size_per_attention_head
                    next_index = current_index + hidden_size_per_attention_head
                    new_w_index = i * (3 * hidden_size_per_attention_head)
                    new_w[new_w_index: new_w_index + (3 * hidden_size_per_attention_head), :] = \
                        torch.cat([
                            wq[current_index: next_index, :],
                            wk[current_index: next_index, :],
                            wv[current_index: next_index, :]
                        ], dim=0)
                refactor_weight_list.append(f"mega-ds:{pname,p.data.shape}<--hf{hf_wq_name,hf_wk_name,hf_wv_name,}  cat q,k,v [{current_index}:{next_index},:]  of q,k,v{wq.shape}")

                p.data.copy_(new_w)

            elif subname in ["mlp.dense_h_to_4h.weight"]:
                hf_w_gate_name =f"model.layers.{layer_i - offset_num}.mlp.gate_proj.weight"
                hf_w_up_name=f"model.layers.{layer_i - offset_num}.mlp.up_proj.weight"
                w_gate = loaded[hf_w_gate_name]
                w_up = loaded[hf_w_up_name]

                hidden_size = w_gate.shape[0]
                per_partition_size, start_index, end_index = compute_partition_range(
                    hidden_size, tp_rank, tp_size)
                new_w = torch.zeros((per_partition_size * 2, w_gate.shape[1]),
                                    dtype=w_gate.dtype)
                new_w[:per_partition_size * 2, :] = \
                        torch.cat([
                            w_gate[start_index:end_index, :],
                            w_up[start_index:end_index, :]
                        ], dim=0)
                refactor_weight_list.append(f"mega-ds:{pname,p.data.shape}<--hf{hf_w_gate_name,hf_w_up_name}  cat gate,up [{start_index}:{end_index},:]  of gate,up{w_gate.shape}")
                p.data.copy_(new_w)

            elif subname in ["self_attention.dense.weight", "mlp.dense_4h_to_h.weight"]:
                if subname == "self_attention.dense.weight":
                    hf_name = f"model.layers.{layer_i - offset_num}.self_attn.o_proj.weight"
                else:
                    hf_name = f"model.layers.{layer_i - offset_num}.mlp.down_proj.weight"

                hf_w = loaded[hf_name]
                hidden_size = hf_w.shape[1]
                per_partition_size, start_index, end_index = compute_partition_range(
                    hidden_size, tp_rank, tp_size)
                new_w = torch.zeros((hf_w.shape[0], per_partition_size),
                                    dtype=hf_w.dtype)
                new_w[:, : per_partition_size] = hf_w[:, start_index: end_index]
                refactor_weight_list.append(f"mega-ds:{pname,p.data.shape}<--hf{hf_name,}  [:,{start_index}:{end_index}]  of {hf_w.shape}")
                p.data.copy_(new_w)
                


            elif subname in ["mlp.dense_h_to_4h1.weight", "mlp.dense_h_to_4h2.weight"]:
                if subname == "mlp.dense_h_to_4h1.weight":
                    hf_name = f"model.layers.{layer_i - offset_num}.mlp.gate_proj.weight"
                   
                else:
                    hf_name = f"model.layers.{layer_i - offset_num}.mlp.up_proj.weight"
                hf_w=loaded[hf_name]
                hidden_size = hf_w.shape[0]
                per_partition_size, start_index, end_index = compute_partition_range(
                    hidden_size, tp_rank, tp_size)
                new_w = torch.zeros((per_partition_size, hf_w.shape[1]),
                                    dtype=hf_w.dtype)

                new_w[:per_partition_size, :] = hf_w[start_index:end_index, :]
                refactor_weight_list.append(f"mega-ds:{pname,p.data.shape}<--hf{hf_name,}  [{start_index}:{end_index},:]  of {hf_w.shape}")

                p.data.copy_(new_w)


            elif subname in ["input_layernorm.weight","post_attention_layernorm.weight"]:
                hf_name = f"model.layers.{layer_i - offset_num}.{subname}"
                new_w = hf_w =loaded[hf_name]
                refactor_weight_list.append(f"mega-ds:{pname,p.data.shape}<--hf{hf_name,}  {hf_w.shape}")

                p.data.copy_(new_w)
            else:
                raise ValueError("Unrecognized weight type")
    return refactor_weight_list
    
def convert_hf_to_mega_ds():
    """Build the model."""
    args = get_args()
    print_rank_0(f'building model ...')
    see_memory_usage(f"Before Building Model", force=True)

    config = core_transformer_config_from_args(args)
    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
        if args.deepspeed and not args.no_pipeline_parallel:
            model = GPTModelPipe(
                config,
                num_tokentypes=0,
                parallel_output=True
            )
        else:
            raise NotImplementedError("Not implemented")
        
    see_memory_usage(f"After Building Model", force=True)
    if torch.distributed.get_rank() < 2:
        print(f"{torch.distributed.get_rank()} {model}")

    # load and initialize HF weight dict
    # print hf weights list & mega-ds weights list 
    hf_ckpt_dir = args.origin_hf_ckpt_dir
    hf_ckpt_num_of_shards = args.hf_ckpt_num_shards
    loaded=load_and_print_hf_weight(hf_ckpt_dir,hf_ckpt_num_of_shards)
    print_distinct_weights(model)

    # refactor weight from hf to mega-ds
    refactor_record = refactor_and_record_weight(model,loaded,args,config)
    print_rank_0(f"----------------------------mapping list----------------------------")
    # print DP0 TP0  records.
    for pipe_rank in range(mpu.get_pipeline_model_parallel_world_size()):        
        if mpu.get_pipeline_model_parallel_rank()==pipe_rank:
            if mpu.get_data_parallel_rank()==0 and mpu.get_tensor_model_parallel_rank()==0:
                for record in refactor_record:
                    print(record)
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
    del loaded

    unwrapped_model = unwrap_model([model],
                                   (torchDDP, LocalDDP, Float16Module))
    optimizer = get_megatron_optimizer(unwrapped_model)
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)


    #init model and save
    print_rank_0(f"before deepspeed init")
    ds_engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=opt_param_scheduler,
        mpu=mpu if args.no_pipeline_parallel else None)
    print_rank_0(f"after deepspeed init")

    print_rank_0(f"mega-ds checkpoint will be saved in {args.save}")
    save_checkpoint(0, [ds_engine], optimizer, opt_param_scheduler)
    print_rank_0(f"save checkpoint completed")

if __name__ == "__main__":

    initialize_megatron(extra_args_provider=add_extra_args)
    convert_hf_to_mega_ds()
