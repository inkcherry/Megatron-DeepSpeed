import torch
import math
import re
from functools import partial
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import get_timers, get_args, get_retro_args, core
from megatron.core import parallel_state, tensor_parallel, mpu
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel, GPTModelPipe
from megatron.model.rotary_pos_embedding import apply_rotary_pos_emb, RotaryEmbedding
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.utils import average_losses_across_data_parallel_group
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module
from megatron.arguments import core_transformer_config_from_args
from megatron.initialize import initialize_megatron
from megatron.optimizer import get_megatron_optimizer
from megatron.checkpointing import save_checkpoint
from megatron.training import get_optimizer_param_scheduler

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator.real_accelerator import get_accelerator
import os
import subprocess

from torch import nn
import torch.nn.functional as F

def add_extra_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='tadd')
    group.add_argument("--llama-fs-ckpt-num-of-shards",
                       type=int,
                       help='num of llama ckpt.')
    group.add_argument("--model-type", 
                        type=str,
                        default="llama",
                        help="select the type of model")
    group.add_argument("--origin-hf-ckpt-dir", 
                        type=str,
                        default="",
                        help="the original path of the llama-hf ckpt")
    return parser

def calc_start_and_end_index(hidden_size, local_rank, tp_parallel_size):
    per_partition_size = core.utils.divide(hidden_size, tp_parallel_size)
    start_index = local_rank * per_partition_size
    end_index = start_index + per_partition_size
    return per_partition_size, start_index, end_index



def set_transfer_flag(mega_ds_weight,hf_weight):
    setattr(mega_ds_weight,"transfer_hf2megads_weight",True)
    setattr(hf_weight,"transfer_hf2megads_weight",True)
    
    a=0

err_weight_list=[]
def check_transfer(mega_ds_or_hf_weight):
    if not hasattr(mega_ds_or_hf_weight, "transfer_hf2megads_weight"):
        err_weight_list.append(mega_ds_or_hf_weight)
        



def convert_hf_to_mega_ds(pre_process=True, post_process=True):
    """Build the model."""
    args = get_args()
    print_rank_0(f'building {args.model_type} model ...')
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
            model = GPTModel(
                config,
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )
    see_memory_usage(f"After Building Model", force=True)
    if torch.distributed.get_rank() < 2:
        print(f"{torch.distributed.get_rank()} {model}")


    tokenizer = get_tokenizer()
    hf_ckpt_dir = args.origin_hf_ckpt_dir
    hfnl = args.num_layers + 2
    llama_fs_ckpt_num_of_shards = args.llama_fs_ckpt_num_of_shards
    # token_vocab = args.vocab_size
    token_vocab =32000

    padded_vocab_size = args.padded_vocab_size
    more_padded = padded_vocab_size - token_vocab
    reduce_num = 2

    # filter params when the model is large
    valid_name = {}
    if args.deepspeed and not args.no_pipeline_parallel and args.num_layers >= 40:
        decoder_pat = re.compile("(\d+)\.(.+)")
        for pname, p in model.named_parameters():
            print_rank_0(f"ppnnnaammme {pname}")
            if torch.distributed.get_rank() < 2:
                print(f"{torch.distributed.get_rank()} {pname}")
            if pname == "1.word_embeddings.weight":
            # if pname == "'tied_modules.embed.word_embeddings.weight":
                valid_name["model.embed_tokens.weight"] = 1
            elif pname == f"{hfnl + 1}.weight":
                valid_name["model.norm.weight"] = 1
            elif pname == f"{hfnl + 2 }.lm_head.weight":
                valid_name["lm_head.weight"] = 1
            else:
                mobj = decoder_pat.match(pname)
                
                layer_i = int(mobj.group(1))
                subname = mobj.group(2)

                if subname in ["self_attention.query_key_value.weight"]:
                        valid_name[
                            f"model.layers.{layer_i - reduce_num}.self_attn.q_proj.weight"] = 1
                        valid_name[
                            f"model.layers.{layer_i - reduce_num}.self_attn.k_proj.weight"] = 1
                        valid_name[
                            f"model.layers.{layer_i - reduce_num}.self_attn.v_proj.weight"] = 1
                elif subname in ["mlp.dense_h_to_4h.weight"]:
                    valid_name[f"model.layers.{layer_i - reduce_num}.mlp.gate_proj.weight"] = 1
                    valid_name[f"model.layers.{layer_i - reduce_num}.mlp.up_proj.weight"] = 1
                elif subname in ["self_attention.dense.weight"]:
                    valid_name[f"model.layers.{layer_i - reduce_num}.self_attn.o_proj.weight"] = 1
                elif subname in ["mlp.dense_4h_to_h.weight"]:
                    valid_name[f"model.layers.{layer_i - reduce_num}.mlp.down_proj.weight"] = 1
                else:
                    valid_name[f"model.layers.{layer_i - reduce_num}.{subname}"] = 1

    loaded = {}
    miss_count = 0
    hit_count = 0

    if args.deepspeed and not args.no_pipeline_parallel and args.num_layers >= 40:
        for hfli in range(1, llama_fs_ckpt_num_of_shards + 1):
            d = torch.load(
                f"{hf_ckpt_dir}/pytorch_model-{hfli:05d}-of-{llama_fs_ckpt_num_of_shards:05d}.bin",
                map_location=torch.device('cpu')
            )
            for k in d:
                assert k not in loaded
                if k in valid_name.keys() and valid_name[k] == 1:
                    hit_count += 1
                    loaded[k] = d[k].clone()
                else:
                    miss_count += 1
            del d
    else:
        for hfli in range(1, llama_fs_ckpt_num_of_shards + 1):
            d = torch.load(
                f"{hf_ckpt_dir}/pytorch_model-{hfli:05d}-of-{llama_fs_ckpt_num_of_shards:05d}.bin",
                map_location=torch.device('cpu')
            )
            for k in d:
                print_rank_0(k)
                assert k not in loaded
                loaded[k] = d[k].clone()
            del d

    tp_parallel_size = mpu.get_tensor_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()

    print(
        f"tensor_parallel_size {tp_parallel_size} {tp_rank} {torch.distributed.get_rank()}"
    )

    if args.deepspeed and not args.no_pipeline_parallel:
        decoder_pat = re.compile("(\d+)\.(.+)")
        # print(model.named_parameters())

        
        # for pname,p in model.named_parameters():
        #     print(pname)
        # exit()
        for pname, p in model.named_parameters():
            print(f"{pname}, {p.shape}, {args.local_rank}")
            # if "position_embeddings.weight" in pname:
            #     print("pass", pname)
            #     continue
            # if "34.weight" ==pname:
            #     print("pass " ,pname)
            #     continue
            if "word_embeddings.weight" in pname:
            # if pname == "'tied_modules.embed.word_embeddings.weight":

                w = loaded["model.embed_tokens.weight"]
                assert w.shape[0] == token_vocab
                per_partition_vocab_size, start_index, end_index = calc_start_and_end_index(
                    padded_vocab_size, tp_rank, tp_parallel_size)
                end_index = min(end_index, token_vocab)
                real_partition_vocab_size = end_index - start_index

                new_w = torch.zeros((per_partition_vocab_size, w.shape[1]),
                                    dtype=w.dtype)
                new_w[:real_partition_vocab_size, :] = w[
                    start_index:end_index, :]
                if tp_rank == tp_parallel_size - 1:
                    new_w[-more_padded:] = w[:token_vocab].mean(dim=0,
                                                                keepdim=True)
                set_transfer_flag(p,new_w)
                p.data.copy_(new_w)

            elif pname == f"{hfnl + 0}.weight":
                w = loaded["model.norm.weight"]
                set_transfer_flag(p,new_w)

                p.data.copy_(w)

            elif pname == f"{hfnl + 1 }.lm_head.weight":
                w = loaded["lm_head.weight"]
                assert w.shape[0] == token_vocab

                per_partition_vocab_size, start_index, end_index = calc_start_and_end_index(
                    padded_vocab_size, tp_rank, tp_parallel_size)
                end_index = min(end_index, token_vocab)
                real_partition_vocab_size = end_index - start_index

                new_w = torch.zeros((per_partition_vocab_size, w.shape[1]),
                                    dtype=w.dtype)
                new_w[:real_partition_vocab_size, :] = w[
                    start_index:end_index, :]
                if tp_rank == tp_parallel_size - 1:
                    new_w[-more_padded:] = w[:token_vocab].mean(dim=0,
                                                                keepdim=True)
                set_transfer_flag(p,new_w)

                p.data.copy_(new_w)
            else:
                mobj = decoder_pat.match(pname)
                layer_i = int(mobj.group(1))
                subname = mobj.group(2)

                if subname in ["self_attention.query_key_value.weight"]:
                    print(f"group 3 {layer_i} {subname}")

                    wq = loaded[
                        f"model.layers.{layer_i - reduce_num}.self_attn.q_proj.weight"]
                    wk = loaded[
                        f"model.layers.{layer_i - reduce_num}.self_attn.k_proj.weight"]
                    wv = loaded[
                        f"model.layers.{layer_i - reduce_num}.self_attn.v_proj.weight"]

                    hidden_size = wq.shape[0]
                    per_partition_size, start_index, end_index = calc_start_and_end_index(
                        hidden_size, tp_rank, tp_parallel_size)
                    hidden_size_per_attention_head = core.utils.divide(
                        hidden_size, config.num_attention_heads)
                    num_attention_heads_per_partition = core.utils.divide(
                        config.num_attention_heads, tp_parallel_size)

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
                    set_transfer_flag(p,new_w)
                    p.data.copy_(new_w)

                elif subname in ["mlp.dense_h_to_4h.weight"]:
                    w_gate = loaded[f"model.layers.{layer_i - reduce_num}.mlp.gate_proj.weight"]
                    w_up = loaded[f"model.layers.{layer_i - reduce_num}.mlp.up_proj.weight"]

                    hidden_size = w_gate.shape[0]
                    per_partition_size, start_index, end_index = calc_start_and_end_index(
                        hidden_size, tp_rank, tp_parallel_size)
                    new_w = torch.zeros((per_partition_size * 2, w_gate.shape[1]),
                                        dtype=w_gate.dtype)
                    new_w[:per_partition_size * 2, :] = \
                            torch.cat([
                                w_gate[start_index:end_index, :],
                                w_up[start_index:end_index, :]
                            ], dim=0)
                    set_transfer_flag(p,new_w)
                    p.data.copy_(new_w)

                elif subname in ["self_attention.dense.weight", "mlp.dense_4h_to_h.weight"]:
                    if subname == "self_attention.dense.weight":
                        w = loaded[f"model.layers.{layer_i - reduce_num}.self_attn.o_proj.weight"]
                    else:
                        w = loaded[f"model.layers.{layer_i - reduce_num}.mlp.down_proj.weight"]

                    hidden_size = w.shape[1]
                    per_partition_size, start_index, end_index = calc_start_and_end_index(
                        hidden_size, tp_rank, tp_parallel_size)
                    new_w = torch.zeros((w.shape[0], per_partition_size),
                                        dtype=w.dtype)
                    new_w[:, : per_partition_size] = w[:, start_index: end_index]
                    set_transfer_flag(p,new_w)
                    p.data.copy_(new_w)


                elif subname in ["mlp.dense_h_to_4h1.weight", "mlp.dense_h_to_4h2.weight"]:
                    if subname == "mlp.dense_h_to_4h1.weight":
                        w = loaded[f"model.layers.{layer_i - reduce_num}.mlp.gate_proj.weight"]
                    else:
                        w = loaded[f"model.layers.{layer_i - reduce_num}.mlp.up_proj.weight"]

                    hidden_size = w.shape[0]
                    per_partition_size, start_index, end_index = calc_start_and_end_index(
                        hidden_size, tp_rank, tp_parallel_size)
                    new_w = torch.zeros((per_partition_size, w.shape[1]),
                                        dtype=w.dtype)

                    new_w[:per_partition_size, :] = w[start_index:end_index, :]
                    set_transfer_flag(p,new_w)
                    p.data.copy_(new_w)


                else:
                    # if "bias" in subname:
                    # # if("input_layernorm.bias" in subname or "query_key_value.bias" in subname or "dense.bias" in subname):
                    #     print(f"pass  model.layers.{layer_i - reduce_num}.{subname}")
                    #     continue
                    new_w = loaded[f"model.layers.{layer_i - reduce_num}.{subname}"]
                    set_transfer_flag(p,new_w)
                    p.data.copy_(new_w)

    else:
        raise ImportError("no implement")

    
    
    # for i in loaded:
    #     check_transfer(i)
    for pname, p in model.named_parameters():
        check_transfer(p)
    # necessary
    
    if len(err_weight_list)>0:
        print("--------------------")
        print("--------------------")
        print("--------------------")
        print("--------------------")
        print(err_weight_list)
        print("error")
        exit()
    del loaded

    unwrapped_model = unwrap_model([model],
                                   (torchDDP, LocalDDP, Float16Module))
    optimizer = get_megatron_optimizer(unwrapped_model)
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    print(f"before deepspeed init")
    ds_engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=opt_param_scheduler,
        mpu=mpu if args.no_pipeline_parallel else None)
    print(f"after deepspeed init")

    print(f"tp ckpt dir will be saved in {args.save}")
    save_checkpoint(0, [ds_engine], optimizer, opt_param_scheduler)
    print(f"save checkpoint finished")

if __name__ == "__main__":
    # To skip mpi_discovery
    world_size = os.getenv('OMPI_COMM_WORLD_SIZE', None)
    if world_size is not None:
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=add_extra_args,
                        args_defaults={})
    
    convert_hf_to_mega_ds()
