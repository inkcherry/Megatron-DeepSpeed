# coding=utf-8

# The following code has been taken from https://github.com/NVIDIA/NeMo/blob/ \
# 782b4e1652aaa43c8be390d9db0dc89544afa080/nemo/collections/nlp/modules/ \
# common/megatron/rotary_pos_embedding.py

import importlib.util
import torch
from deepspeed import get_accelerator
from torch import einsum, nn

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)).to(get_accelerator().current_device_name())
        self.register_buffer('inv_freq', inv_freq)
        if importlib.util.find_spec('einops') is None:
            raise RuntimeError("einops is required for Rotary Embedding")

    def forward(self, max_seq_len, offset=0):
        seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        #[512 64]
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        #[512,128]
        # emb [seq_length, .., dim]
        from einops import rearrange
        tmp = rearrange(emb, 'n d -> n 1 1 d')
        # print("tmp",tmp.shape)
        return tmp

def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    from einops import rearrange
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    # print("x1,x2,", x.shape,x1.shape,x2.shape)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    # print("t,t_pass",t.shape, t_pass.shape)

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * freqs.cos().to(t.dtype)) + (_rotate_half(t) * freqs.sin().to(t.dtype))
    # print(t.shape)
    if t_pass.shape[-1]==0:
        return t
    # return torch.cat((t, t_pass), dim=-1)
