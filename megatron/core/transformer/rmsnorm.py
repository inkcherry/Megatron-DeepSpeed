import torch
from torch.nn import init
from torch.nn.parameter import Parameter
from megatron.core.transformer import TransformerConfig


class RMSNorm(torch.nn.Module):
    def __init__(self, config: TransformerConfig, hidden_size, eps=1e-5):
        super().__init__()
        self.config = config
        self.epsilon = eps
        self.weight = Parameter(torch.Tensor(hidden_size))
        init.ones_(self.weight)

        # set sequence parallelism flag on weight
        setattr(self.weight, 'sequence_parallel', config.sequence_parallel)

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        norm = torch.mean(x ** 2, -1, keepdim=True)
        norm = x.mul(norm.add_(self.epsilon).rsqrt_())
        norm = norm.to(dtype)
        return norm * self.weight
