"""
Layer Normalization
"""
import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        return out * self.gamma + self.beta

if __name__ == '__main__':
    dim_model = 512
    X = torch.rand(2, 5, dim_model)
    ln = LayerNorm(dim_model)
    print(f"dim_model:{dim_model}")
    print(f"ln_gamma:{ln.gamma}")
    print(f"ln_beta:{ln.beta}")
    ln_result = ln(X)
    print(f"ln_result:{ln_result}")
    print(f"ln_result_shape:{ln_result.shape}")

