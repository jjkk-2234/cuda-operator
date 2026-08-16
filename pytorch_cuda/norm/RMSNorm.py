import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, embed_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_size))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

if __name__ == "__main__":
    x = torch.randn(2, 4, 3)
    embed_size = x.size(-1)
    eps = 1e-6
    official = nn.RMSNorm(embed_size, eps=eps)
    my = RMSNorm(embed_size, eps=eps)

    my.weight.data = official.weight.data.clone()

    diff = (official(x) - my(x)).abs().max().item()
    print(f"Max difference: {diff:.2e}")
