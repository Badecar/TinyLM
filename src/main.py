import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention:
    def __init__(self, emb_dim=512, d_k=768, d_v=64):
        self.d_k = d_k
        self.d_v = d_v

        self.wq = nn.Linear(emb_dim, d_k, bias=False)
        self.wk = nn.Linear(emb_dim, d_k, bias=False)
        self.wv = nn.Linear(emb_dim, d_v, bias=False)

    def forward(self, x):
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)

        attention = nn.Softmax(Q @ K.T / torch.Tensor(self.d_k).sqrt()) @ V
        return attention


class MultiheadAttention:
    def __init__(self, H=12, emb_dim=512, d_v=512):
        self.heads = [SelfAttention() for _ in range(H)]
        self.wo = nn.Linear(d_v*H, emb_dim)
    
    def forward(self, x):
        heads_output = [head(x) for head in self.heads]
        attention_heads = torch.cat(heads_output)
        scaled_attention = self.wo(attention_heads)
        return scaled_attention


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # This is 'g_i' from your image
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


class Decoder:
    def __init__(self, ):
        


# if __name__ == "__main__":
    
