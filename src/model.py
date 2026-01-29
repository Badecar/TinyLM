import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, emb_dim:int, d_k:int, d_v:int):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v

        self.wq = nn.Linear(emb_dim, d_k, bias=False)
        self.wk = nn.Linear(emb_dim, d_k, bias=False)
        self.wv = nn.Linear(emb_dim, d_v, bias=False)

    def forward(self, x):
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)

        attention = F.softmax(Q @ K.T / torch.Tensor(self.d_k).sqrt(), dim=1) @ V
        return attention


class MultiheadAttention(nn.Module):
    def __init__(self, H:int, emb_dim:int, d_k:int, d_v:int):
        super().__init__()
        self.heads = [SelfAttention(emb_dim=emb_dim, d_k=d_k, d_v=d_v) for _ in range(H)]
        self.wo = nn.Linear(d_v * H, emb_dim, bias=False)
    
    def forward(self, x):
        heads_output = [head(x) for head in self.heads]
        attention_heads = torch.cat(heads_output)
        scaled_attention = self.wo(attention_heads)
        return scaled_attention


class OptimizedMultiHeadAttention(nn.Module):
    def __init__(self, H:int, emb_dim:int, att_dim:int):
        super().__init__()
        self.att_dim = att_dim
        self.emb_dim = emb_dim
        self.H = H
        self.w_qkv = nn.Linear(emb_dim, 3 * att_dim * H, bias = False) # Since they are initialized the same anyway
        self.wo = nn.Linear(H * att_dim, emb_dim, bias=False)

    def forward(self, x):
        B, context_size, C = x.shape

        qkv = self.w_qkv(x) # (B, context_size, 3*H*att_dim). Utilizes gpu better
        q, k, v = qkv.chunk(3, dim=-1) # 3 * (B, context_size, H*att_dim)

        # (B, context_size, H, att_dim) -> (B, H, context_size, att_dim)
        q_heads = q.view(B, context_size, self.H, self.att_dim).transpose(1,2)
        k_heads = k.view(B, context_size, self.H, self.att_dim).transpose(1,2)
        v_heads = v.view(B, context_size, self.H, self.att_dim).transpose(1,2)

        #manual
        qk = q_heads @ k_heads.transpose(-1, -2) / math.sqrt(self.att_dim) # (B, H, context_size, context_size)
        ###masking?
        attention = F.softmax(qk, dim=-1)
        y = attention @ v_heads # (B, H, context_size, att_dim)

        # wo computation needs: (context_size , h*att_dim) @ (h*att_dim , emb_dim) = (context_size , emb_dim)
        # batched
        y_reshaped = y.transpose(1,2).view(B, context_size, self.H * self.att_dim)
        mh_attention = self.wo(y_reshaped)
        return mh_attention

class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # This is 'g_i' from your image
        self.weight = nn.Parameter(torch.ones(emb_dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
    
class FeedForward(nn.Module):
    def __init__(self, emb_dim:int):
        super().__init__()
        self.emb_dim = emb_dim
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, emb_dim*4, bias=False), # why no bias??
            nn.SiLU(),
            nn.Linear(emb_dim*4, emb_dim*4, bias=False),
            nn.SiLU(),
            nn.Linear(emb_dim*4, emb_dim, bias=False),
            nn.SiLU()
        )
    
    def forward(self, x):
        return self.ff(x)

class Decoder(nn.Module):
    def __init__(self, H=12, emb_dim=512, d_k=64, d_v=512):
        super().__init__()
        self.rms_norm = RMSNorm(emb_dim=emb_dim)
        self.mh_attention = MultiheadAttention(H=H, emb_dim=emb_dim, d_k=d_k, d_v=d_v)
        self.ff = FeedForward(emb_dim=emb_dim)
    
    def forward(self, x):
        
    


# if __name__ == "__main__":
    
