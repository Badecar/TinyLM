import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


# class SelfAttention(nn.Module):
#     def __init__(self, emb_dim:int, d_k:int, d_v:int):
#         super().__init__()
#         self.d_k = d_k
#         self.d_v = d_v

#         self.wq = nn.Linear(emb_dim, d_k, bias=False)
#         self.wk = nn.Linear(emb_dim, d_k, bias=False)
#         self.wv = nn.Linear(emb_dim, d_v, bias=False)

#     def forward(self, x):
#         Q = self.wq(x)
#         K = self.wk(x)
#         V = self.wv(x)

#         attention = F.softmax(Q @ K.T / torch.Tensor(self.d_k).sqrt(), dim=1) @ V
#         return attention


# class MultiheadAttention(nn.Module):
#     def __init__(self, H:int, emb_dim:int, d_k:int, d_v:int):
#         super().__init__()
#         self.heads = [SelfAttention(emb_dim=emb_dim, d_k=d_k, d_v=d_v) for _ in range(H)]
#         self.wo = nn.Linear(d_v * H, emb_dim, bias=False)
    
#     def forward(self, x):
#         heads_output = [head(x) for head in self.heads]
#         attention_heads = torch.cat(heads_output)
#         scaled_attention = self.wo(attention_heads)
#         return scaled_attention

class OptimizedMultiHeadAttention(nn.Module):
    def __init__(
        self,
        H: int,
        emb_dim: int,
        att_dim: int,
        max_seq_len: int,
        attention_impl: str = "sdpa",
    ):
        super().__init__()
        self.att_dim = att_dim
        self.emb_dim = emb_dim
        self.H = H
        self.attention_impl = attention_impl
        self.w_qkv = nn.Linear(emb_dim, 3 * att_dim * H, bias=False)
        self.wo = nn.Linear(H * att_dim, emb_dim, bias=False)

        if self.attention_impl == "manual":
            self.register_buffer(
                "causal_mask",
                torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1),
            )

    def forward(self, x):
        B, T, _ = x.shape

        qkv = self.w_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q_heads = q.view(B, T, self.H, self.att_dim).transpose(1, 2)
        k_heads = k.view(B, T, self.H, self.att_dim).transpose(1, 2)
        v_heads = v.view(B, T, self.H, self.att_dim).transpose(1, 2)

        if self.attention_impl == "sdpa":
            y = F.scaled_dot_product_attention(
                q_heads,
                k_heads,
                v_heads,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )
        elif self.attention_impl == "manual":
            qk = q_heads @ k_heads.transpose(-1, -2) / math.sqrt(self.att_dim)
            qk = qk.masked_fill(self.causal_mask[:T, :T], torch.finfo(qk.dtype).min)
            attention = F.softmax(qk, dim=-1)
            y = attention @ v_heads
        else:
            raise ValueError(f"Unknown attention_impl: {self.attention_impl}")

        y_reshaped = y.transpose(1, 2).contiguous().view(B, T, self.H * self.att_dim)
        return self.wo(y_reshaped)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        hidden_dim = int(2 * (4 * emb_dim) / 3)
        self.w1 = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, emb_dim, bias=False)
        self.w3 = nn.Linear(emb_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Decoder(nn.Module):
    def __init__(self, H: int, emb_dim: int, att_dim: int, max_seq_len: int, attention_impl: str):
        super().__init__()
        self.rms_norm1 = RMSNorm(emb_dim=emb_dim)
        self.rms_norm2 = RMSNorm(emb_dim=emb_dim)
        self.mh_attention = OptimizedMultiHeadAttention(
            H=H,
            emb_dim=emb_dim,
            att_dim=att_dim,
            max_seq_len=max_seq_len,
            attention_impl=attention_impl,
        )
        self.ff = FeedForward(emb_dim=emb_dim)

    def forward(self, x):
        x = x + self.mh_attention(self.rms_norm1(x))
        x = x + self.ff(self.rms_norm2(x))
        return x


class TinyLM(nn.Module):
    """A GPT-style language model with configurable attention."""

    def __init__(
        self,
        vocab_size: int = 50257,
        emb_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        att_dim: int = 64,
        max_seq_len: int = 256,
        attention_impl: str = "sdpa",
        use_activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim
        self.use_activation_checkpointing = use_activation_checkpointing

        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(max_seq_len, emb_dim)

        self.layers = nn.ModuleList(
            [
                Decoder(
                    H=n_heads,
                    emb_dim=emb_dim,
                    att_dim=att_dim,
                    max_seq_len=max_seq_len,
                    attention_impl=attention_impl,
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_f = RMSNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)
        self.output_scale = emb_dim**-0.5
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * len(self.layers)) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb

        for layer in self.layers:
            if self.use_activation_checkpointing and self.training:
                x = checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        x = self.ln_f(x)
        return self.lm_head(x) * self.output_scale

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len :]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
