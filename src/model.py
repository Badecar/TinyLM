import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, H:int, emb_dim:int, att_dim:int, max_seq_len:int):
        super().__init__()
        self.att_dim = att_dim
        self.emb_dim = emb_dim
        self.H = H
        self.w_qkv = nn.Linear(emb_dim, 3 * att_dim * H, bias = False) # Since they are initialized the same anyway
        self.wo = nn.Linear(H * att_dim, emb_dim, bias=False)

        # In __init__, for max_seq_len:
        self.register_buffer("causal_mask", 
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1) * float('-inf'))
   

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.w_qkv(x) # (B, context_size, 3*H*att_dim). Utilizes gpu better
        q, k, v = qkv.chunk(3, dim=-1) # 3 * (B, context_size, H*att_dim)

        # (B, context_size, H, att_dim) -> (B, H, context_size, att_dim)
        q_heads = q.view(B, T, self.H, self.att_dim).transpose(1,2)
        k_heads = k.view(B, T, self.H, self.att_dim).transpose(1,2)
        v_heads = v.view(B, T, self.H, self.att_dim).transpose(1,2)

        #manual
        qk = q_heads @ k_heads.transpose(-1, -2) / math.sqrt(self.att_dim) # (B, H, context_size, context_size)
        qk = qk + self.causal_mask[:T, :T]

        attention = F.softmax(qk, dim=-1)
        y = attention @ v_heads # (B, H, context_size, att_dim)

        # wo computation needs: (context_size , h*att_dim) @ (h*att_dim , emb_dim) = (context_size , emb_dim)
        # batched
        y_reshaped = y.transpose(1,2).view(B, T, self.H * self.att_dim)
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
    def __init__(self, H:int, emb_dim:int, att_dim:int, max_seq_len:int):
        super().__init__()
        self.rms_norm1 = RMSNorm(emb_dim=emb_dim)
        self.rms_norm2 = RMSNorm(emb_dim=emb_dim)
        self.mh_attention = OptimizedMultiHeadAttention(H=H, emb_dim=emb_dim, att_dim=att_dim, max_seq_len=max_seq_len)
        self.ff = FeedForward(emb_dim=emb_dim)
    
    def forward(self, x):
        # Pre-norm architecture with residual connections
        x = x + self.mh_attention(self.rms_norm1(x))
        x = x + self.ff(self.rms_norm2(x))
        return x


class TinyLM(nn.Module):
    """A small GPT-style language model."""
    def __init__(
        self,
        vocab_size: int = 50257,  # GPT-2 tokenizer vocab size
        emb_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        att_dim: int = 64,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(max_seq_len, emb_dim)
        
        # Decoder stack
        self.layers = nn.ModuleList([
            Decoder(H=n_heads, emb_dim=emb_dim, att_dim=att_dim, max_seq_len=max_seq_len)
            for _ in range(n_layers)
        ])
        
        # Final layer norm and output projection
        self.ln_f = RMSNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)
        
        # Weight tying: share weights between token embedding and output projection
        self.lm_head.weight = self.tok_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: Token indices of shape (B, T)
        Returns:
            Logits of shape (B, T, vocab_size)
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"
        
        # Create position indices
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        # Embeddings
        tok_emb = self.tok_emb(idx)  # (B, T, emb_dim)
        pos_emb = self.pos_emb(pos)  # (T, emb_dim)
        x = tok_emb + pos_emb
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x)
        
        # Final norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        return logits
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """Generate tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


class CrossEntropyLoss(nn.Module):
    """Cross-entropy loss for language modeling with label smoothing support."""
    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction='mean'
        )
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss for language modeling.
        
        Args:
            logits: Model output of shape (B, T, vocab_size)
            targets: Target token indices of shape (B, T)
        
        Returns:
            Scalar loss value
        """
        B, T, V = logits.shape
        # Reshape for cross-entropy: (B*T, V) and (B*T,)
        logits = logits.view(B * T, V)
        targets = targets.view(B * T)
        return self.loss_fn(logits, targets)


class Trainer:
    """Training loop for TinyLM."""
    def __init__(
        self,
        model: TinyLM,
        train_loader: torch.utils.data.DataLoader,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        betas: tuple = (0.9, 0.95),
        grad_clip: float = 1.0,
        device: str = 'auto',
        warmup_steps: int = 100,
        max_steps: int = 10000,
        log_interval: int = 10,
        eval_interval: int = 500,
        checkpoint_dir: str = 'checkpoints',
    ):
        self.model = model
        self.train_loader = train_loader
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.checkpoint_dir = checkpoint_dir
        
        # Device setup
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        
        # Loss function
        self.loss_fn = CrossEntropyLoss()
        
        # Optimizer with weight decay
        self.optimizer = self._configure_optimizer(learning_rate, weight_decay, betas)
        
        # Learning rate scheduler with warmup
        self.lr_scheduler = self._get_lr_scheduler(learning_rate)
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def _configure_optimizer(self, lr, weight_decay, betas):
        """Configure optimizer with weight decay only on certain parameters."""
        # Separate parameters into those that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.ndim < 2 or 'bias' in name or 'norm' in name.lower():
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        
        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
    
    def _get_lr_scheduler(self, max_lr):
        """Cosine annealing with linear warmup."""
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train(self):
        """Main training loop."""
        self.model.train()
        data_iter = iter(self.train_loader)
        
        running_loss = 0.0
        best_loss = float('inf')
        
        print(f"Training on {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for step in range(self.max_steps):
            # Get batch (with cycling)
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                x, y = next(data_iter)
            
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward pass
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            self.lr_scheduler.step()
            
            running_loss += loss.item()
            
            # Logging
            if (step + 1) % self.log_interval == 0:
                avg_loss = running_loss / self.log_interval
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Step {step + 1}/{self.max_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                running_loss = 0.0
            
            # Checkpointing
            if (step + 1) % self.eval_interval == 0:
                avg_loss = loss.item()  # Use current loss as proxy
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save_checkpoint(step + 1, 'best.pt')
                self.save_checkpoint(step + 1, 'latest.pt')
        
        print("Training complete!")
        return self.model
    
    def save_checkpoint(self, step: int, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['step']


if __name__ == "__main__":
    from data import get_dataloader, OUT_FILE
    
    # Configuration
    CONTEXT_SIZE = 256
    BATCH_SIZE = 32
    
    # Create model
    model = TinyLM(
        vocab_size=50257,
        emb_dim=512,
        n_layers=6,
        n_heads=8,
        att_dim=64,
        max_seq_len=CONTEXT_SIZE,
    )
    
    # Create dataloader using data.py
    train_loader = get_dataloader(
        data_path=OUT_FILE,
        context_size=CONTEXT_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    print(f"DataLoader ready! Total batches: {len(train_loader):,}")
    
    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        learning_rate=3e-4,
        max_steps=10000,
    )
    
    trainer.train()
