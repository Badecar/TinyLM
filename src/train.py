import argparse
import math
import os
import re

import torch
import torch.nn as nn
import tiktoken
import wandb

from data import get_dataloader, OUT_FILE
from model import TinyLM


class CrossEntropyLoss(nn.Module):
    """Cross-entropy loss for language modeling with label smoothing support."""
    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction="mean",
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


def _is_version_dir(path: str) -> bool:
    return re.fullmatch(r"v\d+", os.path.basename(path)) is not None


def _get_latest_version_dir(base_dir: str) -> str | None:
    if not os.path.isdir(base_dir):
        return None
    candidates = []
    for name in os.listdir(base_dir):
        match = re.fullmatch(r"v(\d+)", name)
        if match:
            candidates.append((int(match.group(1)), os.path.join(base_dir, name)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _resolve_checkpoint_dir(checkpoint_dir: str, resume_latest: bool, resume_from: str | None) -> str:
    checkpoint_dir = os.path.expanduser(checkpoint_dir)
    if resume_from:
        return os.path.dirname(os.path.expanduser(resume_from))
    if resume_latest:
        if _is_version_dir(checkpoint_dir):
            return checkpoint_dir
        latest_dir = _get_latest_version_dir(checkpoint_dir)
        return latest_dir or checkpoint_dir
    if _is_version_dir(checkpoint_dir):
        return checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_dir = _get_latest_version_dir(checkpoint_dir)
    next_version = 1
    if latest_dir is not None:
        next_version = int(os.path.basename(latest_dir)[1:]) + 1
    return os.path.join(checkpoint_dir, f"v{next_version}")


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
        device: str = "auto",
        warmup_steps: int = 100,
        max_steps: int = 10000,
        log_interval: int = 10,
        eval_interval: int = 500,
        checkpoint_dir: str = "checkpoints",
        verbose: bool = False,
        generation_interval: int = 100,
        enable_tf32: bool = True,
        use_wandb: bool = True,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        best_wandb_interval: int = 2500,
        resume_from: str | None = None,
        resume_latest: bool = False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.checkpoint_dir = os.path.expanduser(checkpoint_dir)
        self.verbose = verbose
        self.generation_interval = generation_interval
        self.use_wandb = use_wandb
        self.best_wandb_interval = best_wandb_interval
        self.start_step = 0

        # Device setup (must be before torch.compile)
        if device == "auto":
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = device

        if enable_tf32 and self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

        self.model = self.model.to(self.device)

        # Loss function
        self.loss_fn = CrossEntropyLoss()

        # Optimizer with weight decay
        self.optimizer = self._configure_optimizer(learning_rate, weight_decay, betas)

        # Learning rate scheduler with warmup
        self.lr_scheduler = self._get_lr_scheduler(learning_rate)

        # Tokenizer for text generation
        self.enable_generation = self.verbose or self.use_wandb
        if self.enable_generation:
            self.tokenizer = tiktoken.get_encoding("gpt2")

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.use_wandb:
            project = wandb_project or os.getenv("WANDB_PROJECT", "TinyLM")
            entity = wandb_entity or os.getenv(
                "WANDB_ENTITY", "badecar-danmarks-tekniske-universitet-dtu"
            )
            self.wandb_run = wandb.init(
                project=project,
                entity=entity,
                config={
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "betas": betas,
                    "grad_clip": grad_clip,
                    "warmup_steps": warmup_steps,
                    "max_steps": max_steps,
                    "log_interval": log_interval,
                    "eval_interval": eval_interval,
                    "generation_interval": generation_interval,
                    "best_wandb_interval": best_wandb_interval,
                    "batch_size": getattr(train_loader, "batch_size", None),
                    "device": self.device,
                },
                settings=wandb.Settings(console="wrap"),
            )
            wandb.watch(self.model, log="gradients", log_freq=self.log_interval)
        else:
            self.wandb_run = None

        if resume_latest or resume_from:
            checkpoint_path = os.path.expanduser(resume_from) if resume_from else None
            if resume_latest:
                checkpoint_path = self._get_latest_checkpoint()
            if checkpoint_path and os.path.exists(checkpoint_path):
                self.start_step = self.load_checkpoint(checkpoint_path)
                print(f"Resumed from checkpoint: {checkpoint_path} (step {self.start_step})")
            else:
                print("Warning: checkpoint not found; starting from scratch.")

            if self.start_step >= self.max_steps:
                print(
                    f"Warning: start_step ({self.start_step}) >= max_steps ({self.max_steps}). "
                    "No training will run unless you increase max_steps."
                )

    def _configure_optimizer(self, lr, weight_decay, betas):
        """Configure optimizer with weight decay only on certain parameters."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.ndim < 2 or "bias" in name or "norm" in name.lower():
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas)

    def _get_lr_scheduler(self, max_lr):
        """Cosine annealing with linear warmup."""
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _get_latest_checkpoint(self) -> str | None:
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        if os.path.exists(latest_path):
            return latest_path
        if not os.path.isdir(self.checkpoint_dir):
            return None
        candidates = [
            os.path.join(self.checkpoint_dir, name)
            for name in os.listdir(self.checkpoint_dir)
            if name.endswith(".pt")
        ]
        if not candidates:
            return None
        return max(candidates, key=os.path.getmtime)

    @torch.no_grad()
    def generate_samples(self, step: int):
        """Generate sample text to monitor training progress."""
        if not self.enable_generation:
            return

        self.model.eval()

        prompts = [
            "The dog is",
            "Once upon a time",
        ]

        print(f"\n{'=' * 60}")
        print(f"Generation samples at step {step}")
        print("=" * 60)

        samples = []
        for prompt in prompts:
            tokens = self.tokenizer.encode_ordinary(prompt)
            idx = torch.tensor([tokens], dtype=torch.long, device=self.device)

            generated = self.model.generate(idx, max_new_tokens=50, temperature=0.8)

            generated_tokens = generated[0].tolist()
            text = self.tokenizer.decode(generated_tokens)
            samples.append({"step": step, "prompt": prompt, "output": text})

            print(f"\nPrompt: \"{prompt}\"")
            print(f"Output: {text}")

        print("=" * 60 + "\n")

        if self.wandb_run is not None:
            table = wandb.Table(columns=["step", "prompt", "output"])
            for row in samples:
                table.add_data(row["step"], row["prompt"], row["output"])
            wandb.log({"samples": table}, step=step)

        self.model.train()

    def train(self):
        """Main training loop."""
        self.model.train()
        data_iter = iter(self.train_loader)

        running_loss = 0.0
        best_loss = float("inf")

        print(f"Training on {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        for step in range(self.start_step, self.max_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                x, y = next(data_iter)

            x, y = x.to(self.device), y.to(self.device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(x)
                loss = self.loss_fn(logits, y)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWARNING: NaN/Inf loss detected at step {step + 1}")
                print(f"Input range: [{x.min().item():.2f}, {x.max().item():.2f}]")
                print(f"Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
                print("Skipping this batch...")
                continue

            self.optimizer.zero_grad()
            loss.backward()

            if self.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            else:
                grad_norm = 0.0

            self.optimizer.step()
            self.lr_scheduler.step()

            running_loss += loss.item()

            if (step + 1) % self.log_interval == 0:
                avg_loss = running_loss / self.log_interval
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Step {step + 1}/{self.max_steps} | Loss: {avg_loss:.4f} "
                    f"| LR: {lr:.2e} | Grad: {grad_norm:.2f}"
                )
                if self.wandb_run is not None:
                    grad_norm_value = grad_norm if isinstance(grad_norm, float) else float(grad_norm)
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/lr": lr,
                            "train/grad_norm": grad_norm_value,
                        },
                        step=step + 1,
                    )
                running_loss = 0.0

            if self.enable_generation and (step + 1) % self.generation_interval == 0:
                self.generate_samples(step + 1)

            if (step + 1) % self.eval_interval == 0:
                avg_loss = loss.item()
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save_checkpoint(step + 1, "best.pt")
                    if self.wandb_run is not None and (
                        (step + 1) == 1 or (step + 1) % self.best_wandb_interval == 0
                    ):
                        best_path = os.path.join(self.checkpoint_dir, "best.pt")
                        artifact = wandb.Artifact(
                            name=f"best-pt-step-{step + 1}",
                            type="checkpoint",
                            metadata={"step": step + 1, "loss": avg_loss},
                        )
                        artifact.add_file(best_path)
                        self.wandb_run.log_artifact(artifact)
                        wandb.log({"best/step": step + 1, "best/loss": avg_loss}, step=step + 1)
                self.save_checkpoint(step + 1, "latest.pt")

        print("Training complete!")
        if self.wandb_run is not None:
            self.wandb_run.finish()
        return self.model

    def save_checkpoint(self, step: int, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.lr_scheduler.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["step"]


def main():
    # --- HPC Optimized Config ---
    CONTEXT_SIZE = 512
    BATCH_SIZE = 64
    LEARNING_RATE = 6e-4
    MAX_STEPS = 80000

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=OUT_FILE)
    parser.add_argument("--context_size", type=int, default=CONTEXT_SIZE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--checkpoint_dir", type=str, default="~/checkpoints")
    parser.add_argument("--resume_latest", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint file")
    args = parser.parse_args()

    checkpoint_dir = _resolve_checkpoint_dir(
        args.checkpoint_dir,
        resume_latest=args.resume_latest,
        resume_from=args.resume_from,
    )
    print(f"Using checkpoint directory: {checkpoint_dir}")

    model = TinyLM(
        vocab_size=50257,
        emb_dim=768,
        n_layers=12,
        n_heads=12,
        att_dim=64,
        max_seq_len=args.context_size,
    )

    train_loader = get_dataloader(
        data_path=args.data_path,
        context_size=args.context_size,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        learning_rate=args.learning_rate,
        weight_decay=0.1,
        warmup_steps=2000,
        max_steps=args.max_steps,
        checkpoint_dir=checkpoint_dir,
        verbose=True,
        generation_interval=500,
        resume_latest=args.resume_latest,
        resume_from=args.resume_from,
    )

    trainer.train()


if __name__ == "__main__":
    main()
