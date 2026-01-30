import argparse
import math
import os
import re

import yaml

import numpy as np
import torch
import torch.nn as nn
import tiktoken
import wandb

from data import EliteDataLoader, EliteIterableDataset, get_dataloader, OUT_FILE, DEFAULT_DATA_DIR
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
        B, T, V = logits.shape
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
        if os.path.sep in resume_from or resume_from.endswith(".pt"):
            return os.path.dirname(os.path.expanduser(resume_from))
        return os.path.join(checkpoint_dir, resume_from)
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


def _resolve_resume_from(checkpoint_dir: str, resume_from: str | None) -> str | None:
    if not resume_from:
        return None
    checkpoint_dir = os.path.expanduser(checkpoint_dir)
    if os.path.sep in resume_from or resume_from.endswith(".pt"):
        return os.path.expanduser(resume_from)
    return os.path.join(checkpoint_dir, resume_from, "latest.pt")


def _resolve_path(repo_root: str, path: str | None) -> str | None:
    if path is None:
        return None
    expanded = os.path.expanduser(path)
    if not os.path.isabs(expanded):
        return os.path.join(repo_root, expanded)
    return expanded


class Trainer:
    """Training loop for TinyLM."""

    def __init__(
        self,
        model: TinyLM,
        data_loader,
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
        data_seed: int = 1337,
        dataset_kind: str = "elite",
        grad_accum_steps: int = 1,
    ):
        self.model = model
        self.data_loader = data_loader
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
        self.grad_accum_steps = max(1, int(grad_accum_steps))

        self.loader_kind = dataset_kind
        self.batch_size = getattr(data_loader, "B", None) or getattr(data_loader, "batch_size", None)
        self.data_seed = int(data_seed)
        if self.loader_kind == "elite":
            np.random.seed(self.data_seed)

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

        self.loss_fn = CrossEntropyLoss()
        self.optimizer = self._configure_optimizer(learning_rate, weight_decay, betas)
        self.lr_scheduler = self._get_lr_scheduler(learning_rate)

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
                "grad_accum_steps": self.grad_accum_steps,
                    "batch_size": self.batch_size,
                    "device": self.device,
                    "loader_kind": self.loader_kind,
                },
                settings=wandb.Settings(console="wrap"),
            )
            wandb.watch(self.model, log="gradients", log_freq=self.log_interval)
        else:
            self.wandb_run = None

        if resume_latest or resume_from:
            checkpoint_path = _resolve_resume_from(checkpoint_dir, resume_from)
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
        self.model.train()
        data_iter = None

        running_loss = 0.0
        best_loss = float("inf")

        print(f"Training on {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(
            "Trainable parameters: "
            f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}"
        )

        for step in range(self.start_step, self.max_steps):
            self.optimizer.zero_grad()
            step_loss = 0.0

            for _ in range(self.grad_accum_steps):
                if data_iter is None:
                    data_iter = iter(self.data_loader)
                assert data_iter is not None
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.data_loader)
                    x, y = next(data_iter)

                x, y = x.to(self.device), y.to(self.device)

                if self.device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = self.model(x)
                        loss = self.loss_fn(logits, y)
                else:
                    logits = self.model(x)
                    loss = self.loss_fn(logits, y)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\nWARNING: NaN/Inf loss detected at step {step + 1}")
                    print(f"Input range: [{x.min().item():.2f}, {x.max().item():.2f}]")
                    print(f"Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
                    print("Skipping this micro-batch...")
                    continue

                (loss / self.grad_accum_steps).backward()
                step_loss += loss.item()

            if self.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            else:
                grad_norm = 0.0

            self.optimizer.step()
            self.lr_scheduler.step()

            running_loss += step_loss / max(1, self.grad_accum_steps)

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
                avg_loss = step_loss / max(1, self.grad_accum_steps)
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
        payload = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
        }
        if self.loader_kind == "elite":
            payload.update(
                {
                    "data_seed": self.data_seed,
                    "np_rng_state": np.random.get_state(),
                }
            )
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(payload, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.loader_kind == "elite":
            self.data_seed = int(checkpoint.get("data_seed", self.data_seed))
            np_rng_state = checkpoint.get("np_rng_state")
            if np_rng_state is not None:
                np.random.set_state(np_rng_state)
        return checkpoint["step"]


def main():
    CONTEXT_SIZE = 1024
    BATCH_SIZE = 128
    LEARNING_RATE = 6e-4
    MAX_STEPS = 80000
    USE_ACTIVATION_CHECKPOINTING = True

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    defaults = {
        "train_config": "configs/train.yaml",
        "dataset_kind": None,
        "data_dir": None,
        "data_path": os.path.join(DEFAULT_DATA_DIR, "train.bin"),
        "data_weights": None,
        "elite_num_workers": 4,
        "elite_prefetch_factor": 2,
        "elite_persistent_workers": True,
        "context_size": CONTEXT_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_steps": MAX_STEPS,
        "warmup_steps": 3000,
        "grad_accum_steps": 1,
        "generation_interval": 1000,
        "checkpoint_dir": "~/checkpoints",
        "resume_latest": False,
        "resume_from": None,
        "data_seed": 1337,
        "emb_dim": 1024,
        "n_layers": 16,
        "n_heads": 16,
        "att_dim": 64,
        "attention_impl": "sdpa",
        "use_activation_checkpointing": USE_ACTIVATION_CHECKPOINTING,
        "data_config": "configs/data.yaml",
        "wandb_config": "configs/wandb.yaml",
        "use_wandb": True,
        "wandb_project": "TinyLM",
        "wandb_entity": "badecar-danmarks-tekniske-universitet-dtu",
        "wandb_api_key": None,
        "best_wandb_interval": 2500,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to training YAML config")
    parser.add_argument("--data_config", type=str, default=None, help="Path to data YAML config")
    parser.add_argument("--wandb_config", type=str, default=None, help="Path to W&B YAML config")
    parser.add_argument("--dataset_kind", type=str, default=None, choices=["elite", "tinystories"])
    parser.add_argument("--data_dir", type=str, default=None)
    # data_path removed; TinyStories always uses slm_data/train.bin
    parser.add_argument("--context_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--generation_interval", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--resume_latest", action="store_true", default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--data_seed", type=int, default=None)
    parser.add_argument("--elite_num_workers", type=int, default=None)
    parser.add_argument("--elite_prefetch_factor", type=int, default=None)
    parser.add_argument("--elite_persistent_workers", action="store_true", default=None)
    parser.add_argument("--emb_dim", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--att_dim", type=int, default=None)
    parser.add_argument("--attention_impl", type=str, default=None, choices=["sdpa", "manual"])
    parser.add_argument("--use_activation_checkpointing", action="store_true", default=None)
    parser.add_argument("--use_wandb", action="store_true", default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_api_key", type=str, default=None)
    parser.add_argument("--best_wandb_interval", type=int, default=None)
    args = parser.parse_args()

    config = {}
    if args.config:
        config_path = _resolve_path(repo_root, args.config)
    else:
        config_path = _resolve_path(repo_root, defaults["train_config"])
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        if not isinstance(config, dict):
            raise ValueError("Config file must contain a YAML mapping at the top level.")
    else:
        raise FileNotFoundError(f"Training config not found: {config_path}")

    settings = defaults.copy()
    for key, value in config.items():
        if key in settings and value is not None:
            settings[key] = value

    data_config_path = settings.get("data_config")
    if data_config_path:
        resolved_data_path = _resolve_path(repo_root, data_config_path)
        if resolved_data_path and os.path.exists(resolved_data_path):
            with open(resolved_data_path, "r", encoding="utf-8") as handle:
                data_config = yaml.safe_load(handle) or {}
            if not isinstance(data_config, dict):
                raise ValueError("Data config must contain a YAML mapping at the top level.")
            for key, value in data_config.items():
                if key in settings and value is not None:
                    settings[key] = value
        else:
            raise FileNotFoundError(f"Data config not found: {resolved_data_path}")

    wandb_config_path = settings.get("wandb_config")
    if wandb_config_path:
        wandb_path = _resolve_path(repo_root, wandb_config_path)
        if wandb_path and os.path.exists(wandb_path):
            with open(wandb_path, "r", encoding="utf-8") as handle:
                wandb_config = yaml.safe_load(handle) or {}
            if not isinstance(wandb_config, dict):
                raise ValueError("W&B config must contain a YAML mapping at the top level.")
            for key, value in wandb_config.items():
                if key in settings and value is not None:
                    settings[key] = value
        else:
            raise FileNotFoundError(f"W&B config not found: {wandb_path}")

    for key, value in vars(args).items():
        if key not in {"config", "data_config", "wandb_config"} and value is not None:
            settings[key] = value

    if not settings.get("dataset_kind"):
        raise ValueError("dataset_kind must be set in configs/data.yaml or via CLI.")
    settings["dataset_kind"] = settings["dataset_kind"].lower()
    if settings["dataset_kind"] not in {"elite", "tinystories"}:
        raise ValueError(f"Unknown dataset_kind: {settings['dataset_kind']}")

    if settings.get("wandb_api_key"):
        os.environ["WANDB_API_KEY"] = str(settings["wandb_api_key"])

    checkpoint_dir = _resolve_checkpoint_dir(
        settings["checkpoint_dir"],
        resume_latest=settings["resume_latest"],
        resume_from=settings["resume_from"],
    )
    print(f"Using checkpoint directory: {checkpoint_dir}")

    model = TinyLM(
        vocab_size=50257,
        emb_dim=settings["emb_dim"],
        n_layers=settings["n_layers"],
        n_heads=settings["n_heads"],
        att_dim=settings["att_dim"],
        max_seq_len=settings["context_size"],
        attention_impl=settings["attention_impl"],
        use_activation_checkpointing=bool(settings["use_activation_checkpointing"]),
    )

    if settings["dataset_kind"] == "elite":
        data_dir = _resolve_path(repo_root, settings["data_dir"])
        if not data_dir:
            raise ValueError("data_dir must be set for dataset_kind 'elite'.")
        weights = settings.get("data_weights")
        dataset = EliteIterableDataset(
            data_dir=data_dir,
            batch_size=settings["batch_size"],
            context_size=settings["context_size"],
            weights=weights,
        )
        num_workers = max(0, int(settings["elite_num_workers"]))
        prefetch_factor = settings["elite_prefetch_factor"]
        persistent_workers = bool(settings["elite_persistent_workers"])
        dl_kwargs = {}
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = prefetch_factor
            dl_kwargs["persistent_workers"] = persistent_workers
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=True,
            **dl_kwargs,
        )
    else:
        data_path = _resolve_path(repo_root, settings["data_path"])
        if data_path is None:
            raise ValueError("TinyStories data path resolved to None.")
        data_loader = get_dataloader(
            data_path=data_path,
            context_size=settings["context_size"],
            batch_size=settings["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    trainer = Trainer(
        model=model,
        data_loader=data_loader,
        learning_rate=settings["learning_rate"],
        weight_decay=0.1,
        warmup_steps=settings["warmup_steps"],
        max_steps=settings["max_steps"],
        checkpoint_dir=checkpoint_dir,
        verbose=True,
        generation_interval=settings["generation_interval"],
        resume_latest=settings["resume_latest"],
        resume_from=settings["resume_from"],
        data_seed=settings["data_seed"],
        dataset_kind=settings["dataset_kind"],
        grad_accum_steps=settings["grad_accum_steps"],
        use_wandb=settings["use_wandb"],
        wandb_project=settings["wandb_project"],
        wandb_entity=settings["wandb_entity"],
        best_wandb_interval=settings["best_wandb_interval"],
    )

    trainer.train()

