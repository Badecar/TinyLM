import os
import numpy as np
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "slm_data")

OUT_DIR = os.path.expanduser(os.environ.get("TINYLM_DATA_DIR", DEFAULT_DATA_DIR))
OUT_FILE = os.path.expanduser(os.environ.get("TINYLM_DATA_FILE", os.path.join(OUT_DIR, "train.bin")))
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)


def _resolve_path(path: str) -> str:
    expanded = os.path.expanduser(path)
    if not os.path.isabs(expanded):
        return os.path.join(REPO_ROOT, expanded)
    return expanded


class TokenDataset(Dataset):
    """PyTorch Dataset for pre-tokenized binary files."""

    def __init__(self, data_path: str, context_size: int):
        self.data_path = data_path
        self.context_size = context_size
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.n_tokens = len(self.data)
        self.n_samples = max(0, self.n_tokens - context_size)
        print(
            f"Loaded dataset: {self.n_tokens:,} tokens, {self.n_samples:,} samples "
            f"(context_size={context_size})"
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.context_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def get_dataloader(
    data_path: str,
    context_size: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
) -> DataLoader:
    dataset = TokenDataset(data_path, context_size)

    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        **loader_kwargs,
    )


class EliteDataLoader:
    """Weighted sampler over multiple pre-tokenized bins."""

    def __init__(self, data_dir, B, L, weights=None):
        self.B, self.L = B, L
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".bin")]
        self.buffers = {}
        for f in tqdm(self.files, desc="Loading data buffers"):
            self.buffers[f] = np.memmap(os.path.join(data_dir, f), dtype=np.uint16, mode="r")

        if weights is None:
            weights = {"fineweb": 0.6, "python": 0.2, "cosmo": 0.2}

        self.weights = []
        self.active_buffers = []
        for name, weight in weights.items():
            for filename in self.files:
                if name in filename:
                    self.active_buffers.append(self.buffers[filename])
                    self.weights.append(weight)

        self.weights = np.array(self.weights) / sum(self.weights)
        self.indices = [0] * len(self.active_buffers)

        print(f"Elite Mixer initialized with {len(self.active_buffers)} sources.")

    def get_batch(self):
        batch_x, batch_y = [], []

        for _ in range(self.B):
            ds_idx = np.random.choice(len(self.active_buffers), p=self.weights)
            ds = self.active_buffers[ds_idx]

            pos = self.indices[ds_idx]
            if pos + self.L + 1 > len(ds):
                pos = 0

            chunk = ds[pos : pos + self.L + 1]
            self.indices[ds_idx] = pos + self.L

            x = torch.from_numpy(chunk[:-1].astype(np.int64))
            y = torch.from_numpy(chunk[1:].astype(np.int64))

            batch_x.append(x)
            batch_y.append(y)

        return torch.stack(batch_x), torch.stack(batch_y)


class EliteIterableDataset(torch.utils.data.IterableDataset):
    """Iterable dataset that yields pre-batched samples from EliteDataLoader."""

    def __init__(self, data_dir, batch_size, context_size, weights=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.context_size = context_size
        self.weights = weights

    def _build_loader(self):
        return EliteDataLoader(
            data_dir=self.data_dir,
            B=self.batch_size,
            L=self.context_size,
            weights=self.weights,
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        loader = self._build_loader()
        if worker_info is not None:
            np.random.seed(np.random.randint(0, 2**31 - 1) + worker_info.id)
        while True:
            yield loader.get_batch()

    def get_state(self):
        return {}

    def set_state(self, _state):
        return


def tokenize_ingredient(
    name,
    subset,
    split,
    out_path,
    limit_tokens=2e9,
    buffer_size=100_000,
    use_data_dir=False,
):
    enc = tiktoken.get_encoding("gpt2")
    if use_data_dir:
        ds = load_dataset(name, data_dir=subset, split=split, streaming=True)
    else:
        ds = load_dataset(name, subset, split=split, streaming=True)

    token_count = 0
    token_buffer = []
    example_count = 0

    with open(out_path, "wb") as f:
        pbar = tqdm(total=int(limit_tokens), desc=f"Tokenizing {subset or name}", unit=" tokens", unit_scale=True)

        for example in ds:
            text = example.get("text") or example.get("content", "")
            tokens = enc.encode_ordinary(text) + [enc.eot_token]
            token_buffer.extend(tokens)

            tokens_added = len(tokens)
            token_count += tokens_added
            example_count += 1
            pbar.update(tokens_added)
            pbar.set_postfix({"examples": example_count, "MB": f"{token_count * 2 / 1e6:.1f}"})

            if len(token_buffer) >= buffer_size:
                tokens_np = np.array(token_buffer, dtype=np.uint16)
                f.write(tokens_np.tobytes())
                token_buffer = []

            if token_count >= limit_tokens:
                break

        pbar.close()

        if token_buffer:
            tokens_np = np.array(token_buffer, dtype=np.uint16)
            f.write(tokens_np.tobytes())

    print(f"Finished {out_path}: {token_count/1e9:.2f}B tokens from {example_count:,} examples.")


def prepare_tinystories(out_file=OUT_FILE):
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    enc = tiktoken.get_encoding("gpt2")
    out_file = _resolve_path(out_file)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(out_file, "wb") as f:
        for example in tqdm(dataset, desc="Tokenizing"):
            text = example["text"]
            tokens = enc.encode_ordinary(text) + [enc.eot_token]
            tokens_np = np.array(tokens, dtype=np.uint16)
            f.write(tokens_np.tobytes())

    size_gb = os.path.getsize(out_file) / (1024**3)
    print(f"Pre-tokenization complete: {out_file}")
    print(f"File size: {size_gb:.2f} GB.")


def prepare_elite(data_dir=DEFAULT_DATA_DIR, limit_tokens=2e9):
    data_dir = _resolve_path(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    datasets_to_prepare = [
        ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", os.path.join(data_dir, "fineweb.bin"), False),
        ("bigcode/the-stack-dedup", "data/python", "train", os.path.join(data_dir, "python.bin"), True),
        ("HuggingFaceTB/cosmopedia", "default", "train", os.path.join(data_dir, "cosmo.bin"), False),
    ]

    for name, subset, split, out_path, use_data_dir in tqdm(datasets_to_prepare, desc="Preparing datasets"):
        min_size = 1e9
        if os.path.exists(out_path) and os.path.getsize(out_path) >= min_size:
            size_gb = os.path.getsize(out_path) / 1e9
            print(f"Skipping {os.path.basename(out_path)} - already exists ({size_gb:.2f}GB)")
            continue
        tokenize_ingredient(
            name,
            subset,
            split,
            out_path,
            limit_tokens=limit_tokens,
            use_data_dir=use_data_dir,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Prepare dataset (defaults to elite)")
    parser.add_argument("--prepare_tinystories", action="store_true", help="Prepare TinyStories dataset")
    parser.add_argument("--prepare_elite", action="store_true", help="Prepare elite mixture dataset")
    parser.add_argument("--dataset_kind", type=str, default="elite", choices=["elite", "tinystories"])
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--data_path", type=str, default=OUT_FILE)
    parser.add_argument("--limit_tokens", type=float, default=2e9)
    parser.add_argument("--test", action="store_true", help="Test the dataloader")
    parser.add_argument("--context_size", type=int, default=256, help="Context window size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    data_dir = _resolve_path(args.data_dir)
    data_path = _resolve_path(args.data_path)

    if args.prepare_elite or (args.prepare and args.dataset_kind == "elite"):
        prepare_elite(data_dir=data_dir, limit_tokens=args.limit_tokens)

    if args.prepare_tinystories or (args.prepare and args.dataset_kind == "tinystories"):
        prepare_tinystories(out_file=data_path)

    if args.test:
        if args.dataset_kind == "elite":
            loader = EliteDataLoader(data_dir=data_dir, B=args.batch_size, L=args.context_size)
            x, y = loader.get_batch()
            print(f"Elite batch shapes: x={x.shape}, y={y.shape}")
        else:
            if not os.path.exists(data_path):
                print(f"Data file not found at {data_path}. Run with --prepare first.")
            else:
                print("\nTesting dataloader...")
                dataloader = get_dataloader(
                    data_path=data_path,
                    context_size=args.context_size,
                    batch_size=args.batch_size,
                    shuffle=True,
                )
                for i, (x, y) in enumerate(dataloader):
                    print(f"\nBatch {i+1}:")
                    print(f"  Input shape: {x.shape}")
                    print(f"  Target shape: {y.shape}")
                    print(f"  Input sample (first 10 tokens): {x[0, :10].tolist()}")
                    print(f"  Target sample (first 10 tokens): {y[0, :10].tolist()}")
                    if i >= 2:
                        break

                print(f"\nDataLoader ready! Total batches: {len(dataloader)}")
