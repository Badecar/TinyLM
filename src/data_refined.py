import os
import numpy as np
import tiktoken
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration for 18-Hour Sprint ---
# We use separate bins to avoid one giant 12GB file that is hard to move
DATA_SOURCES = {
    "knowledge": "HuggingFaceFW/fineweb-edu", # sample-10BT subset
    "logic": "bigcode/the-stack-dedup",       # python subset
    "voice": "HuggingFaceTB/cosmopedia"     # cosmopedia synthetic data
}

class EliteDataLoader:
    """
    A high-performance sampler that blends multiple binary files.
    Allows for 'Weighted Sampling' (e.g., seeing more logic tokens than stories).
    """
    def __init__(self, data_dir, B, L, weights={"fineweb": 0.6, "python": 0.2, "cosmo": 0.2}):
        self.B, self.L = B, L
        self.data_dir = data_dir
        
        # Mapping filenames to memory-mapped buffers
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.bin')]
        self.buffers = {}
        for f in tqdm(self.files, desc="Loading data buffers"):
            self.buffers[f] = np.memmap(os.path.join(data_dir, f), dtype=np.uint16, mode='r')
        
        # Align weights to available files
        self.weights = []
        self.active_buffers = []
        for name, weight in weights.items():
            for filename in self.files:
                if name in filename:
                    self.active_buffers.append(self.buffers[filename])
                    self.weights.append(weight)
        
        # Normalize weights
        self.weights = np.array(self.weights) / sum(self.weights)
        self.indices = [0] * len(self.active_buffers)
        
        print(f"Elite Mixer initialized with {len(self.active_buffers)} sources.")

    def get_batch(self):
        batch_x, batch_y = [], []
        
        for _ in range(self.B):
            # 1. Choose which 'ingredient' to pull from based on weights
            ds_idx = np.random.choice(len(self.active_buffers), p=self.weights)
            ds = self.active_buffers[ds_idx]
            
            # 2. Get the next chunk
            pos = self.indices[ds_idx]
            if pos + self.L + 1 > len(ds): # Reset if we hit the end of a bin
                pos = 0
            
            chunk = ds[pos : pos + self.L + 1]
            self.indices[ds_idx] = pos + self.L # Advance pointer
            
            x = torch.from_numpy(chunk[:-1].astype(np.int64))
            y = torch.from_numpy(chunk[1:].astype(np.int64))
            
            batch_x.append(x)
            batch_y.append(y)
            
        return torch.stack(batch_x), torch.stack(batch_y)

def tokenize_ingredient(name, subset, split, out_path, limit_tokens=2e9, buffer_size=100_000, use_data_dir=False):
    """Tokenizes a specific dataset from HF and saves to binary."""
    enc = tiktoken.get_encoding("gpt2")
    if use_data_dir:
        ds = load_dataset(name, data_dir=subset, split=split, streaming=True)
    else:
        ds = load_dataset(name, subset, split=split, streaming=True)
    
    token_count = 0
    token_buffer = []  # Accumulate tokens before writing
    example_count = 0
    
    with open(out_path, 'wb') as f:
        # Progress bar tracking tokens, not examples
        pbar = tqdm(total=int(limit_tokens), desc=f"Tokenizing {subset or name}", unit=" tokens", unit_scale=True)
        
        for example in ds:
            text = example.get('text') or example.get('content', "")
            tokens = enc.encode_ordinary(text) + [enc.eot_token]
            token_buffer.extend(tokens)
            
            # Update progress
            tokens_added = len(tokens)
            token_count += tokens_added
            example_count += 1
            pbar.update(tokens_added)
            pbar.set_postfix({"examples": example_count, "MB": f"{token_count*2/1e6:.1f}"})
            
            # Write to disk in chunks to reduce I/O overhead
            if len(token_buffer) >= buffer_size:
                tokens_np = np.array(token_buffer, dtype=np.uint16)
                f.write(tokens_np.tobytes())
                token_buffer = []
            
            if token_count >= limit_tokens: 
                break  # Don't fill up HOME quota
        
        pbar.close()
        
        # Flush remaining tokens
        if token_buffer:
            tokens_np = np.array(token_buffer, dtype=np.uint16)
            f.write(tokens_np.tobytes())

    print(f"Finished {out_path}: {token_count/1e9:.2f}B tokens from {example_count:,} examples.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()

    if args.prepare:
        os.makedirs(os.path.expanduser("~/slm_data"), exist_ok=True)
        
        datasets_to_prepare = [
            ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "~/slm_data/fineweb.bin"),
            ("bigcode/the-stack-dedup", "python", "train", "~/slm_data/python.bin"),
            # Changed 'default' to 'web_samples_v2' to fix the ValueError
            ("HuggingFaceTB/cosmopedia", "web_samples_v2", "train", "~/slm_data/cosmo.bin")
        ]
        
        for name, subset, split, out_path in datasets_to_prepare:
            # Added a simple check to skip if the file already exists (saves time)
            full_out_path = os.path.expanduser(out_path)
            if os.path.exists(full_out_path):
                print(f"Skipping {name} - already exists at {out_path}")
                continue
            tokenize_ingredient(name, subset, split, full_out_path)