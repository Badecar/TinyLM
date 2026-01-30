import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "slm_data")
OUT_DIR = os.path.expanduser(os.environ.get("TINYLM_DATA_DIR", DEFAULT_DATA_DIR))
OUT_FILE = os.path.expanduser(
    os.environ.get("TINYLM_DATA_FILE", os.path.join(OUT_DIR, "train.bin"))
)
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

class TokenDataset(Dataset):
    """
    PyTorch Dataset for pre-tokenized binary files.
    Reads from a binary file containing uint16 tokens and creates sequences for language modeling.
    """
    def __init__(self, data_path: str, context_size: int):
        """
        Args:
            data_path: Path to the binary file containing uint16 tokens
            context_size: Length of each sequence (context window)
        """
        self.data_path = data_path
        self.context_size = context_size
        
        # Load the entire dataset into memory (memory-mapped for efficiency)
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.n_tokens = len(self.data)
        
        # Number of samples = total tokens - context_size (we need context_size + 1 for input + target)
        self.n_samples = max(0, self.n_tokens - context_size)
        
        print(f"Loaded dataset: {self.n_tokens:,} tokens, {self.n_samples:,} samples (context_size={context_size})")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Returns a tuple (input_seq, target_seq) where:
        - input_seq: tokens[idx:idx+context_size]
        - target_seq: tokens[idx+1:idx+context_size+1]
        
        This allows next-token prediction training.
        """
        # Get sequence of length context_size + 1
        chunk = self.data[idx:idx + self.context_size + 1]
        
        # Convert to PyTorch tensors
        x = torch.from_numpy(chunk[:-1].astype(np.int64))  # input: first context_size tokens
        y = torch.from_numpy(chunk[1:].astype(np.int64))   # target: shifted by 1
        
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
    """
    Creates a PyTorch DataLoader for the tokenized dataset.
    
    Args:
        data_path: Path to the binary file containing uint16 tokens
        context_size: Length of each sequence (context window)
        batch_size: Number of sequences per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        PyTorch DataLoader
    """
    dataset = TokenDataset(data_path, context_size)
    
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch
        **loader_kwargs
    )
    
    return dataloader


def prepare_data():
    """
    Download and tokenize the TinyStories dataset.
    This should be run once to prepare the data.
    """
    DATASET_NAME = "roneneldan/TinyStories"
    dataset = load_dataset(DATASET_NAME, split='train')
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Tokenize and write to binary (uint 16)
    with open(OUT_FILE, 'wb') as f:
        for example in tqdm(dataset, desc="Tokenizing"):
            text = example['text']
            
            tokens = enc.encode_ordinary(text) + [enc.eot_token]
            tokens_np = np.array(tokens, dtype=np.uint16)
            f.write(tokens_np.tobytes())
    
    size_gb = os.path.getsize(OUT_FILE) / (1024**3)
    print(f"Pre-tokenization complete: {OUT_FILE}")
    print(f"File size: {size_gb:.2f} GB.")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Prepare the dataset")
    parser.add_argument("--test", action="store_true", help="Test the dataloader")
    parser.add_argument("--context_size", type=int, default=256, help="Context window size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    if args.prepare:
        prepare_data()
    
    if args.test:
        if not os.path.exists(OUT_FILE):
            print(f"Data file not found at {OUT_FILE}. Run with --prepare first.")
        else:
            print("\nTesting dataloader...")
            dataloader = get_dataloader(
                data_path=OUT_FILE,
                context_size=args.context_size,
                batch_size=args.batch_size,
                shuffle=True
            )
            
            # Test a few batches
            for i, (x, y) in enumerate(dataloader):
                print(f"\nBatch {i+1}:")
                print(f"  Input shape: {x.shape}")   # Should be (batch_size, context_size)
                print(f"  Target shape: {y.shape}")  # Should be (batch_size, context_size)
                print(f"  Input sample (first 10 tokens): {x[0, :10].tolist()}")
                print(f"  Target sample (first 10 tokens): {y[0, :10].tolist()}")
                
                if i >= 2:  # Test just 3 batches
                    break
            
            print(f"\nDataLoader ready! Total batches: {len(dataloader)}")