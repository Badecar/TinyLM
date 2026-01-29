import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = os.path.expanduser("~/slm_data")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "train.bin")

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