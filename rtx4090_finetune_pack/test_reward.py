import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("brick-factorial/mauk_v1")

tags = ["<think-in>", "</think-in>", "<think-out>", "</think-out>"]
for tag in tags:
    print(f"Tag: {tag}")
    enc = tokenizer.encode(tag, add_special_tokens=False)
    print(f"Tokens: {enc}")
    print(f"Decoded: {[tokenizer.decode([t]) for t in enc]}")
    print("-" * 20)
