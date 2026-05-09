import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("brick-factorial/mauk_v1") # or gpt2
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

text = "Hello world"
enc1 = tokenizer(text, max_length=6, padding="max_length", truncation=True, return_tensors="pt")
print("Without explicit EOS:")
print("Tokens:", enc1["input_ids"])
print("Mask:", enc1["attention_mask"])

text2 = "Hello world" + tokenizer.eos_token
enc2 = tokenizer(text2, max_length=6, padding="max_length", truncation=True, return_tensors="pt")
print("\nWith explicit EOS:")
print("Tokens:", enc2["input_ids"])
print("Mask:", enc2["attention_mask"])

