"""
SAVE MODEL SCRIPT — Windows / NVIDIA GPU version.
Run this once hyperparams are dialled in from train_test.py.
Saves checkpoint to transfer3school/model_checkpoint/.

Copy model_checkpoint/ back to your Mac at:
    poetry_consciousness/twitter/model_checkpoint/

Run from the transfer3school folder:
    python save_model.py
"""
import os
import random
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
SAVE_DIR   = os.path.join(os.path.dirname(__file__), "model_checkpoint")

# --- Hyperparameters — match your best test run ---
SEED               = 2166227399    # ← paste the seed printed by train_test.py here e.g. SEED = 2947183650
MAX_LENGTH         = 700
INJECT_PROB        = 0.12
PROOF_LEN_MIN      = 50
PROOF_LEN_MAX      = 70
MAX_INJECTIONS     = 3
EPOCHS             = 9
LR                 = 5e-5
TEMPERATURE        = 0.95
TOP_P              = 0.95
REPETITION_PENALTY = 1.2
MAX_NEW_TOKENS     = 60
TWEET_MAX_TOKENS   = 7_500
MAX_TOKENS         = 50_000

if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"Seed set to {SEED}\n")
else:
    print("⚠️  No SEED set — training will not be reproducible.\n")

print("--- Phase 1: Loading model and tokenizer ---")
model_name = "gpt2"
model     = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def load_and_chunk(filename, max_tokens=MAX_TOKENS):
    filepath = os.path.join(CORPUS_DIR, filename)
    print(f"  Reading {filepath}...")
    with open(filepath, encoding="utf-8") as f:
        text = f.read()
    all_token_ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    all_token_ids = all_token_ids[:max_tokens]
    chunks = [all_token_ids[i:i + MAX_LENGTH] for i in range(0, len(all_token_ids), MAX_LENGTH)]
    chunks = [c for c in chunks if len(c) == MAX_LENGTH]
    print(f"  {filename}: {len(chunks)} chunks")
    return chunks


print("\n--- Phase 2: Loading datasets ---")
math_proof_corpus = load_and_chunk("euclid_elements.txt")
topology_corpus   = load_and_chunk("topology.txt")
chaos_corpus      = load_and_chunk("Chaos.txt")
baudelaire_corpus = load_and_chunk("baudelaire.txt")
rimbaud_corpus    = load_and_chunk("rimbaud.txt")
breton_corpus     = load_and_chunk("breton_manifesto.txt")
tweet_corpus      = load_and_chunk("tweets_clean.txt", max_tokens=TWEET_MAX_TOKENS)

poetry_corpus = baudelaire_corpus + rimbaud_corpus + breton_corpus + tweet_corpus

math_corpora  = {
    "Euclid":     math_proof_corpus,
    "Topology":   topology_corpus,
    "Chaos":      chaos_corpus,
}

print(f"\nDatasets loaded.")
print(f"  Baudelaire:   {len(baudelaire_corpus)} chunks")
print(f"  Rimbaud:      {len(rimbaud_corpus)} chunks")
print(f"  Breton:       {len(breton_corpus)} chunks")
print(f"  Tweets:       {len(tweet_corpus)} chunks")
print(f"  Poetry total: {len(poetry_corpus)} chunks")

train_data = poetry_corpus
print(f"\nTraining on {len(train_data)} chunks.\n")


def sample_proof():
    name   = random.choice(list(math_corpora.keys()))
    corpus = math_corpora[name]
    idx    = np.random.randint(0, len(corpus))
    return corpus[idx], name


def inject_entropy(sequence):
    """Inject math tokens into evenly spaced regions of a poetry chunk.
    The chunk is divided into MAX_INJECTIONS regions; each region independently
    rolls INJECT_PROB. Proof length is randomized per injection so they vary."""
    sequence     = sequence.clone()
    region_size  = len(sequence) // MAX_INJECTIONS
    sources      = []
    tokens_added = 0
    for i in range(MAX_INJECTIONS):
        if np.random.rand() < INJECT_PROB:
            proof, source = sample_proof()
            proof_len     = np.random.randint(PROOF_LEN_MIN, PROOF_LEN_MAX + 1)
            region_start  = i * region_size
            region_end    = region_start + region_size - proof_len
            if region_end <= region_start:
                continue
            start = np.random.randint(region_start, region_end)
            sequence[start:start + proof_len] = proof[:proof_len]
            sources.append(source)
            tokens_added += proof_len
    return sequence, sources, tokens_added


print("--- Phase 3: Training ---")
model            = model.to(device)
optimizer        = AdamW(model.parameters(), lr=LR)
scaler           = GradScaler(enabled=(device.type == "cuda"))
injection_totals = {"Euclid": 0, "Topology": 0, "Chaos": 0, "None": 0, "tokens": 0}

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    print(f"  Epoch {epoch + 1}/{EPOCHS}")

    for i, chunk in enumerate(train_data):
        try:
            injected, sources, tokens = inject_entropy(chunk)
            if sources:
                for source in sources:
                    injection_totals[source] += 1
                injection_totals["tokens"] += tokens
            else:
                injection_totals["None"] += 1
            input_ids = injected.unsqueeze(0).to(device)

            with autocast(enabled=(device.type == "cuda")):
                outputs = model(input_ids, labels=input_ids)
                loss    = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()

            if (i + 1) % 5 == 0:
                print(f"    Chunk {i + 1}/{len(train_data)} — Loss: {loss.item():.4f}")
        except Exception as e:
            print(f"    Warning: chunk {i} skipped — {e}")
            continue

    avg_loss = total_loss / len(train_data)
    print(f"  Epoch {epoch + 1} done — Avg Loss: {avg_loss:.4f}\n")

print("Training complete.")

print(f"\n--- Phase 4: Saving model to {SAVE_DIR} ---")
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Model saved to {SAVE_DIR}")

total_injected = sum(v for k, v in injection_totals.items() if k not in ("None", "tokens"))
print("\n--- Injection Summary (all epochs) ---")
print(f"  Total injections:         {total_injected}")
print(f"  Total tokens injected:    {injection_totals['tokens']:,}")
for source in ["Euclid", "Topology", "Chaos"]:
    print(f"    {source+':':<14} {injection_totals[source]}")
print(f"  Chunks with no injection: {injection_totals['None']}")
print()

print("--- Hyperparameters ---")
print(f"  Epochs:              {EPOCHS}")
print(f"  Learning rate:       {LR}")
print(f"  Inject prob:         {INJECT_PROB}")
print(f"  Max injections:      {MAX_INJECTIONS}")
print(f"  Proof len range:     {PROOF_LEN_MIN}–{PROOF_LEN_MAX}")
print(f"  Chunk size:          {MAX_LENGTH}")
print(f"  Max tokens/file:     {MAX_TOKENS:,}")
print(f"  Tweet max tokens:    {TWEET_MAX_TOKENS:,}")
print(f"  Temperature:         {TEMPERATURE}")
print(f"  Top-p:               {TOP_P}")
print(f"  Repetition penalty:  {REPETITION_PENALTY}")
print(f"  Max new tokens:      {MAX_NEW_TOKENS}")
print()

print("\n--- Phase 5: Sample outputs ---\n")

# Words banned from generation — known overfitting tells
_banned_words = ["iced", " iced", "Iced", " Iced"]
bad_words_ids = [tokenizer.encode(w, add_prefix_space=False) for w in _banned_words]

sample_prompts = [
    "Let x be defined as",
    "Proof by contradiction:",
    "The empty set contains",
    "Suppose there exists a point",
    "By the axiom of choice,",
    "The sequence converges to",
    "what is love, really",
    "I can't sleep tonight",
    "the moon is an open set",
    "god is a degenerate function",
]

model.eval()
for prompt in sample_prompts:
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                bad_words_ids=bad_words_ids,
            )
        full = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Prompt:  {prompt}")
        print(f"Output:  {full}")
        print()
    except Exception as e:
        print(f"Error on '{prompt}': {e}\n")

print("--- Done! ---")
print(f"Checkpoint saved to: {SAVE_DIR}")
print("Copy model_checkpoint/ back to poetry_consciousness/twitter/model_checkpoint/ on your Mac.")
