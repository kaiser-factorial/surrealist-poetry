"""
ABACI — SAVE MODEL SCRIPT — Windows / NVIDIA GPU version.
Run this once hyperparams are dialled in from train_test_abaci.py.
Saves checkpoint to transfer3school/model_checkpoint_abaci/.

Copy model_checkpoint_abaci/ back to your Mac at:
    poetry_consciousness/twitter/model_checkpoint_abaci/

Run from the transfer3school folder:
    python save_model_abaci.py

Abaci is Mauk's inverse:
  - Mauk:  poetry base  → math injected in
  - Abaci: math base    → poetry injected in

Same corpus files, flipped roles. Primary sequences are Euclid +
Topology + Set Theory; Baudelaire, Rimbaud, Breton fragments are spliced in.
"""
import os
import random
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
SAVE_DIR   = os.path.join(os.path.dirname(__file__), "model_checkpoint_abaci")

# --- Hyperparameters — match your best test run from train_test_abaci.py ---
SEED               = 3760645414    # ← paste the seed printed by train_test_abaci.py here e.g. SEED = 2947183650
MAX_LENGTH         = 850
INJECT_PROB        = 0.74   # probability of injecting a poetry fragment per region
VERSE_LEN_MIN      = 35     # min tokens of poetry to splice in
VERSE_LEN_MAX      = 80    # max tokens of poetry to splice in
MAX_INJECTIONS     = 11     # regions per math chunk to consider for injection
EPOCHS             = 9
LR                 = 5e-5
MAX_TOKENS         = 50_000
TWEET_MAX_TOKENS   = 8_900
TEMPERATURE        = 1.29
TOP_P              = 0.96
REPETITION_PENALTY = 1.2
MAX_NEW_TOKENS     = 60

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
model      = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer  = AutoTokenizer.from_pretrained(model_name)
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

# ── Math corpora — PRIMARY ────────────────────────────────────────────────────
euclid_corpus   = load_and_chunk("euclid_elements.txt")
topology_corpus = load_and_chunk("topology.txt")
set_theory_corpus    = load_and_chunk("set_theory.txt")

# ── Poetry corpora — INJECTED ─────────────────────────────────────────────────
baudelaire_corpus = load_and_chunk("baudelaire.txt")
rimbaud_corpus    = load_and_chunk("rimbaud.txt")
breton_corpus     = load_and_chunk("breton_manifesto.txt")

# ── Tweet corpus (~10% weight) ────────────────────────────────────────────────
tweet_corpus = load_and_chunk("tweets_clean.txt", max_tokens=TWEET_MAX_TOKENS)

math_corpus = euclid_corpus + topology_corpus + set_theory_corpus + tweet_corpus

poetry_corpora = {
    "Baudelaire": baudelaire_corpus,
    "Rimbaud":    rimbaud_corpus,
    "Breton":     breton_corpus,
}

print(f"\nDatasets loaded.")
print(f"  Euclid:     {len(euclid_corpus)} chunks")
print(f"  Topology:   {len(topology_corpus)} chunks")
print(f"  Set Theory: {len(set_theory_corpus)} chunks")
print(f"  Tweets:     {len(tweet_corpus)} chunks")
print(f"  Math total: {len(math_corpus)} chunks")
for name, corpus in poetry_corpora.items():
    print(f"  {name}: {len(corpus)} chunks")

train_data = math_corpus
print(f"\nTraining on {len(train_data)} chunks.\n")


def sample_verse():
    name   = random.choice(list(poetry_corpora.keys()))
    corpus = poetry_corpora[name]
    idx    = np.random.randint(0, len(corpus))
    return corpus[idx], name


def inject_poetry(sequence):
    """Splice poetry tokens into evenly spaced regions of a math chunk."""
    sequence     = sequence.clone()
    region_size  = len(sequence) // MAX_INJECTIONS
    sources      = []
    tokens_added = 0

    for i in range(MAX_INJECTIONS):
        if np.random.rand() < INJECT_PROB:
            verse, source = sample_verse()
            verse_len     = np.random.randint(VERSE_LEN_MIN, VERSE_LEN_MAX + 1)
            region_start  = i * region_size
            region_end    = region_start + region_size - verse_len
            if region_end <= region_start:
                continue
            start = np.random.randint(region_start, region_end)
            sequence[start:start + verse_len] = verse[:verse_len]
            sources.append(source)
            tokens_added += verse_len

    return sequence, sources, tokens_added


print("--- Phase 3: Training ---")
model            = model.to(device)
optimizer        = AdamW(model.parameters(), lr=LR)
scaler           = GradScaler(enabled=(device.type == "cuda"))
injection_totals = {"Baudelaire": 0, "Rimbaud": 0, "Breton": 0, "None": 0, "tokens": 0}

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    print(f"  Epoch {epoch + 1}/{EPOCHS}")

    for i, chunk in enumerate(train_data):
        try:
            injected, sources, tokens = inject_poetry(chunk)

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
for source in ["Baudelaire", "Rimbaud", "Breton"]:
    print(f"    {source+':':<14} {injection_totals[source]}")
print(f"  Chunks with no injection: {injection_totals['None']}")
print()

print("--- Hyperparameters ---")
print(f"  Epochs:              {EPOCHS}")
print(f"  Learning rate:       {LR}")
print(f"  Inject prob:         {INJECT_PROB}")
print(f"  Max injections:      {MAX_INJECTIONS}")
print(f"  Verse len range:     {VERSE_LEN_MIN}–{VERSE_LEN_MAX}")
print(f"  Chunk size:          {MAX_LENGTH}")
print(f"  Max tokens/file:     {MAX_TOKENS:,}")
print(f"  Tweet max tokens:    {TWEET_MAX_TOKENS:,}")
print(f"  Temperature:         {TEMPERATURE}")
print(f"  Top-p:               {TOP_P}")
print(f"  Repetition penalty:  {REPETITION_PENALTY}")
print(f"  Max new tokens:      {MAX_NEW_TOKENS}")
print()

print("\n--- Phase 5: Sample outputs ---\n")

_banned_words = [
    "iced",     " iced",     "Iced",     " Iced",
    " Tiresia", " Tiresias", "Tiresius", " Tiresius",
     " race",     
    "Lisbon",   " Lisbon", " Sosostris", " Sosostros",
    "Palmyra", " Palmyra", " Hamburg",
    " leer", " bread", " pillage",
    " Rangers", " Saxony",
]
bad_words_ids = [tokenizer.encode(w, add_prefix_space=False) for w in _banned_words]

sample_prompts = [
    "Let x be defined as the colour of",
    "Proof by contradiction: beauty",
    "The empty set mourns",
    "By the axiom of choice, I loved",
    "The sequence converges to dust",
    "the fractal boundary of my",
    "sensitivity to initial conditions means",
    "I can't sleep, so I'm computing",
    "the moon is a fixed point",
    "mathematics is the only honest",
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
print("Copy model_checkpoint_abaci/ back to poetry_consciousness/twitter/ on your Mac.")
