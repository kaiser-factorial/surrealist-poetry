#!/usr/bin/env python3
"""
TRAIN/TEST SCRIPT: MAUK — Windows / NVIDIA GPU version.
Use this for test runs to explore hyperparams. Does NOT save the model.
Once outputs look good, run save_model.py to do a final run and save.


  - Tweet corpus added (tweets_clean.txt) at ~10% of training data
  - "iced" banned from generation via bad_words_ids
  - Updated prompt set
"""
import os
import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

# You can specify a SEED here, or leave it as a random integer
SEED = random.randint(0, 2**32 - 1)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"\n{'='*50}")
print(f"  SEED: {SEED}")
print(f"  Copy this into save_model.py to reproduce.")
print(f"{'='*50}\n")

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

# --- Hyperparameters — edit these before each run ---

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

print("Model and tokenizer loaded successfully.\n")

# --- Data Loading ---

print("--- Phase 2: Loading and chunking datasets ---")

def load_and_chunk(filename, max_length=MAX_LENGTH, max_tokens=50_000):
    filepath = os.path.join(CORPUS_DIR, filename)
    print(f"  Reading {filepath}...")
    with open(filepath, encoding="utf-8") as f:
        text = f.read()
    all_token_ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    all_token_ids = all_token_ids[:max_tokens]
    chunks = [all_token_ids[i:i + max_length] for i in range(0, len(all_token_ids), max_length)]
    chunks = [c for c in chunks if len(c) == max_length]
    print(f"  {filename}: {len(chunks)} chunks")
    return chunks

math_proof_corpus = load_and_chunk("euclid_elements.txt")
topology_corpus   = load_and_chunk("topology.txt")
chaos_corpus      = load_and_chunk("Chaos.txt")

baudelaire_corpus = load_and_chunk("baudelaire.txt")
rimbaud_corpus    = load_and_chunk("rimbaud.txt")
breton_corpus     = load_and_chunk("breton_manifesto.txt")
tweet_corpus      = load_and_chunk("tweets_clean.txt", max_tokens=TWEET_MAX_TOKENS)

# Tweets included at ~10% weight to teach tweet-length completion
# without diluting the math-poetry voice
poetry_corpus = baudelaire_corpus + rimbaud_corpus + breton_corpus + tweet_corpus

math_corpora = {
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
for name, corpus in math_corpora.items():
    print(f"  {name}: {len(corpus)} chunks")
print()

# --- Entropy Injection ---

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



# --- Entropy Visualization ---

def visualize_entropy(tokens):
    token_array = tokens.numpy() if torch.is_tensor(tokens) else np.array(tokens)
    counts = np.bincount(token_array)
    probs  = counts[counts > 0] / len(token_array)
    entropy = -np.sum(probs * np.log2(probs))
    print(f"  Entropy: {entropy:.4f}")
    return entropy

# --- Training Loop (AMP-accelerated) ---

print("--- Phase 3: Setting up training ---")
train_data = poetry_corpus

entropy_values   = []
loss_values      = []
injection_totals = {"Euclid": 0, "Topology": 0, "Chaos": 0, "None": 0, "tokens": 0}

def train(model, train_data, epochs=EPOCHS):
    model     = model.to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    print(f"Starting training for {epochs} epochs on {len(train_data)} chunks...\n")

    for epoch in range(epochs):
        print(f"  --- Epoch {epoch + 1}/{epochs} ---")
        model.train()
        total_loss    = 0
        last_injected = None
        injection_counts = {"Euclid": 0, "Topology": 0, "Chaos": 0, "None": 0}

        for i, chunk in enumerate(train_data):
            try:
                injected, sources, tokens = inject_entropy(chunk)
                if sources:
                    for source in sources:
                        injection_counts[source] += 1
                        injection_totals[source] += 1
                    injection_totals["tokens"] += tokens
                else:
                    injection_counts["None"] += 1
                    injection_totals["None"] += 1
                last_injected = injected.cpu()
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
                    print(f"    Chunk {i + 1}/{len(train_data)} - Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"    Warning: error on chunk {i} - {e}. Skipping.")
                continue

        avg_loss = total_loss / len(train_data)
        entropy  = visualize_entropy(last_injected if last_injected is not None else train_data[0])
        entropy_values.append(entropy)
        loss_values.append(avg_loss)

        injections_made = {k: v for k, v in injection_counts.items() if k != "None"}
        print(f"  Epoch {epoch + 1} complete — Avg Loss: {avg_loss:.4f}, Entropy: {entropy:.4f}")
        print(f"  Injections: {injections_made}")

        if entropy < 0.7:
            print(f"  ⚠️  LOW ENTROPY WARNING at epoch {epoch + 1}.")
        print()

    print(f"Training complete after {epoch + 1} epochs.\n")
    return model

print()
model = train(model, train_data, epochs=EPOCHS)

# --- Plot Entropy ---

print("--- Phase 4: Plotting entropy and loss ---")

def plot_entropy_over_time(entropy_values):
    epochs = [i + 1 for i, e in enumerate(entropy_values) if not np.isnan(e)]
    values = [e for e in entropy_values if not np.isnan(e)]

    min_val = min(values)
    max_val = max(values)
    padding = max((max_val - min_val) * 0.5, 0.005)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    fig.subplots_adjust(hspace=0.08)

    ax_top.bar(epochs, values, color="steelblue", edgecolor="navy", alpha=0.85)
    ax_top.set_ylim(min_val - padding, max_val + padding)
    ax_top.set_ylabel("Shannon Entropy (zoomed)", fontsize=12)
    ax_top.set_title("Entropy Evolution During Training", fontsize=15)
    ax_top.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))

    ax_bot.bar(epochs, values, color="steelblue", edgecolor="navy", alpha=0.85)
    ax_bot.set_ylim(0, 1.2)
    ax_bot.axhline(y=0.7, color="r", linestyle="--", linewidth=1.5)
    ax_bot.text(0.5, 0.72, "Low entropy warning threshold (0.7)", fontsize=10, color="red")
    ax_bot.set_ylabel("Shannon Entropy (full scale)", fontsize=12)
    ax_bot.set_xlabel("Epoch", fontsize=13)
    ax_bot.set_xticks(epochs)

    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(bottom=False)

    d = 0.012
    kw = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1)
    ax_top.plot((-d, +d), (-d, +d), **kw)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kw)
    kw.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kw)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)

plot_entropy_over_time(np.array(entropy_values))
plt.savefig(f"entropy_plot_{RUN_TS}.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Entropy plot saved as entropy_plot_{RUN_TS}.png\n")

def plot_loss_over_time(loss_values):
    epochs = list(range(1, len(loss_values) + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, loss_values, "o-", color="darkorange", linewidth=2, markersize=6)
    ax.fill_between(epochs, loss_values, alpha=0.15, color="darkorange")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Average Loss", fontsize=13)
    ax.set_title("Training Loss Over Time", fontsize=15)
    ax.set_xticks(epochs)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)

plot_loss_over_time(loss_values)
plt.savefig(f"loss_plot_{RUN_TS}.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Loss plot saved as loss_plot_{RUN_TS}.png\n")

# --- Injection Summary + Hyperparameters ---

total_injected = sum(v for k, v in injection_totals.items() if k not in ("None", "tokens"))
print("--- Injection Summary (all epochs) ---")
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
print(f"  Tweet max tokens:    {TWEET_MAX_TOKENS:,}")
print(f"  Temperature:         {TEMPERATURE}")
print(f"  Top-p:               {TOP_P}")
print(f"  Repetition penalty:  {REPETITION_PENALTY}")
print(f"  Max new tokens:      {MAX_NEW_TOKENS}")
print()

# --- Phase 5: Sample Outputs ---

print("--- Phase 5: Generating example outputs ---\n")

# Words banned from generation — known overfitting tells
_banned_words = [
    "iced",     " iced",     "Iced",     " Iced",
    " Tiresia", " Tiresias", "Tiresius", " Tiresius",
     " race",     
    "Lisbon",   " Lisbon", " Sosostris", " Sosostros",
    "Palmyra", " Palmyra",
    " leer", " bread", " pillage",
    " Rangers",
]
bad_words_ids = [tokenizer.encode(w, add_prefix_space=False) for w in _banned_words]

prompts = [
    # --- mathy ---
    "Let x be defined as",
    "Proof by contradiction:",
    "The empty set",
    "Suppose there exists",
    "It follows necessarily that",
    "The boundary",
    "The limit",
    "Assume the ",
    "There exists no such",
    # --- existential / philosophical tweets ---
    "what is love",
    "everything feels ",
    "do you ever think ",
    "consciousness is ",
    "after death, ",
    # --- math bait ---
    "did you know",
    "infinity",
    "god exists",
    # --- provocations ---
    "an algorithm",
    "prove",
    "say something beautiful",
    "i don't believe",
    # --- casual / human moments ---
    "I can't sleep",
    "my heart",
    "I feel like",
    "dream",
    # --- surrealist openers ---


    "god is",

    "the moon",
]

model.eval()
for prompt in prompts:
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
                no_repeat_ngram_size=4,  # Uncomment to hard-block repetitive phrases
            )
        print(f"Prompt:  {prompt}")
        print(f"Output:  {tokenizer.decode(output[0], skip_special_tokens=True)}")
        print()
    except Exception as e:
        print(f"Error on '{prompt}': {e}\n")

print("--- Done! ---")
print("(Model NOT saved — run save_model.py when ready to save.)")

print(f"\n{'='*50}")
print(f"  SEED: {SEED}")
print(f"  Copy this into save_model_mauk.py to reproduce.")
print(f"{'='*50}\n")