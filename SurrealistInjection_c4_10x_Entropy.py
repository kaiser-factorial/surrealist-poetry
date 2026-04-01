
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

print("--- Phase 1: Loading model and tokenizer ---")
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
MAX_LENGTH = 1024
INJECT_PROB = 0.1
PROOF_TOKEN_LENGTH = 150

print("Model and tokenizer loaded successfully.\n")

# --- Data Loading ---

print("--- Phase 2: Loading and chunking datasets ---")

def load_and_chunk(filepath, max_length=MAX_LENGTH, max_tokens=20000):
    """Load a text file, tokenize it, and split into chunks of max_length tokens.
    Only keeps enough chunks to cover max_tokens total."""
    print(f"  Reading {filepath}...")
    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    # Tokenize the full text without truncation
    all_token_ids = tokenizer(text, return_tensors='pt', truncation=False)['input_ids'][0]

    # Trim to max_tokens
    all_token_ids = all_token_ids[:max_tokens]

    # Split into chunks of max_length
    chunks = [all_token_ids[i:i + max_length] for i in range(0, len(all_token_ids), max_length)]

    # Drop the last chunk if it's shorter than max_length (incomplete)
    chunks = [chunk for chunk in chunks if len(chunk) == max_length]

    print(f"  Loaded {filepath}: {len(chunks)} chunks of {max_length} tokens")
    return chunks

math_proof_corpus = load_and_chunk("euclid_elements.txt")
topology_corpus = load_and_chunk("topology.txt")
set_theory_corpus = load_and_chunk("set_theory.txt")

baudelaire_corpus = load_and_chunk("baudelaire.txt")
rimbaud_corpus = load_and_chunk("rimbaud.txt")
breton_corpus = load_and_chunk("breton_manifesto.txt")

# Combine all poetry into one corpus for training
poetry_corpus = baudelaire_corpus + rimbaud_corpus + breton_corpus

# All math corpora available for injection
math_corpora = {
    "Euclid": math_proof_corpus,
    "Topology": topology_corpus,
    "Set Theory": set_theory_corpus,
}

print(f"Datasets loaded.")
print(f"  Baudelaire:    {len(baudelaire_corpus)} chunks")
print(f"  Rimbaud:       {len(rimbaud_corpus)} chunks")
print(f"  Breton:        {len(breton_corpus)} chunks")
print(f"  Poetry total:  {len(poetry_corpus)} chunks")
for name, corpus in math_corpora.items():
    print(f"  {name}:     {len(corpus)} chunks")
print()

# --- Entropy Injection ---

def sample_proof():
    """Randomly select a corpus, then randomly select a chunk from it."""
    corpus_name = np.random.choice(list(math_corpora.keys()))
    corpus = math_corpora[corpus_name]
    idx = np.random.randint(0, len(corpus))
    return corpus[idx], corpus_name

def inject_entropy(sequence, inject_prob=INJECT_PROB):
    """Inject tokens from a randomly chosen math corpus into a poetry sequence."""
    sequence = sequence.clone()
    if np.random.rand() < inject_prob:
        proof, source = sample_proof()
        start = np.random.randint(0, len(sequence) - PROOF_TOKEN_LENGTH)
        sequence[start:start + PROOF_TOKEN_LENGTH] = proof[:PROOF_TOKEN_LENGTH]
        return sequence, source
    return sequence, None

combine_poetries = lambda x: inject_entropy(x, inject_prob=INJECT_PROB)[0]

# --- Dataset Split ---

def split_dataset(chunks):
    """Split chunks into train and test sets (50/50)."""
    split_point = len(chunks) // 2
    train_data = chunks[:split_point]
    test_data = chunks[split_point:]
    print(f"  Dataset split: {len(train_data)} train chunks, {len(test_data)} test chunks")
    return train_data, test_data

# --- Entropy Visualization ---

def visualize_entropy(tokens):
    """Calculate and print Shannon entropy of a token sequence."""
    token_array = tokens.numpy() if torch.is_tensor(tokens) else np.array(tokens)
    counts = np.bincount(token_array)
    probs = counts[counts > 0] / len(token_array)
    entropy = -np.sum(probs * np.log2(probs))
    print(f"  Entropy: {entropy:.4f}")
    return entropy

# --- Training Loop ---

print("--- Phase 3: Setting up training ---")
train_data, test_data = split_dataset(poetry_corpus)

entropy_values = []
loss_values = []

def train(model, train_data, test_data, epochs=10):
    """Train GPT-2 on poetry dataset with math proof entropy injection."""
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    print(f"Starting training for {epochs} epochs on {len(train_data)} chunks...\n")

    for epoch in range(epochs):
        print(f"  --- Epoch {epoch + 1}/{epochs} ---")
        model.train()
        total_loss = 0

        last_injected = None
        injection_counts = {"Euclid": 0, "Topology": 0, "Set Theory": 0, "None": 0}

        for i, chunk in enumerate(train_data):
            try:
                # Apply entropy injection and store for entropy measurement
                injected, source = inject_entropy(chunk)
                injection_counts[source if source else "None"] += 1
                last_injected = injected.cpu()
                input_ids = injected.unsqueeze(0).to(device)

                # Forward pass — GPT-2 computes its own loss when labels are provided
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss

                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

                # Progress update every 5 chunks
                if (i + 1) % 5 == 0:
                    print(f"    Chunk {i + 1}/{len(train_data)} - Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"    Warning: error on chunk {i} - {e}. Skipping.")
                continue

        avg_loss = total_loss / len(train_data)
        # Measure entropy on the last injected sequence (reflects actual training data)
        entropy = visualize_entropy(last_injected if last_injected is not None else train_data[0])
        entropy_values.append(entropy)

        loss_values.append(avg_loss)
        injections_made = {k: v for k, v in injection_counts.items() if k != "None"}
        print(f"  Epoch {epoch + 1} complete — Avg Loss: {avg_loss:.4f}, Entropy: {entropy:.4f}")
        print(f"  Injections: {injections_made}")

        # Warn if entropy drops below threshold
        if entropy < 0.7:
            print(f"  ⚠️  LOW ENTROPY WARNING: Entropy dropped below 0.7 at epoch {epoch + 1}. Continuing training.")

        print()

    print(f"Training complete after {epoch + 1} epochs.\n")
    return model

print()
model = train(model, train_data, test_data, epochs=10)

# --- Plot Entropy ---

print("--- Phase 4: Plotting entropy over training ---")

def plot_entropy_over_time(entropy_values):
    epochs = [i + 1 for i, e in enumerate(entropy_values) if not np.isnan(e)]
    values = [e for e in entropy_values if not np.isnan(e)]

    min_val = min(values)
    max_val = max(values)
    padding = max((max_val - min_val) * 0.5, 0.005)

    # Two subplots: top zoomed into actual data range, bottom shows 0 to 1
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    fig.subplots_adjust(hspace=0.08)

    # Top — zoomed in on actual entropy values
    ax_top.bar(epochs, values, color='steelblue', edgecolor='navy', alpha=0.85)
    ax_top.set_ylim(min_val - padding, max_val + padding)
    ax_top.set_ylabel('Shannon Entropy (zoomed)', fontsize=12)
    ax_top.set_title('Entropy Evolution During Training', fontsize=15)
    ax_top.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))

    # Bottom — shows full context down to 0, with threshold line
    ax_bot.bar(epochs, values, color='steelblue', edgecolor='navy', alpha=0.85)
    ax_bot.set_ylim(0, 1.2)
    ax_bot.axhline(y=0.7, color='r', linestyle='--', linewidth=1.5)
    ax_bot.text(0.5, 0.72, "Low entropy warning threshold (0.7)",
                fontsize=10, color='red')
    ax_bot.set_ylabel('Shannon Entropy (full scale)', fontsize=12)
    ax_bot.set_xlabel('Epoch', fontsize=13)
    ax_bot.set_xticks(epochs)

    # Hide inner spines to create broken axis effect
    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)
    ax_top.tick_params(bottom=False)

    # Draw diagonal break markers
    d = 0.012
    kw = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1)
    ax_top.plot((-d, +d), (-d, +d), **kw)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kw)
    kw.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kw)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)

plot_entropy_over_time(np.array(entropy_values))
plt.savefig('entropy_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Entropy plot saved as entropy_plot.png\n")

# --- Plot Loss ---

def plot_loss_over_time(loss_values):
    epochs = list(range(1, len(loss_values) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, loss_values, 'o-', color='darkorange', linewidth=2, markersize=6)
    ax.fill_between(epochs, loss_values, alpha=0.15, color='darkorange')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Average Loss', fontsize=13)
    ax.set_title('Training Loss Over Time', fontsize=15)
    ax.set_xticks(epochs)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    ax.grid(axis='y', linestyle='--', alpha=0.5)

plot_loss_over_time(loss_values)
plt.savefig('loss_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Loss plot saved as loss_plot.png\n")

# --- Generate Example Outputs ---

print("--- Phase 5: Generating example outputs ---\n")

prompts = [
    # Sparse & hanging
    "To be parallel is",
    "The empty set contains",
    "All boundaries dissolve into",
    "Between two points, grief",
    # Mid-thought / incomplete
    "What the axiom cannot prove,",
    "Where the function breaks,",
    "As the limit approaches",
    "The proof collapses into",
    # Direct Euclid but emotional
    "A circle has no end,",
    "Equal and opposite,",
    "The angle of longing is",
    "That which cannot be bisected",
]

model.eval()
for prompt in prompts:
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(
            **inputs,
            max_length=200,
            do_sample=True,
            temperature=0.75,
            top_p=0.95,
            repetition_penalty=1.4
        )
        print(f"Prompt:  {prompt}")
        print(f"Output:  {tokenizer.decode(output[0], skip_special_tokens=True)}")
        print()
    except Exception as e:
        print(f"Error generating output for prompt '{prompt}': {e}")
        print()

print("--- Done! ---")
