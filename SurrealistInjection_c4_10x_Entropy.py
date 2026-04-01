
#!/usr/bin/env jax
from flax import linen as nn
import jax.numpy as np
import transformers

def load_tokenizer(model_name):
    tokenizer_class = transformers.load_pretrained tokenizer
    return tokenizer_class.from_pretrained(model_name)

# Tokenizer for mathematical proofs
math_proof_tokenizer = load_tokenizer("mathematical_proofs_corpus")

# Original poetry tokenizer (assuming we're using something like the Baudelaire dataset)
surrealist_poetry_tokenizer = load_tokenizer("baudelaire_dataset")

# Constants for controlling entropy injection into the training loop
INJECT_PROB = 0.1  # Probability of inserting a mathematical proof at each timestep
PROOF_TOKEN_LENGTH = 50  # Average length, in tokens, of the mathematical proofs

def sample_proof(math_proof_tokenizer):
    """Randomly select and tokenize a mathematical proof."""
    while True:
        proof_text = np.random.choice(math_proof_corpus).strip()
        if len(proof_text) > PROOF_TOKEN_LENGTH:
            continue
        math_tokens = math_proof_tokenizer(tokenizer_use='partner', text=[proof_text])
        return math_tokens

def inject_entropy(sequence, inject_prob=INJECT_PROB):
    """Inject a mathematical proof into the poetry sequence."""
    for i in range(len(sequence)):
        if np.random.rand() < inject_prob:
            partner = sample_proof(math_proof_tokenizer)
            sequence[i] = partner
    return sequence

# Function to be used during training, combining both poetries
combine_poetries = lambda x: inject_entropy(x, inject_prob=INJECT_PROB)

# Utility for loading and tokenizing the entire dataset in advance
math_proof_corpus = list(chain.from_iterable(math_proof_tokenizer("mathematical_proofs_dataset")))
poetry_corpus = list(chain.from_iterable(surrealist_poetry_tokenizer("baudelaire_dataset")))

# Tokenized corpora
tokenized_math_proofs = math_proof_tokenizer(tokenizer_use='partner', text=math_proof_corpus)
tokenized_poetry = surrealist_poetry_tokenizer(tokenizer_use='partner', text=poetry_corpus)

dataset = (np.array(tokenized_math_proofs + tokenized_poetry),
           np.where([t == 1 for t in tokenized_math_proofs[0] + tokenized_poetry[0]],
                    np.ones_like(np.array(tokenized_math_proofs)[0] + np.array(tokenized_poetry)[0]),
                    np.zeros_like(np.array(tokenized_math_proofs)[0] + np.array(tokenized_poetry)[0])))

print(f"Dataset loaded with {len(dataset[1]):,} examples.")

# Training loop setup and entropy injection visualization
train_step = lambda params, key, math_poetry_combined, batch: (
    loss_fn_apply(model.apply({'params': params},
                              key=key,
                              math_poetry=combine_poetries(math_poetry_combined),
                              train=True),
                  batch) / batch_size)

def split_dataset(dataset):
    """
    Returns two random splits of the provided dataset.
    The second half will be used for injecting entropy into poetry.
    """
    randperm = np.random.permutation(len(dataset[1]))
    split_point = len(randperm)//2
    train_data = (dataset[0][randperm[:split_point]],
                  dataset[1][randperm[:split_point]])
    test_data = (dataset[0][randperm[split_point:]],
                 dataset[1][randperm[split_point:]])
    return train_data, test_data

def visualize_entropy(math_poetry_combined):
    """Visualize the entropy injected into poetry."""
    math_proof_tokens, poetry_tokens = math_poetry_combined
    entropy = -np.sum([p * np.log2(p) for p in np.unique(np.array(poetry_tokens),
                                                        return_counts=True)[0]]) / len(poetry_tokens)
    print(f"Entropy after injection: {entropy:.4f}")

# Training function
def train(model, train_data, test_data, epochs=3):
    """Train the model on both poetries with entropy injection."""
    key = jax.random.PRNGKey(0)
    for epoch in range(epochs):
        # Visualize and potentially log entropy before each epoch for comparison
        visualize_entropy(train_data)

        # Training step function across all training data (not batched here for simplicity)
        train_step(jax.random.randint(0, 2**16, model.nparams()),
                    key,
                    dataset[1],
                    (dataset[0], np.ones_like(dataset[1])))

        # Periodically test model outputs to monitor entropy injection quality
    print(f"Training complete after {epochs} epochs.")


# Function to plot entropy of poetry over time during training
def plot_entropy_over_time(entropy_values):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(entropy_values)), [e for e in entropy_values if not np.isnan(e)], 'o-', linewidth=2)
    plt.axhline(y=0.7, color='r', linestyle='-')
    plt.text(10, 0.72, "Entropy threshold", fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Entropy in Poetry', fontsize=14)
    plt.title('Entropy Evolution During Training', fontsize=16)

# Utility to store entropy during training
entropy_values = []

def entropy_hook(key, batch):
  math_poetry = batch[0]
  entropy_values.append(visualize_entropy(math_poetry))

train_data, test_data = split_dataset(dataset)
model = train(model, train_data, test_data, epochs=10)

# Plot final entropy values over epochs
plot_entropy_over_time(np.array(entropy_values))
plt.show()