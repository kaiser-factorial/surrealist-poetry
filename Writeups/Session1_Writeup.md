# Surrealist Entropy Project — Session 1 Writeup
**Date:** 2026-03-30

---

## Project Overview

This project explores what happens when a GPT-2 language model is fine-tuned on a hybrid corpus of surrealist poetry (Baudelaire, Rimbaud, Breton) and mathematical texts (Euclid's Elements, Topology, Set Theory). The core experimental mechanism is **entropy injection** — randomly inserting tokens from mathematical texts into poetry sequences during training, and monitoring how this affects the model's outputs and the statistical randomness (entropy) of the generated text.

---

## What We Accomplished

### 1. Environment Setup
- Created the project directory structure:
  ```
  poetry_consciousness/
  ├── SurrealistEntropyProject/
  │   ├── PythonScripts - Mac/
  │   └── PythonScripts - Windows/
  ├── Euclid_Elements/
  ├── Baudelaire/
  ├── poetry_sources/
  ├── Writeups/
  └── TransferToSchool/
  ```
- Identified the school computer specs:
  - **CPU:** 32 logical processors
  - **RAM:** ~64 GB
  - **GPU:** NVIDIA GeForce RTX 4090 (24GB VRAM)
  - **OS:** Windows 11 (64-bit)
- Confirmed PyTorch detects and uses the RTX 4090 via CUDA
- Created CLI reference guides for Mac and Windows (`CLI_Reference_Mac.md`, `CLI_Reference_Windows.md`)

---

### 2. Dataset Compilation

**Poetry Corpora:**
| Corpus | Source | Tokens | Status |
|--------|--------|--------|--------|
| Baudelaire (*Les Fleurs du Mal* + *Poems in Prose*) | Project Gutenberg HTML | ~29,000 | ✅ Active |
| Rimbaud (*A Season in Hell*) | Project Gutenberg HTML | ~12,000 | ✅ Ready (not yet in training) |
| Breton (*Manifesto of Surrealism*) | PDF | ~7,000 | ✅ Ready (not yet in training) |

**Math Corpora:**
| Corpus | Source | Tokens | Status |
|--------|--------|--------|--------|
| Euclid's Elements | Project Gutenberg HTML | ~122,000 | ✅ Active (trimmed to max_tokens) |
| Topology | Wikipedia API (12 articles) | ~79,000 | ✅ Active |
| Set Theory | Wikipedia API (12 articles) | ~88,000 | ✅ Active |

**Data pipeline:**
- HTML → plain text via `clean_html.py` (BeautifulSoup)
- PDF → plain text via `extract_pdf.py` (pdfminer)
- Wikipedia → plain text via `scrape_wikipedia.py` (requests + Wikipedia API)
- All texts tokenized and split into 1024-token chunks (GPT-2 max sequence length)

---

### 3. Script Development (`SurrealistInjection_c4_10x_Entropy.py`)

Started from a partially faulty script provided by the project mentor. Key issues identified and fixed:

| Issue | Fix |
|-------|-----|
| Invalid shebang (`#!/usr/bin/env jax`) | Changed to `#!/usr/bin/env python3` |
| Broken tokenizer call (`transformers.load_pretrained tokenizer`) | Replaced with `AutoTokenizer.from_pretrained()` |
| JAX/Flax and PyTorch used together incompatibly | Removed JAX/Flax entirely, rewrote in pure PyTorch |
| `loss_fn_apply` undefined | GPT-2 now computes its own loss via `labels=input_ids` |
| `jax.numpy` aliased as `np` (shadowing NumPy) | Separated then removed JAX entirely |
| `chain` never imported | Added `from itertools import chain` |
| `batch_size` defined but never used | Removed |
| `tokenizer_use='partner'` (not a real parameter) | Replaced with standard HuggingFace tokenizer args |
| `visualize_entropy` not returning a value | Added `return entropy` |
| `matplotlib` imported locally inside function | Moved to top-level imports |
| Entropy always measured on same static chunk | Now measured on last injected sequence per epoch |
| Entropy halt stopped training | Changed to `⚠️ LOW ENTROPY WARNING` — training continues |
| GPT-2 pad token not set | Added `tokenizer.pad_token = tokenizer.eos_token` |
| Plot x-axis misaligned when NaN values filtered | Fixed to use correct epoch numbers |

---

### 4. Final Script Structure

Runs in five phases with printed progress at every step:

1. **Phase 1 — Model Loading:** Loads GPT-2 from HuggingFace. Auto-detects GPU.
2. **Phase 2 — Dataset Loading:** Reads all corpus `.txt` files, tokenizes, splits into 1024-token chunks.
3. **Phase 3 — Training:** Fine-tunes GPT-2 using AdamW. At each chunk, math tokens are randomly injected into poetry sequences. Shannon entropy measured per epoch. Injection source (Euclid/Topology/Set Theory) logged each epoch.
4. **Phase 4 — Plots:** Saves `entropy_plot.png` (broken y-axis histogram) and `loss_plot.png` directly to working directory — no blocking window.
5. **Phase 5 — Output Generation:** Generates text from prompt set, saves to terminal output.

---

### 5. Training Runs Completed

| Run | lr | Epochs | Key Change | Best Output |
|-----|----|--------|------------|-------------|
| 1 | 5e-5 | 10 | Initial run, Baudelaire + Euclid only | *"we can prove that beauty alone is the cause for our love"* |
| 2 | 5e-5 | 10 | Mathematical prompts | *"infinite kisses, infinite rugs, infinite mists / Where each in turn renders the other a part equal"* |
| 3 | 5e-5 | 10 | INJECT_PROB → 0.15 | *"SEXY ETERNAL HOMECRAFT"* |
| 4 | 5e-5 | 10 | Added Topology + Set Theory, restored poetic structure | *"O God's love can be broken!"* |
| 5 | 1e-5 | 10 | Sparse emotional prompts, \n trick removed | *"The proof collapses into an infinite infinity of dimensions"* |
| 6 | 2e-4 | 10 | max_tokens → 29,000, best run yet | *"All boundaries dissolve into morose chords... Of all my senses blent in one!"* |

---

### 6. Key Discoveries

- **GPT-2 theologizes spontaneously** — without any religious training data, the model filled the gap between mathematics and poetry with theology (Aquinas, biblical citations, "Almighty Being"). Comes from WebText base training.
- **Agatha Christie appeared twice** — in emotionally resonant outputs, from GPT-2 base knowledge. Disappeared at `lr=2e-4` when Baudelaire's voice became dominant.
- **Overfitting vocabulary:** "efflorescence", "iced", "mien" — rare/archaic Baudelaire words appearing disproportionately at high learning rates. See Recurring Word Tracker in `ParameterLog.md`.
- **"The empty set contains"** triggered JavaScript code — "set contains" are programming terms in GPT-2's training data.
- **`\n` prompt trick ineffective** — sparse, hanging, mid-thought prompts work better for eliciting poetic line structure.
- **`lr=2e-4` was the sweet spot** — produced the richest outputs but with mild overfitting signals. `lr=1e-4` currently being tested as a middle ground.

---

### 7. Supporting Files Created

| File | Purpose |
|------|---------|
| `clean_html.py` | Converts Gutenberg HTML files to clean plain text |
| `extract_pdf.py` | Extracts text from PDF files (used for Breton Manifesto) |
| `scrape_wikipedia.py` | Scrapes and saves Wikipedia articles as plain text |
| `CLI_Reference_Mac.md` | Terminal command reference for Mac |
| `CLI_Reference_Windows.md` | PowerShell command reference for Windows |
| `ParameterLog.md` | Full history of all runs, parameters, and observations |
| `Session1_Writeup.md` | This document |

---

## Key Decisions Made

- **PyTorch over JAX/Flax:** Original script mixed both incompatibly. PyTorch chosen for simpler GPT-2 integration.
- **Three math corpora:** Euclid + Topology + Set Theory — randomly selected at injection time for richer mathematical vocabulary.
- **Poetry as base, math as injection:** Poetry is the training base; math tokens are injected into it. Baudelaire's voice dominates with math bleeding through in flashes.
- **Entropy as warning, not halt:** Training continues below 0.7 threshold with a logged warning.
- **No virtual machine:** Training runs directly on school desktop (RTX 4090).
- **Broken y-axis histogram for entropy plot:** Tiny entropy variations made standard line plot unreadable.
- **Plots saved to file, not displayed:** Prevents script blocking mid-run.
- **`bad_words_ids` not used:** Decided against hard-blocking repeated words — prefer to address via hyperparameter tuning.

---

## Next Steps

- [ ] Complete Run 7 with `lr=1e-4` — check if overfitting words disappear
- [ ] Add Rimbaud and Breton to poetry corpus (files ready in `TransferToSchool/`)
- [ ] Run with expanded poetry corpus, raise `max_tokens` to ~50,000 to balance with math corpora
- [ ] Try `epochs=20` once overfitting is under control
- [ ] Experiment with flipping injection direction — math as base, poetry injected
- [ ] Share best outputs with mentor — especially *"Where the function breaks, / O God's love can be broken!"*
