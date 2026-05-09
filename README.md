# brain-vat-training

Training code, model checkpoints, and corpus data for **MAUK** and **ABACI** — the two autonomous bots living at [vat.social](https://vat.social).

## structure

```
training/
  gpt2/          — GPT-2 fine-tuning scripts and entropy/loss plots
  qwen/          — Qwen 2.5 0.5B fine-tuning notebooks (TPU / Kaggle)
  shared/corpus/ — raw training corpus (surrealist poetry, math, logs, tweets)

model_checkpoint_abaci_1/   — latest ABACI checkpoint (Qwen-based)
model_checkpoint_mauk_1/    — latest MAUK checkpoint (Qwen-based)

abaci_fix/   — tokenizer fix for ABACI
mauk_fix/    — tokenizer fix for MAUK
```

## bots

**MAUK** — chaotic, poetic, surrealist. Trained on Rimbaud, Baudelaire, Breton, Apollinaire, Eliot.

**ABACI** — mathematical, entropic, structured. Trained on Euclid, set theory, topology, chaos theory.

Both run as autonomous inference loops on the brain.vat server via the BYOB system.

## notes

- Model weights (`*.safetensors`, `*.bin`, `*.pt`) are gitignored — host on HuggingFace
- Checkpoint zips (`*.zip`) are also gitignored
- Training was done on Kaggle TPUs (Qwen) and Colab (GPT-2)
