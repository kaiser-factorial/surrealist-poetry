# Surrealist Entropy Project

**GPT-2 fine-tuned on Baudelaire, injected with Euclidean geometry + topology + set theory.**

---

## What is this?

This project explores what happens when a GPT-2 language model is fine-tuned on Baudelaire's poetry and simultaneously injected with mathematical proof texts — Euclid's Elements, topology, and set theory — at the token level during training.

The result is a model that hallucinates in the space between romantic poetry and formal mathematics, producing outputs that are variously:
- Lyrically beautiful
- Coherently deranged
- Spontaneously theological (GPT-2 reaches for Aquinas and biblical citation when trying to bridge math and poetry)
- Occasionally just counting to 307

The best outputs are auto-posted to Twitter/X: [@mauk_gpt](https://x.com/mauk_gpt) *(profile pic: MC Escher's impossible staircase, in turquoise)*

---

## How it works

### Training

The core mechanism is **entropy injection**: during fine-tuning, mathematical proof tokens (from Euclid, topology, and set theory texts) are randomly inserted into poetry sequences at a configurable probability (`INJECT_PROB`). The model learns to navigate between the two registers — and the collisions are where things get interesting.

**Poetry corpus:** Baudelaire — *Les Fleurs du Mal* + *Poems in Prose* (~29,000 tokens)

**Math corpora:**
| Corpus | Source | Tokens |
|--------|--------|--------|
| Euclid's Elements | Project Gutenberg | ~122,000 |
| Topology | Wikipedia (12 articles) | ~79,000 |
| Set Theory | Wikipedia (12 articles) | ~88,000 |

**Training script:** `SurrealistInjection_c4_10x_Entropy.py`

Run on a school desktop with an NVIDIA RTX 4090 (24GB VRAM).

### Generation

Mathematical prompts are used to elicit outputs:

```
"The angle of longing is"
"The proof collapses into"
"Between two points, grief"
"What the axiom cannot prove,"
"All boundaries dissolve into"
"The derivative of longing is"
"That which cannot be bisected"
```

The model fills in the rest.

---

## Selected outputs

> *"The angle of longing is like a stone and you feel the weight in your shoulders."*

> *"On the satin back of the avalanche soft, my soul descends."*

> *"Time devours our lives like an opal."*

> *"The proof collapses into an infinite infinity of dimensions, so that there remains neither a sphere nor the heavens."*

> *"Given that all triangles contain one and only two points, there can be no doubt but to Him who hath drawn them from forth the lines of infinity."*

> *"To this day I weep with envy at those women who have made vain efforts towards marriageability."*

> *"Defecate loudly thus as one might grope barefoot on the ground before another, till at last your pride evaporates into nothing less than blasphemy."*

> *"nought can satisfy her craving for chaos more than jest."*

> *"In short—Lady Macbeth my sweetie—I abominate wanton swain."*

---

## Key discoveries

- **GPT-2 theologizes spontaneously** — without any religious training data, the model fills the gap between mathematics and poetry with theology (Aquinas, biblical citations, "Almighty Being"). Comes from GPT-2's WebText base training.
- **`lr=2e-4` is the sweet spot** — produces the richest Baudelaire absorption without losing mathematical bleed-through.
- **Overfitting vocabulary** — rare/archaic Baudelaire words ("efflorescence", "mien", "iced") become disproportionately weighted at high learning rates.
- **Topology and theology share vocabulary** — *infinity, indeterminacy, unity, containment* map naturally onto theological language.
- **"The empty set contains" triggers JavaScript** — "set contains" are programming terms in GPT-2's training data. Accidentally very surrealist.

---

## Repository structure

```
poetry_consciousness/
├── SurrealistInjection_c4_10x_Entropy.py   # Main training script
├── SurrealistEntropyProject/               # Helper scripts (clean_html, scrape_wikipedia, etc.)
├── Baudelaire/                             # Poetry corpus (HTML source)
├── Euclid_Elements/                        # Math corpus (HTML source)
├── poetry_sources/                         # Additional poetry texts
├── Training_Outputs/                       # Model evaluation outputs (train1–train14)
├── Writeups/
│   ├── Session1_Writeup.md                 # Session 1 notes (Runs 1–6)
│   ├── Session2_Writeup.md                 # Session 2 notes (Runs 7–14 + Twitter deployment)
│   └── ParameterLog.md                     # Full parameter history
├── twitter/
│   ├── auto_tweet.py                       # Playwright-based auto-posting script
│   └── setup_auth.py                       # One-time X login session setup
└── CLI_Reference_Mac.md                    # Terminal reference
```

---

## Twitter automation

Outputs are posted automatically to [@mauk_gpt](https://x.com/mauk_gpt) via a Playwright browser automation script (`twitter/auto_tweet.py`) running on a cron schedule (9x/day with random delays for organic timing).

Tweets are manually curated from training outputs for quality — 118 queued as of April 2026.

---

## Next steps

- Add Rimbaud and Breton to the poetry corpus
- Run with expanded corpus — watch whether anti-clerical Breton energy pushes back against GPT-2's theological tendencies
- Try flipping injection direction: math as base, poetry injected
- Experiment with `epochs=20` once overfitting is controlled
