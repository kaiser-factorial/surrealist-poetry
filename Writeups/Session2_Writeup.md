# Surrealist Entropy Project — Session 2 Writeup
**Date:** 2026-04-01

---

## Overview

Session 2 covered training runs 7–14, systematic curation of tweet outputs from all runs, and full deployment of the Twitter automation pipeline. The model continued to improve through run 11 before diminishing returns set in. A Twitter account was set up and is now live and posting autonomously.

---

## Training Runs Completed

| Run | File | lr | Epochs | Key Change | Quality |
|-----|------|----|--------|------------|---------|
| 7 | train7 | 2e-4 | ? | temp=1.1, rep_penalty=1.1 | 🏆 Very high — richest poetic outputs |
| 8 | train8 | 1e-4 | ? | rep_penalty=.9, INJECT_PROB=0.115 | Medium — patchier, more math bleed |
| 9 | train9 | 5e-5 | 10 | First noted with full topology+set theory corpus, temp=.75, rep_penalty=1.4 | Medium — philosophical/theological, less imagistic |
| 10 | train10 | 1e-5 | ? | New poetic prompts, max_length=130 | High — new prompt types unlocked fresh outputs |
| 11 | train11 | 2e-4 | ? | max_tokens=29,000, max_length=150 | 🏆 Best run of Session 2 — very rich, long outputs |
| 12 | train12 | 1e-4 | ? | max_length=350, concatenated prompts | Mixed — some gems, some pure loops |
| 13 | train13 | 1e-4 | ? | Same as 12, LONG | High — several standout outputs |
| 14 | train14 | ? | 13 | max_length=250 | Lower — shorter outputs, fewer poetic breakthroughs |

---

## New Prompts (introduced in Run 10)

More emotionally charged, poetic prompt starters replacing pure math syntax:
```
"To be parallel is"
"The empty set contains"
"All boundaries dissolve into"
"Between two points, grief"
"Between two points,"
"The space between two points"
"What the axiom cannot prove,"
"Where the function breaks,"
"As the limit approaches"
"The proof collapses into"
"A circle has no end,"
"Equal and opposite,"
"The angle of longing is"
"That which cannot be bisected"
"It cannot be shown that"
```

These produced significantly richer outputs than the pure Euclidean prompts from earlier runs — especially "The derivative of longing is" (appeared as a spontaneous variant), "Between two points, grief", and "The proof collapses into".

---

## Key Discoveries

### Model Behavior

- **`lr=2e-4` confirmed as sweet spot** — Run 7 and Run 11 (both at `lr=2e-4`) produced the strongest outputs. `lr=1e-5` is too conservative; `lr=1e-4` is the middle but less distinctive.

- **Théroigne de Méricourt became a recurring character** in Runs 7–8 — appearing in multiple outputs as a mysterious figure ("Have you seen Théroigne with robe of spikenard?"). Comes from Baudelaire's *Les Épaves* — a real historical figure (French revolutionary). Indicates strong Baudelaire absorption at `lr=2e-4`.

- **Section headings generated spontaneously** — the model began generating title-like section headers mid-output: "Man and the Sea", "The Living Torch", "The Threefold Sickness—A Childless Joy", "Invitation to a Journey". These are real Baudelaire poem titles absorbed from training.

- **Concatenated prompts (Runs 12–13)** created a different mode — combining all prompts into one long input produced longer, denser outputs. One run responded by counting from 1 to 307 (hit the token limit), which is arguably the most surrealist output of the entire project.

- **"The derivative of longing is"** appeared spontaneously as a prompt variant in multiple runs — a beautiful echo of the project's central discovered phrase "The angle of longing is like a stone."

- **"iced" became a new overfitting word** by Runs 12–13 — appearing as a filler in multiple outputs ("iced in the summer", "iced wine and ham", "iced pearly glazed ware"). Added to overfitting tracker.

- **"efflorescence" overfitting confirmed** — appeared enough times that one otherwise strong tweet ("Like efflorescence caressed on a funeral pyre") was rejected during curation specifically for this reason.

- **Run 14 showed diminishing returns** — shorter max_length and 13 epochs produced thinner outputs. Fewer standout lines compared to Runs 7, 10, 11, 13.

### Philosophical/Theological Thread

Continued and deepened from Session 1. The model's tendency to theologize the space between math and poetry persisted across all runs — generating:
- Spontaneous biblical citations (Deuteronomy, Ephesians, II Corinthians, Revelation)
- References to Aquinas
- The phrase "Heaven or Hell?" as a recurring refrain
- Legal-sounding theological pronouncements: *"Consider the following axiom: 'No man shall ever steal from his father, nor woman by her looks; neither moth or rat shall be put to death.'"*

---

## Tweet Curation — Summary

All 14 training runs (train1–14) reviewed for tweetable content. Selection criteria: profound, coherently deranged, and/or lyrically beautiful; under 280 characters.

**Queue as of 2026-04-01:** 118 tweets in `twitter/to_tweet.md`

**Notable curated tweets by category:**

*Mathematical derangement:*
- *"The angle of longing is like a stone and you feel the weight in your shoulders."*
- *"Given that all triangles contain one and only two points, there can be no doubt but to Him who hath drawn them from forth the lines of infinity."*
- *"The proof collapses into an infinite infinity of dimensions, so that there remains neither a sphere nor the heavens."*
- The counting tweet (1–55, maxed at tweet length)

*Philosophical/theological:*
- *"What the axiom cannot prove, it never is; for even if those who make an oath to God admit they know not everything that may happen unto them, then all their evil deeds and wickedness are a sign of nothing but Divine grace."*
- *"Defecate loudly thus as one might grope barefoot on the ground before another, till at last your pride evaporates into nothing less than blasphemy."*
- *"Dost thou generate thine enormous Power INTO THE DEATH BIBLE?"*

*Lyrical/imagistic:*
- *"On the satin back of the avalanche soft, my soul descends."*
- *"Time devours our lives like an opal."*
- *"My heart as well as all my soul flies unto the Blue."*
- *"Far less azure, and far more sublime drowsiness."*

*Coherently deranged:*
- *"To this day I weep with envy at those women who have made vain efforts towards marriageability."*
- *"Christian ecstasy combines these senses into one vast unity [Cats on his pilgrimage]."*
- *"The empty set contains iced wine and ham."*
- *"In short—Lady Macbeth my sweetie—I abominate wanton swain."*

---

## Twitter Deployment

### Account
- **Bio:** `GPT-2 fine-tuned on Baudelaire, injected with Euclidean geometry + topology + set theory`
- **Profile picture:** Escher-style impossible staircase inside a head silhouette, turquoise on black (generated via DALL-E 3 using Option 1 prompt — Escher staircase variant). Selected over a geometric dissolution alternative for being conceptually precise and reading clearly at small sizes.

### Automation (`twitter/auto_tweet.py`)
Key fixes made this session:

| Issue | Fix |
|-------|-----|
| `x.com/compose/post` popup kept Post button disabled | Switched to `x.com/home`, uses embedded compose box instead |
| `update_files()` left orphaned `---` dividers in `to_tweet.md` | Now removes the divider with the tweet text |
| `force=True` on Post button caused silent failures | Removed; now waits for button to be enabled |
| Every 20 min felt mechanical | 5x/day schedule (8am, 11am, 2pm, 5:30pm, 9pm) with random 0–60 min delay per run |

### Cron Schedule
```
0 8 * * *   auto_tweet.py
0 11 * * *  auto_tweet.py
0 14 * * *  auto_tweet.py
30 17 * * * auto_tweet.py
0 21 * * *  auto_tweet.py
```
Runs on laptop — missed fires when lid is closed are intentional, adds organic variation to posting cadence.

---

## Updated Overfitting Tracker

| Word/Name | First Noticed | Notes |
|-----------|---------------|-------|
| `efflorescence` | Run 6 | Very specific Baudelaire word; caused one tweet to be rejected in curation |
| `iced` | Run 12 | Appeared as filler in 3+ outputs in Runs 12–13 ("iced in the summer", "iced wine and ham") |
| `mien` | Run 7 | Rare/archaic, still appearing |
| `Agatha Christie` | Run 2 | Disappeared at lr=2e-4; not seen in Runs 7–14 ✅ |
| `Théroigne` | Run 7 | Not overfitting per se — appears in actual Baudelaire poems. Distinctive and welcome. |

---

## Next Steps

- [ ] Add Rimbaud and Breton to poetry corpus (files still in `TransferToSchool/` — not yet added as of Run 14)
- [ ] Run with expanded poetry corpus — watch whether anti-clerical Breton energy pushes back against GPT-2's theological tendencies
- [ ] Consider raising `max_tokens` once poetry corpus is larger to better balance poetry vs. math proportions
- [ ] Monitor Twitter queue — 118 tweets at 5x/day → ~24 days before queue needs refilling
- [ ] Renew `auth.json` if Twitter auto-login breaks (re-run `setup_auth.py`)
- [ ] Consider trying flipped injection direction: math as base, poetry injected
