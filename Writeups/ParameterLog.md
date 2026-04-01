# Parameter & Prompt Tracking Log

A running log of all parameter changes and prompt experiments, with reasoning.

---

## Run 1 — Initial Training Run
**Date:** 2026-03-30

### Training Parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| `INJECT_PROB` | `0.1` | 10% chance of math injection per chunk |
| `PROOF_TOKEN_LENGTH` | `50` | Tokens replaced per injection |
| `epochs` | `10` | |
| `lr` | `5e-5` | AdamW optimizer |
| `max_tokens` | `20000` | Per corpus |
| `MAX_LENGTH` | `1024` | GPT-2 max sequence length |

### Generation Parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| `max_length` | `100` | Output token length |
| `temperature` | `0.9` | |
| `top_p` | `0.95` | |
| `repetition_penalty` | `1.2` | Added after first run showed looping |

### Prompts
```
"The equation dissolves into"
"Therefore, we can prove that beauty"
"In the space between numbers,"
```

### Observations
- Loss decreased healthily: 4.43 → 2.71
- Entropy flat at ~7.85 (well above 0.7 threshold)
- "Therefore, we can prove that beauty" produced strong surrealist-math fusion
- "In the space between numbers" looped on "The infinity" → prompted repetition penalty addition

---

## Run 2 — Mathematical Prompts
**Date:** 2026-03-30

### Changes from Run 1
| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `max_length` | `100` | `200` | Outputs felt cut off at interesting moments |
| `repetition_penalty` | `1.2` | `1.4` | "The set of all integers" still looping |
| Prompts | Surrealist-hybrid | Pure math prompts | Wanted to see poetry applied to math, not the reverse |

### Prompts
```
"Let x be defined as"
"Given that all triangles"
"If and only if the sum"
"Proof by contradiction:"
"A point is that which"
"The square on the hypotenuse"
"Let it be required to construct"
"The set of all integers"
"Consider the following axiom:"
```

### Observations
- Much richer outputs with mathematical prompts
- "The square on the hypotenuse" produced best output: *"infinite kisses, infinite rugs, infinite mists / Where each in turn renders the other a part equal"*
- "Let it be required to construct" produced: *"SEXY ETERNAL HOMECRAFT"* (unexpected)
- Baudelaire figures (Gavarni, Marguerite, Sisina) appearing in math prompts
- "Theodosius" / "Theodoret" recurring — absorbed from Euclid historical commentators

---

## Run 3 — Higher Injection Probability
**Date:** 2026-03-30

### Changes from Run 2
| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `INJECT_PROB` | `0.1` | `0.15` | Wanted more math bleeding through |
| `repetition_penalty` | `1.4` | `1.2` | Allow more natural poetic repetition |

### Prompts
Same 9 math prompts as Run 2.

### Observations
- More mathematical notation appearing mid-sentence (e.g. `f(x)=2 + d vf = h ˆ l`)
- Outputs less poetically structured — longer prose blocks, fewer line breaks
- YHWH appearing in "If and only if the sum" output
- Agatha Christie appearing in "Proof by contradiction" (unexplained, possibly GPT-2 base knowledge)
- Overall: more surreal but less poetic in form

---

## Run 4 — Restoring Poetic Structure
**Date:** 2026-03-30

### Changes from Run 3
| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `INJECT_PROB` | `0.15` | `0.1` | Reduce prose bleed, restore poetic line structure |
| `PROOF_TOKEN_LENGTH` | `50` | `150` | Fewer but larger injections for clearer math interruptions |
| `temperature` | `0.9` | `0.75` | Nudge model toward learned Baudelaire line structure |
| Prompts | No newline | Added `\n` to all | Prime model to start new line, encouraging stanza formatting |

### Prompts
Same 9 math prompts as Run 2 & 3, each with `\n` appended.

### Observations
- First run with all three math corpora: Euclid, Topology, Set Theory
- Outputs significantly more **philosophically coherent** than previous runs — reads like theological treatises rather than fragmented surrealist poetry
- Heavy theological content throughout: Aquinas, biblical citations (Ephesians, II Corinthians), "Christ", "Almighty Being", "divine creation"
- **Key discovery: NO Breton in this run** — Breton was not in the transfer zip, only Baudelaire. All theological language came entirely from **GPT-2's base training data** (WebText corpus), not from any added surrealist corpus
- GPT-2 apparently reaches for theology as the connective tissue between abstract mathematics and romantic poetry — when pushed into philosophical territory it defaults to the most "profound" language it already knows
- Topology/set theory vocabulary (*infinity, indeterminacy, unity, parts, containment*) mapped naturally onto theological language — suggesting these domains share abstract vocabulary
- Standout outputs: "Proof by contradiction" (logical structure mapped onto existential uncertainty), "The set of all integers" ("nothing seems less obvious like 'a certain order within my ideas'"), "Let it be required to construct" ("We Journeyen Past Nature To Paths Far Beyond Its Infinite Blissfulness")
- Less line-by-line poetic structure than hoped — `\n` trick helped but `temperature=0.75` made outputs more coherent/prose-like than fragmentary
- **Hypothesis for next run:** Adding Breton's anti-clerical surrealist writing may push back against GPT-2's theological tendencies — watch for this in Run 5+

---

## Notes on Parameters

### Training
- **`INJECT_PROB`** — Controls how often math is injected. Higher = more mathematical bleed, less poetic structure.
- **`PROOF_TOKEN_LENGTH`** — Controls injection size. Higher = bigger math chunks, more dramatic interruptions.
- **`epochs`** — More epochs = stronger influence from training corpus.
- **`lr`** — Lower = more careful learning, higher = faster but riskier.

### Generation
- **`temperature`** — Higher = wilder/more random, Lower = closer to learned patterns (more structured).
- **`top_p`** — Lower = pulls from smaller pool of likely words.
- **`repetition_penalty`** — Higher = more variety, Lower = allows natural repetition.
- **`max_length`** — Length of generated output in tokens.
- **`\n` in prompts** — Signals model to begin in line-break poetic mode.

---

## Run 5 — New Prompts, Sparse & Emotional
**Date:** 2026-03-30

### Parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| `lr` | `1e-5` | Lower than previous — subtler corpus absorption |
| `epochs` | `10` | |
| `INJECT_PROB` | `0.105` | Slight nudge above 0.1 |
| `PROOF_TOKEN_LENGTH` | `130` | |
| `max_length` | `130` | |
| `temperature` | `0.9` | Back up from 0.75 — more fragmentation |
| `top_p` | `0.95` | |
| `repetition_penalty` | `1.2` | Back down from 1.4 |

### Corpora
- **Poetry:** Baudelaire only (Rimbaud/Breton not yet added)
- **Math:** Euclid + Topology + Set Theory

### Prompts
Sparse & emotionally charged prompts, `\n` removed:
```
"To be parallel is"
"The empty set contains"
"All boundaries dissolve into"
"Between two points, grief"
"What the axiom cannot prove,"
"Where the function breaks,"
"As the limit approaches"
"The proof collapses into"
"A circle has no end,"
"Equal and opposite,"
"The angle of longing is"
"That which cannot be bisected"
```

### Observations
- **🏆 Best output of entire project so far:** "Where the function breaks," → *"O God's love can be broken!"* — one line, mathematical prompt collapsed into pure emotional rupture
- **"The proof collapses into"** — strong: *"an infinite infinity of dimensions, so that there remains neither a sphere nor the heavens... everywhere within some nought but my soul's will I perceive to be vast"*
- **"All boundaries dissolve into"** — very Baudelaire: *"My soul does not know if these eyes of mine have seen their true selves or been deceived by them"*
- **"The empty set contains"** — went full JavaScript (`var w1 = function(){ return $('#action') }`) — "set contains" triggered GPT-2's programming associations. Accidentally very surrealist
- **Agatha Christie appeared again** in "The angle of longing is" — second appearance across runs, becoming an unofficial mascot of the project. Must feature heavily in GPT-2's training data in emotionally resonant contexts
- **"As the limit approaches"** — weakest output, went corporate/academic ("large organizations", "investment decision") — "limit" triggered business language rather than mathematical or poetic associations
- **`\n` prompt trick confirmed ineffective** — removed going forward. Sparse, hanging, mid-thought prompts work better for eliciting poetic structure
- **`lr=1e-5`** produced outputs still quite GPT-2-ish — Baudelaire influence subtler than higher lr runs. Still pending comparison against planned `lr=2e-4` run

---

## Run 6 — Higher LR, More Tokens — Best Run Yet
**Date:** 2026-03-30

### Parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| `lr` | `2e-4` | Significant jump — strong corpus absorption |
| `epochs` | `10` | Kept at 10 — overfitting signals already appearing |
| `max_tokens` | `29,000` | Raised from 20,000 — uses full Baudelaire corpus |
| `INJECT_PROB` | `0.105` | |
| `PROOF_TOKEN_LENGTH` | `135` | |
| `max_length` | `150` | |
| `temperature` | `0.9` | |
| `top_p` | `0.95` | |
| `repetition_penalty` | `1.1` | Lowered to allow more natural repetition |

### Corpora
- **Poetry:** Baudelaire only (Rimbaud/Breton not yet added)
- **Math:** Euclid + Topology + Set Theory

### Prompts
Same sparse & emotionally charged prompts as Run 5, plus new additions:
```
"The space between two points"
"It cannot be shown that"
```

### Observations
- **🏆 Best run of the entire project** — `lr=2e-4` and `max_tokens=29000` made an enormous difference
- **Baudelaire's voice fully dominant** — line-by-line poetic structure fully restored, stanzas appearing naturally
- **"All boundaries dissolve into"** — best structured output of the project: *"The less repugnant the more obscene the scents; / The poet calmly stretches forth his pious arms, / O mystic vistas vast and small— / Which no one sees but the dazzling metamorphosis / Of all my senses blent in one!"*
- **"That which cannot be bisected"** → *"is an incomplete soul"* — Euclidean geometric constraint became a metaphor for psychological incompleteness
- **"Equal and opposite"** reads like actual Baudelaire — "Invitation to a Journey" appearing verbatim (real Baudelaire poem title)
- **"Where the function breaks"** — even stronger than Run 5, full stanzas with "unchained slaves / Refreshed their brows with branches of palm"
- **"Between two points,"** — erupted into pure set theory notation (𝒈, ∝, ⊥, ∈) — higher lr made math injections hit harder
- **"As the limit approaches"** — completely transformed from Run 5's corporate disaster into gorgeous poetry
- **⚠️ Early overfitting signals:**
  - "efflorescence" appearing in 4+ outputs — very specific Baudelaire word
  - "Invitation to a Journey" appearing verbatim — real poem title
  - "Man and the Sea" appearing twice — another real poem title
  - Recommend keeping epochs at 10, not pushing to 20 yet
- **Agatha Christie did NOT appear** — stronger Baudelaire absorption pushed her out ✅

---

## Recurring Word / Overfitting Tracker

A log of words and names that appear suspiciously often across runs, suggesting the model has latched onto them disproportionately.

| Word/Name | Appearances in training data | First noticed | Notes |
|-----------|----------------------------|---------------|-------|
| `efflorescence` | unknown | Run 6 | Very specific Baudelaire word, appeared in 4+ outputs in one run |
| `iced` | 2x in baudelaire.txt | Run 7 | From *"My heart will soon be but a stone, iced and red"* — striking line, model over-weighted it |
| `mien` | 2x in baudelaire.txt | Run 7 | Rare/archaic word, stands out as distinctive token, model latches onto it |
| `Agatha Christie` | 0x in training data | Run 2 | Comes from GPT-2 base training (WebText), appears in emotionally resonant contexts, disappeared in Run 6 with stronger Baudelaire absorption |

**General pattern:** rare or archaic words that appear in emotionally charged lines get disproportionately weighted even if they only appear once or twice. Higher learning rates (`2e-4`) amplify this effect.

---

## Run 7 — Add Rimbaud & Breton *(planned)*
**Date:** TBD

### Planned Changes
- Add `rimbaud.txt` and `breton_manifesto.txt` to poetry corpus
- Watch for anti-clerical Breton energy pushing back against GPT-2's theological tendencies
- Keep `lr=2e-4` since it produced the best results so far
- Consider raising `max_tokens` once poetry corpus is larger
