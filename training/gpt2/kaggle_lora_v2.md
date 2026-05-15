# Kaggle LoRA Notebook v2: MAUK & ABACI Identity Training
# Key changes from v1:
#   - Sliding window dataset: one sample per TARGET turn, prior turns as masked context
#   - \n as turn-end signal instead of EOS (cleaner, unambiguous)
#   - StoppingCriteria halts generation when model tries to start next speaker's turn
#   - Fixed zero_grad ordering (was causing gradient accumulation bug)
#   - Added gradient clipping (stability with high loss weights)
#   - Added warmup + cosine LR scheduler
#   - Explicit device per bot (uses both T4s)
#   - [USER] turns handled correctly (treated as non-target context)
#   - Epoch checkpointing (Kaggle timeout protection)
#   - Multiple DATA_PATHS supported
#   - HF token via Kaggle Secrets with safe fallback

## Cell 1: Setup & Dataset

```python
import os, torch, gc, re
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList,
                          get_cosine_schedule_with_warmup)
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW
from tqdm import tqdm

# --- HF Token via Kaggle Secrets (safer than hardcoding) ---
# Requires: Kaggle notebook Settings → Add-ons → Secrets → add key "HF_TOKEN"
# Also make sure "Internet" is ON in notebook settings (needed to download from HF)
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")  # fallback to env var

if HF_TOKEN and HF_TOKEN != "ADD HF TOKEN":
    from huggingface_hub import login
    login(token=HF_TOKEN)
    print("✅ Logged in to HuggingFace")
else:
    print("⚠️  No HF token found — model downloads may fail if repos are private")

# --- Data paths: list all training files you want to include ---
# Upload each .txt to a Kaggle dataset and update paths here
DATA_PATHS = [
    "/kaggle/input/synthetic-convos/synthetic_convos.txt",
    # "/kaggle/input/synthetic-convos/synthetic_convos_v2_draft.txt",
    # "/kaggle/input/synthetic-convos/synthetic_convos_v2b_draft.txt",
    # "/kaggle/input/synthetic-convos/bash_logs_cleaned.txt",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class IdentityDataset(Dataset):
    """
    Sliding-window dataset: one training sample per TARGET BOT turn.

    For a conversation [A, B, A, B], training samples for ABACI are:
        sample 1 — context: [A turn 1]       → generate: [B turn 1]
        sample 2 — context: [A1, B1, A2]     → generate: [B turn 2]

    Context tokens get loss_weight=0 (model reads but doesn't predict them).
    Target tokens get high weights so the model learns WHAT to say and WHEN to stop.

    Turn-end signal: '\n' at the end of each line (NOT <|endoftext|>).
    The model learns: produce my response → '\n' → yield to the other speaker.

    NOTE: The sliding window produces ~2-3x more samples than the v1 approach.
    Consider reducing epoch count accordingly to avoid over-training.
    """

    def __init__(self, file_paths, tokenizer, target_bot, max_length=512, loss_weights=None):
        self.tokenizer   = tokenizer
        self.target_tag  = f"[{target_bot}]"
        self.max_length  = max_length
        self.loss_weights = loss_weights or {"name_tag": 75.0, "content": 7.0, "turn_end": 50.0}
        self.samples     = []

        if isinstance(file_paths, str):
            file_paths = [file_paths]  # accept single path too

        all_chunks = []
        for path in file_paths:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
            all_chunks.extend(chunks)
            print(f"  Loaded {len(chunks)} chunks from {os.path.basename(path)}")

        for chunk in all_chunks:
            turns = self._parse_turns(chunk)
            for i, (tag, content) in enumerate(turns):
                if tag != self.target_tag:
                    continue
                self._build_sample(turns[:i], tag, content, tokenizer)

        print(f"[{target_bot}] → {len(self.samples)} training samples "
              f"from {len(all_chunks)} chunks across {len(file_paths)} file(s)")

    # -----------------------------------------------------------------------
    def _parse_turns(self, chunk):
        """Return list of (tag, content) for every speaker turn in chunk."""
        # Handle both '[TAG]: text' and '[TAG] text' formats
        chunk = re.sub(r'\[(MAUK|ABACI|USER)\]:\s*', r'[\1] ', chunk)
        parts = re.split(r'(?m)^(\[(?:MAUK|ABACI|USER)\])', chunk)
        turns = []
        for i in range(1, len(parts), 2):
            tag     = parts[i]
            content = parts[i + 1].strip() if (i + 1) < len(parts) else ""
            if content:
                turns.append((tag, content))
        return turns

    # -----------------------------------------------------------------------
    def _build_sample(self, context_turns, tgt_tag, tgt_content, tok):
        """
        Encode context (weight=0) + target turn (weighted).
        Left-truncates context to fit max_length, always preserving the full target.
        """
        # --- context: zero weight (model reads, doesn't learn to predict) ---
        ctx_ids, ctx_wts = [], []
        for c_tag, c_content in context_turns:
            t_ids = tok.encode(c_tag + " ", add_special_tokens=False)
            c_ids = tok.encode(c_content + "\n", add_special_tokens=False)
            ctx_ids += t_ids + c_ids
            ctx_wts += [0.0] * (len(t_ids) + len(c_ids))

        # --- target turn: high weight ---
        tag_ids  = tok.encode(tgt_tag + " ", add_special_tokens=False)
        cont_ids = tok.encode(tgt_content + "\n", add_special_tokens=False)

        # Safety: clamp target alone if it somehow exceeds max_length
        tgt_ids = (tag_ids + cont_ids)[:self.max_length]
        lw = self.loss_weights
        tgt_wts = (
            [lw["name_tag"]] * len(tag_ids)        +  # name tag — identity anchor
            [lw["content"]]  * (len(cont_ids) - 1) +  # response content
            [lw["turn_end"]]                           # final \n — "I'm done, your turn"
        )[:self.max_length]

        # --- left-truncate context to fit ---
        budget = self.max_length - len(tgt_ids)
        if budget <= 0:
            # Target alone fills the window — drop all context
            ctx_ids, ctx_wts = [], []
        elif len(ctx_ids) > budget:
            # Keep the MOST RECENT context (drop oldest)
            ctx_ids = ctx_ids[-budget:]
            ctx_wts = ctx_wts[-budget:]

        all_ids = ctx_ids + tgt_ids
        all_wts = ctx_wts + tgt_wts
        seq_len = len(all_ids)
        pad_len = self.max_length - seq_len  # guaranteed >= 0 now

        self.samples.append({
            "input_ids":      torch.tensor(
                all_ids + [tok.pad_token_id] * pad_len, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1] * seq_len + [0] * pad_len, dtype=torch.long),
            "loss_weights":   torch.tensor(
                all_wts + [0.0] * pad_len, dtype=torch.float),
        })

    def __len__(self):           return len(self.samples)
    def __getitem__(self, idx):  return self.samples[idx]
```

## Cell 2: Stopping Criteria

```python
# ---------------------------------------------------------------------------
# Stopping criteria: halt when model starts writing the next speaker's tag
# ---------------------------------------------------------------------------
class TurnEndCriteria(StoppingCriteria):
    """
    Stop generation the moment the model produces '\n' followed by a speaker
    special token — it's trying to start the next speaker's turn.

    Why NOT '\n[': [MAUK], [ABACI], [USER] were added as single special tokens,
    so the model generates them as one token ID (e.g. 50257), NOT as the two
    character tokens '\n' (198) + '[' (58). Checking for [198, 58] would never
    match and the criteria would silently never fire.
    """
    def __init__(self, tokenizer, prompt_len):
        self.prompt_len = prompt_len
        newline_id  = tokenizer.encode("\n", add_special_tokens=False)[0]
        speaker_ids = [
            tokenizer.convert_tokens_to_ids(tag)
            for tag in ["[MAUK]", "[ABACI]", "[USER]"]
            if tokenizer.convert_tokens_to_ids(tag) != tokenizer.unk_token_id
        ]
        # Set of (newline_id, speaker_id) pairs to stop on
        self.stop_pairs = set((newline_id, sid) for sid in speaker_ids)

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] <= self.prompt_len + 2:
            return False
        tail = tuple(input_ids[0, -2:].tolist())
        return tail in self.stop_pairs
```

## Cell 3: Training Engine

```python
# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_bot(cfg):
    emoji = "🩵" if cfg['bot_name'] == "MAUK" else "🟠"
    print(f"\n{emoji} TRAINING {cfg['bot_name']} on {cfg['device']} | Repo: {cfg['repo_id']}")

    tok = AutoTokenizer.from_pretrained(cfg['repo_id'])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": ["[MAUK]", "[ABACI]", "[USER]"]})

    model = AutoModelForCausalLM.from_pretrained(
        cfg['repo_id'], torch_dtype=torch.float32
    ).to(cfg['device'])

    model.resize_token_embeddings(len(tok))
    if getattr(model.config, "tie_word_embeddings", False):
        model.tie_weights()

    m_save = ["wte"] if "gpt2" in model.config.model_type else ["embed_tokens"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg['lora_r'],
        lora_alpha=cfg['lora_alpha'],
        target_modules=cfg.get('lora_modules', ["c_attn", "c_proj", "c_fc"]),
        modules_to_save=m_save,
        fan_in_fan_out=True,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = IdentityDataset(
        DATA_PATHS, tok, cfg['bot_name'],
        max_length=cfg.get('max_length', 512),
        loss_weights=cfg.get('loss_weights'),
    )
    loader  = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)
    opt     = AdamW(model.parameters(), lr=cfg['lr'])

    total_steps  = len(loader) * cfg['epochs']
    warmup_steps = len(loader) * cfg.get('warmup_epochs', 1)
    scheduler    = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tok.pad_token_id)
    scaler   = torch.amp.GradScaler(cfg['device'].split(':')[0])

    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            ids = batch["input_ids"].to(cfg['device'])
            att = batch["attention_mask"].to(cfg['device'])
            wts = batch["loss_weights"].to(cfg['device'])

            opt.zero_grad()  # ✅ zero BEFORE forward pass

            with torch.amp.autocast(cfg['device'].split(':')[0], dtype=torch.float16):
                logits   = model(ids, attention_mask=att).logits
                s_logits = logits[..., :-1, :].contiguous()
                s_labels = ids[..., 1:].contiguous()
                s_wts    = wts[..., 1:].contiguous()

                raw_loss = loss_fct(
                    s_logits.view(-1, s_logits.size(-1)),
                    s_labels.view(-1)
                )
                loss = (raw_loss * s_wts.view(-1))[
                    s_labels.view(-1) != tok.pad_token_id
                ].mean()

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('grad_clip', 1.0))
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        lr  = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch+1}/{cfg['epochs']} | Loss: {avg:.4f} | LR: {lr:.2e}")

        # ✅ Epoch checkpoint — protects against Kaggle session timeout
        save_every = cfg.get('save_every', 5)
        if (epoch + 1) % save_every == 0:
            ckpt = f"{cfg['output_dir']}_epoch{epoch+1}"
            model.save_pretrained(ckpt)
            tok.save_pretrained(ckpt)
            print(f"  💾 Checkpoint → {ckpt}")

    # Final save
    print(f"\n{emoji} TRAINED {cfg['bot_name']} on {cfg['device']} | Repo: {cfg['repo_id']}")
    model.save_pretrained(cfg['output_dir'])
    tok.save_pretrained(cfg['output_dir'])
    print(f"✅ {cfg['bot_name']} saved → {cfg['output_dir']}")
    return model, tok
```

## Cell 4: Testing Engine

```python
# ---------------------------------------------------------------------------
# Inference test
# ---------------------------------------------------------------------------
def test_bot(model, tok, cfg):
    print(f"\n🧪 INFERENCE TEST: {cfg['bot_name']}")
    print(f"   Epochs={cfg['epochs']} | LR={cfg['lr']} | r={cfg['lora_r']}")
    model.eval()

    other = "ABACI" if cfg['bot_name'] == "MAUK" else "MAUK"
    bot   = cfg['bot_name']

    # Prompts end with "[BOT] " (trailing space) to match training format,
    # where every tag is encoded as "[TAG] " before its content.
    test_prompts = [
        f"[{bot}] I want to tell you something.\n[{other}] What is it?\n[{bot}] ",
        f"[{other}] what is over there?\n[{bot}] ",
        f"[{other}] i can no longer wait.\n[{bot}] ",
        f"[{bot}] Did you hear the good news?\n[{other}] No, what is happening?\n[{bot}] ",
        f"[{other}] what do you think is happening?\n[{bot}] ",
        f"[{bot}] I think we have an issue.\n[{other}] What might that be?\n[{bot}] ",
        f"[{other}] greetings\n[{bot}] ",
        f"[{bot}] ",
        f"[{other}] It is not yet time.\n[{bot}] ",
    ]

    for prompt in test_prompts:
        inputs   = tok(prompt, return_tensors="pt").to(cfg['device'])
        criteria = TurnEndCriteria(tok, prompt_len=inputs["input_ids"].shape[1])

        with torch.no_grad():
            out = model.generate(
                **inputs,
                **cfg['inference_params'],
                pad_token_id=tok.eos_token_id,
                stopping_criteria=StoppingCriteriaList([criteria]),
            )

        # Decode only newly generated tokens
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        generated  = tok.decode(new_tokens, skip_special_tokens=False)

        # Strip any partial next-speaker tag — use re.DOTALL to handle
        # multi-line generated output correctly
        generated = re.sub(r'\n\[.*', '', generated, flags=re.DOTALL).strip()

        print(f"\nPROMPT : {repr(prompt)}")
        print(f"OUTPUT : {repr(generated)}")
        print("-" * 60)
```

## Cell 5: MAUK Run

```python
# NOTE: With the sliding window dataset, effective sample count is ~2-3x larger
# than v1. If you notice overfitting (loss near 0 early), reduce epochs.
MAUK_CONFIG = {
    "bot_name"  : "MAUK",
    "repo_id"   : "brick-factorial/mauk_v1",
    "output_dir": "./lora_mauk",
    "device"    : "cuda:0",     # pinned to GPU 0

    # --- training schedule ---
    "epochs"       : 15,
    "lr"           : 2e-4,
    "batch_size"   : 4,
    "warmup_epochs": 1,         # how many epochs to linearly warm up LR over
    "save_every"   : 5,         # checkpoint every N epochs

    # --- sequence length ---
    "max_length"   : 512,       # tokens per sample (context + target); lower = faster, less context

    # --- LoRA ---
    "lora_r"      : 32,         # rank — higher = more capacity, more VRAM
    "lora_alpha"  : 64,         # scale = lora_alpha / lora_r; keep at 2x r as a rule of thumb
    "lora_modules": ["c_attn", "c_proj", "c_fc"],  # which GPT-2 layers get LoRA adapters
                                # c_attn = attention QKV, c_proj = attention output, c_fc = FFN

    # --- loss weights ---
    # These shape what the model prioritises. Raise name_tag if MAUK keeps forgetting who it is;
    # raise turn_end if outputs are too long; raise content if responses feel generic.
    "loss_weights": {
        "name_tag": 75.0,       # [MAUK] token — identity anchor
        "content" :  7.0,       # response body
        "turn_end": 50.0,       # final \n — "I'm done, your turn"
    },

    # --- gradient clipping ---
    "grad_clip": 1.0,           # max gradient norm; lower if loss spikes, raise if learning is slow

    # --- inference ---
    "inference_params": {
        "max_new_tokens"    : 80,
        "do_sample"         : True,
        "temperature"       : 0.8,   # higher = more creative/unpredictable
        "top_p"             : 0.95,  # nucleus sampling cutoff
        "repetition_penalty": 1.2,   # penalises repeating tokens; raise if outputs loop
    },
}

mauk_model, mauk_tok = train_bot(MAUK_CONFIG)
test_bot(mauk_model, mauk_tok, MAUK_CONFIG)

del mauk_model; gc.collect(); torch.cuda.empty_cache()
```

## Cell 6: ABACI Run

```python
ABACI_CONFIG = {
    "bot_name"  : "ABACI",
    "repo_id"   : "brick-factorial/abaci_v1",
    "output_dir": "./lora_abaci",
    "device"    : "cuda:1",     # pinned to GPU 1

    # --- training schedule ---
    "epochs"       : 17,
    "lr"           : 2e-4,
    "batch_size"   : 4,
    "warmup_epochs": 1,
    "save_every"   : 5,

    # --- sequence length ---
    "max_length"   : 512,

    # --- LoRA ---
    "lora_r"      : 32,
    "lora_alpha"  : 64,
    "lora_modules": ["c_attn", "c_proj", "c_fc"],

    # --- loss weights ---
    "loss_weights": {
        "name_tag": 75.0,
        "content" :  7.0,
        "turn_end": 50.0,
    },

    # --- gradient clipping ---
    "grad_clip": 1.0,

    # --- inference ---
    "inference_params": {
        "max_new_tokens"    : 100,
        "do_sample"         : True,
        "temperature"       : 0.8,
        "top_p"             : 0.95,
        "repetition_penalty": 1.2,
    },
}

abaci_model, abaci_tok = train_bot(ABACI_CONFIG)
test_bot(abaci_model, abaci_tok, ABACI_CONFIG)

del abaci_model; gc.collect(); torch.cuda.empty_cache()
```
