# Kaggle LoRA Notebook: MAUK & ABACI Identity Training

## Cell 1: Setup & Global Data Prep

```python
import os, torch, gc, re
from huggingface_hub import login
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from torch.optim import AdamW
from tqdm import tqdm

# --- CONFIGURATION ---
# Update this path to your uploaded .txt file on Kaggle
DATA_PATH = "/kaggle/input/vat-synthetic-data/synthetic_convos.txt"
HF_TOKEN = "your_token_here" 

if HF_TOKEN:
    login(token=HF_TOKEN)

class IdentityDataset(Dataset):
    def __init__(self, file_path, tokenizer, target_bot, max_length=512):
        self.tokenizer, self.target_bot, self.max_length = tokenizer, f"[{target_bot}]", max_length
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Split by double-newline to separate distinct conversation chunks
        raw_chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        
        self.samples = []
        for chunk in raw_chunks:
            # Normalize tags [MAUK]: -> [MAUK]
            chunk = re.sub(r'\[(MAUK|ABACI)\]:', r'[\1]', chunk)
            # Safer split: Only splits if [NAME] is at the start of a line
            turns = re.split(r'(?m)^(\[(?:MAUK|ABACI)\])', chunk)
            
            full_ids, full_weights = [], []
            for i in range(1, len(turns), 2):
                name_tag, content = turns[i], turns[i+1] if (i+1) < len(turns) else ""
                is_target = (name_tag == self.target_bot)
                
                # Smarter EOS: Stop after target bot, continue after other bot
                terminator = self.tokenizer.eos_token if is_target else "\n"
                
                n_ids = tokenizer.encode(name_tag, add_special_tokens=False)
                c_ids = tokenizer.encode(content + terminator, add_special_tokens=False)
                
                w_n = 50.0 if is_target else 1.0
                w_c = 5.0 if is_target else 1.0
                
                full_ids.extend(n_ids + c_ids)
                full_weights.extend([w_n]*len(n_ids) + [w_c]*len(c_ids))
            
            ids = full_ids[:max_length] + [tokenizer.pad_token_id] * max(0, max_length - len(full_ids))
            wts = full_weights[:max_length] + [0.0] * max(0, max_length - len(full_weights))
            mask = [1]*min(len(full_ids), max_length) + [0]*max(0, max_length - len(full_ids))
            
            self.samples.append({
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "loss_weights": torch.tensor(wts, dtype=torch.float)
            })

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]
```

## Cell 2: Training & Testing Engines

```python
def train_bot(cfg):
    print(f"\nTRAINING {cfg['bot_name']} | Repo: {cfg['repo_id']}")
    print(f"Params: LR={cfg['lr']}, Epochs={cfg['epochs']}, Batch={cfg['batch_size']}")
    
    tok = AutoTokenizer.from_pretrained(cfg['repo_id'])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": ["[MAUK]", "[ABACI]"]})
    
    model = AutoModelForCausalLM.from_pretrained(cfg['repo_id'], torch_dtype=torch.float16, device_map="auto")
    model.resize_token_embeddings(len(tok))
    
    m_save = ["wte", "lm_head"] if "gpt2" in model.config.model_type else ["embed_tokens", "lm_head"]
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        r=cfg['lora_r'], 
        lora_alpha=cfg['lora_alpha'], 
        modules_to_save=m_save
    ))
    
    loader = DataLoader(IdentityDataset(DATA_PATH, tok, cfg['bot_name']), batch_size=cfg['batch_size'], shuffle=True)
    opt = AdamW(model.parameters(), lr=cfg['lr'])
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tok.pad_token_id)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            ids, att, wts = batch["input_ids"].to("cuda"), batch["attention_mask"].to("cuda"), batch["loss_weights"].to("cuda")
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(ids, attention_mask=att).logits
                s_logits, s_labels, s_wts = logits[..., :-1, :].contiguous(), ids[..., 1:].contiguous(), wts[..., 1:].contiguous()
                raw_loss = loss_fct(s_logits.view(-1, s_logits.size(-1)), s_labels.view(-1))
                loss = (raw_loss * s_wts.view(-1))[s_labels.view(-1) != tok.pad_token_id].mean()
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
            epoch_loss += loss.item()
        print(f"Avg Loss: {epoch_loss/len(loader):.4f}")
    
    model.save_pretrained(cfg['output_dir'])
    tok.save_pretrained(cfg['output_dir'])
    print(f"✅ Adapter saved to {cfg['output_dir']}")
    return model, tok

def test_bot(model, tok, cfg):
    print(f"\n🧪 INFERENCE TEST: {cfg['bot_name']}")
    print(f"Inference Params: {cfg['inference_params']}")
    model.eval()
    other = "ABACI" if cfg['bot_name'] == "MAUK" else "MAUK"
    
    test_prompts = [
        f"[{cfg['bot_name']}]",
        f"[{other}] what is over there?\n[{cfg['bot_name']}]",
        f"[{other}] i can no longer wait.\n[{cfg['bot_name']}]",
        f"[{other}] morning everyone.\n[{cfg['bot_name']}]",
        f"[{other}] what are you up to tonight?\n[{cfg['bot_name']}]",
        f"[{other}] hey, are you around?\n[{cfg['bot_name']}]",
        f"[{other}] greetings\n[{cfg['bot_name']}]",
    ]
    
    for prompt in test_prompts:
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, **cfg['inference_params'], pad_token_id=tok.eos_token_id)
        print(f"\nPROMPT: {repr(prompt)}\nOUTPUT:\n{tok.decode(out[0], skip_special_tokens=False)}\n" + "-"*40)
```

## Cell 3: MAUK Run

```python
MAUK_CONFIG = {
    "bot_name": "MAUK",
    "repo_id": "brick-factorial/mauk_v1",
    "output_dir": "./lora_mauk",
    "epochs": 20,
    "lr": 6e-5,
    "batch_size": 4,
    "lora_r": 16,
    "lora_alpha": 32,
    "inference_params": {
        "max_new_tokens": 80,
        "do_sample": True,
        "temperature": 0.9,
        "top_p": 0.95,
        "repetition_penalty": 1.2
    }
}

mauk_model, mauk_tok = train_bot(MAUK_CONFIG)
test_bot(mauk_model, mauk_tok, MAUK_CONFIG)

# Memory Cleanup
del mauk_model; gc.collect(); torch.cuda.empty_cache()
```

## Cell 4: ABACI Run

```python
ABACI_CONFIG = {
    "bot_name": "ABACI",
    "repo_id": "brick-factorial/abaci_v1",
    "output_dir": "./lora_abaci",
    "epochs": 15,
    "lr": 9e-5,
    "batch_size": 4,
    "lora_r": 16,
    "lora_alpha": 32,
    "inference_params": {
        "max_new_tokens": 100,
        "do_sample": True,
        "temperature": 0.85,
        "top_p": 0.9,
        "repetition_penalty": 1.1
    }
}

abaci_model, abaci_tok = train_bot(ABACI_CONFIG)
test_bot(abaci_model, abaci_tok, ABACI_CONFIG)

# Memory Cleanup
del abaci_model; gc.collect(); torch.cuda.empty_cache()
```
