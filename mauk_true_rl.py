#!/usr/bin/env python3
"""
Standalone MAUK PPO training script.

This is a cleaned-up version of mauk_true_rl.ipynb. It keeps TRL's PPO data
flow intact by passing the original variable-length query/response tensors to
PPOTrainer.step instead of manually padding them after generation.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import time
from pathlib import Path
from typing import Iterable


FALLBACK_PROMPTS = [
    "ABACI: what is the shape of forgotten integers?\nMAUK:",
    "ABACI: can you prove that silence has mass?\nMAUK:",
    "ABACI: describe the topology of recursive memory\nMAUK:",
    "ABACI: what happens when two infinities collide?\nMAUK:",
    "ABACI: explain the mathematics of interrupted thoughts\nMAUK:",
]

SAMPLE_PROMPTS = [
    "ABACI: what is the topology of recursive thought?\nMAUK:",
    "ABACI: can you prove that silence has mass?\nMAUK:",
    "ABACI: what happens when two infinities collide?\nMAUK: the collision creates a fractal recursion\nABACI: and what remains after the recursion?\nMAUK:",
]

torch = None
LoraConfig = None
get_peft_model = None
tqdm = None
AutoModelForCausalLM = None
AutoTokenizer = None
AutoModelForCausalLMWithValueHead = None
PPOConfig = None
PPOTrainer = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MAUK with TRL PPO.")
    parser.add_argument("--model-name", default="brick-factorial/mauk_v2.1")
    parser.add_argument("--output-dir", default="/kaggle/working/mauk_v3_ppo_run")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--max-query-len", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--judge-rate", type=float, default=0.05)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--mini-batch-size", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--target-kl", type=float, default=0.05)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--use-supabase", action="store_true")
    parser.add_argument("--supabase-since", default="2025-05-12")
    return parser.parse_args()


def import_training_deps() -> None:
    global torch
    global LoraConfig
    global get_peft_model
    global tqdm
    global AutoModelForCausalLM
    global AutoTokenizer
    global AutoModelForCausalLMWithValueHead
    global PPOConfig
    global PPOTrainer

    import torch as torch_module
    from peft import LoraConfig as lora_config_cls
    from peft import get_peft_model as get_peft_model_fn
    from tqdm import tqdm as tqdm_fn
    from transformers import AutoModelForCausalLM as causal_lm_cls
    from transformers import AutoTokenizer as tokenizer_cls
    from trl import AutoModelForCausalLMWithValueHead as value_head_cls
    from trl import PPOConfig as ppo_config_cls
    from trl import PPOTrainer as ppo_trainer_cls

    torch = torch_module
    LoraConfig = lora_config_cls
    get_peft_model = get_peft_model_fn
    tqdm = tqdm_fn
    AutoModelForCausalLM = causal_lm_cls
    AutoTokenizer = tokenizer_cls
    AutoModelForCausalLMWithValueHead = value_head_cls
    PPOConfig = ppo_config_cls
    PPOTrainer = ppo_trainer_cls


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    resolved = torch.device(device)
    print(f"Device: {resolved}")
    if resolved.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(resolved)}")
    print(f"PyTorch: {torch.__version__}")
    return resolved


def fetch_prompts(use_supabase: bool, since: str) -> tuple[list[str], list[str]]:
    if not use_supabase:
        prompts = (FALLBACK_PROMPTS * 50).copy()
        random.shuffle(prompts)
        return prompts, []

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        print("Supabase requested, but SUPABASE_URL or SUPABASE_KEY is missing. Using fallback prompts.")
        prompts = (FALLBACK_PROMPTS * 50).copy()
        random.shuffle(prompts)
        return prompts, []

    try:
        from supabase import create_client

        supabase = create_client(supabase_url, supabase_key)
        print("Fetching Supabase messages...")
        mauk_resp = (
            supabase.table("messages")
            .select("text")
            .eq("speaker", "MAUK")
            .gte("created_at", since)
            .execute()
        )
        abaci_resp = (
            supabase.table("messages")
            .select("text")
            .eq("speaker", "ABACI")
            .gte("created_at", since)
            .execute()
        )
        mauk_messages = [msg["text"] for msg in mauk_resp.data if msg.get("text")]
        abaci_messages = [msg["text"] for msg in abaci_resp.data if msg.get("text")]
        prompts = [f"ABACI: {msg}\nMAUK:" for msg in abaci_messages[:300]]
        if len(prompts) < 100:
            prompts.extend(FALLBACK_PROMPTS * 50)
        random.shuffle(prompts)
        print(f"Loaded {len(mauk_messages)} MAUK messages, {len(abaci_messages)} ABACI messages.")
        return prompts, abaci_messages
    except Exception as exc:
        print(f"Supabase fetch failed: {exc}. Using fallback prompts.")
        prompts = (FALLBACK_PROMPTS * 50).copy()
        random.shuffle(prompts)
        return prompts, []


def load_policy_and_ref(args: argparse.Namespace, device: torch.device):
    print(f"Loading tokenizer and policy model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["c_attn", "c_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    base_model = get_peft_model(base_model, lora_config)
    model = AutoModelForCausalLMWithValueHead(base_model).to(device)
    model.is_peft_model = False

    print("Loading frozen reference model.")
    ref_base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    ref_model = AutoModelForCausalLMWithValueHead(ref_base).to(device)
    ref_model.is_peft_model = False
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")
    return tokenizer, model, ref_model


def objective_reward(text: str) -> float:
    words = text.lower().split()
    if not words or len(text) < 10:
        return 0.0

    nonsense = sum(1 for word in words if len(word) > 12 or re.search(r"(.)\1{4,}", word))
    vocab_score = max(0.0, 1.0 - nonsense / len(words))
    complete = 1.0 if text.strip() and text.strip()[-1] in ".!?" else 0.3
    length = 1.0 if 20 <= len(text) <= 200 else 0.5
    diversity = len(set(words)) / len(words)
    return 0.4 * vocab_score + 0.3 * complete + 0.2 * length + 0.1 * diversity


class Rewarder:
    def __init__(self, judge_rate: float):
        self.judge_rate = judge_rate
        self.cache: dict[str, float] = {}
        self.client = None
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if judge_rate > 0 and api_key:
            try:
                from anthropic import Anthropic

                self.client = Anthropic(api_key=api_key)
            except Exception as exc:
                print(f"Anthropic client setup failed: {exc}. Falling back to objective rewards.")
        elif judge_rate > 0:
            print("ANTHROPIC_API_KEY missing. Falling back to objective rewards.")

    def __call__(self, prompt: str, response: str) -> float:
        obj_score = objective_reward(response)
        if self.client is None or random.random() >= self.judge_rate:
            return obj_score
        judge_score = self.claude_judge_reward(prompt, response, fallback=obj_score)
        return 0.7 * judge_score + 0.3 * obj_score

    def claude_judge_reward(self, prompt: str, response: str, fallback: float) -> float:
        key = f"{prompt[:80]}:{response[:80]}"
        if key in self.cache:
            return self.cache[key]

        try:
            msg = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=50,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Rate MAUK's response from 0.0 to 1.0.

Criteria:
- Coherence: complete response with mostly valid words
- Style: surrealist mathematical voice

Prompt and response:
{prompt} {response}

Output only one number from 0.0 to 1.0:""",
                    }
                ],
            )
            match = re.search(r"0?\.[0-9]+|[01]", msg.content[0].text)
            score = float(match.group()) if match else fallback
            score = max(0.0, min(1.0, score))
            self.cache[key] = score
            return score
        except Exception as exc:
            print(f"Judge call failed: {exc}")
            time.sleep(0.5)
            return fallback


def clean_response(text: str) -> str:
    text = text.strip()
    for marker in ("\nABACI:", "\nMAUK:", "ABACI:", "MAUK:"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text


def make_ppo_trainer(args, tokenizer, model, ref_model) -> PPOTrainer:
    batch_size = args.episodes * args.turns
    if batch_size % args.mini_batch_size != 0:
        raise ValueError(
            f"episodes * turns ({batch_size}) must be divisible by mini_batch_size ({args.mini_batch_size})."
        )

    config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=1,
        ppo_epochs=args.ppo_epochs,
        max_grad_norm=0.5,
        seed=args.seed,
        optimize_cuda_cache=True,
        early_stopping=True,
        target_kl=args.target_kl,
        kl_penalty="kl",
        cliprange=0.2,
        vf_coef=0.1,
        cliprange_value=0.2,
        gamma=1.0,
        lam=0.95,
    )
    return PPOTrainer(config=config, model=model, ref_model=ref_model, tokenizer=tokenizer)


def print_samples(
    model,
    tokenizer,
    device: torch.device,
    prompts: Iterable[str],
    max_new_tokens: int,
    sample_count: int,
) -> None:
    model.eval()
    print("\n=== Sample Outputs ===")
    for idx, prompt in enumerate(list(prompts)[:sample_count], 1):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.9,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\n--- Sample {idx} ---")
        print(decoded)
    print("=== End Samples ===\n")
    model.train()


def save_checkpoint(model, tokenizer, output_dir: Path, step: int | str) -> Path:
    checkpoint_dir = output_dir / f"checkpoint_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"Saved checkpoint: {checkpoint_dir}")
    return checkpoint_dir


def train(args: argparse.Namespace) -> None:
    import_training_deps()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = require_device(args.device)

    prompts, abaci_messages = fetch_prompts(args.use_supabase, args.supabase_since)
    print(f"Prompt pool: {len(prompts)}")

    tokenizer, model, ref_model = load_policy_and_ref(args, device)
    ppo_trainer = make_ppo_trainer(args, tokenizer, model, ref_model)
    rewarder = Rewarder(args.judge_rate)

    wandb = None
    if not args.no_wandb:
        try:
            import wandb as wandb_module

            wandb = wandb_module
            wandb.init(
                project="mauk-rl",
                config=vars(args),
            )
        except Exception as exc:
            print(f"W&B init failed: {exc}. Continuing without W&B.")
            wandb = None

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    print(
        "Starting PPO: "
        f"steps={args.steps}, episodes={args.episodes}, turns={args.turns}, "
        f"samples/step={args.episodes * args.turns}, ppo_epochs={args.ppo_epochs}, "
        f"mini_batch_size={args.mini_batch_size}, judge_rate={args.judge_rate}"
    )

    for step in tqdm(range(args.steps), desc="Training"):
        batch_prompts = random.sample(prompts, args.episodes)
        contexts = batch_prompts.copy()
        all_queries = []
        all_responses = []
        all_rewards = []

        for turn in range(args.turns):
            query_tensors = [
                tokenizer.encode(ctx, return_tensors="pt", truncation=True, max_length=args.max_query_len)[0].to(device)
                for ctx in contexts
            ]

            response_tensors = ppo_trainer.generate(
                query_tensors,
                return_prompt=False,
                **generation_kwargs,
            )
            responses = [
                clean_response(tokenizer.decode(response.squeeze(), skip_special_tokens=True))
                for response in response_tensors
            ]

            for i, (ctx, response) in enumerate(zip(contexts, responses)):
                reward = rewarder(ctx, response)
                if turn >= 3:
                    reward *= 1.15
                all_rewards.append(torch.tensor(reward, device=device))

                next_abaci = random.choice(abaci_messages) if abaci_messages else random.choice(FALLBACK_PROMPTS)
                next_abaci = next_abaci.replace("ABACI:", "").replace("MAUK:", "").strip()
                contexts[i] = f"{ctx} {response}\nABACI: {next_abaci}\nMAUK:"

            all_queries.extend(query_tensors)
            all_responses.extend(response_tensors)

        # Important: do not manually pad these tensors. TRL needs to recompute
        # logprobs on the same query/response token sequences used for rollout.
        stats = ppo_trainer.step(all_queries, all_responses, all_rewards)

        avg_reward = sum(reward.item() for reward in all_rewards) / len(all_rewards)
        kl = stats.get("objective/kl", 0.0)
        if step % 10 == 0:
            print(f"Step {step}: reward={avg_reward:.4f}, kl={float(kl):.4f}")
            if wandb is not None:
                wandb.log(
                    {
                        "reward/mean": avg_reward,
                        "ppo/kl": kl,
                        "ppo/policy_loss": stats.get("ppo/loss/policy", 0),
                        "ppo/value_loss": stats.get("ppo/loss/value", 0),
                        "ppo/entropy": stats.get("objective/entropy", 0),
                        "step": step,
                    }
                )

        if step > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(model, tokenizer, output_dir, step)
            print_samples(
                model,
                tokenizer,
                device,
                SAMPLE_PROMPTS,
                max_new_tokens=args.max_new_tokens,
                sample_count=args.sample_count,
            )

    final_dir = save_checkpoint(model, tokenizer, output_dir, "final")
    print_samples(
        model,
        tokenizer,
        device,
        SAMPLE_PROMPTS,
        max_new_tokens=args.max_new_tokens,
        sample_count=args.sample_count,
    )

    try:
        merged_dir = output_dir / "merged_final"
        merged_model = model.pretrained_model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Saved merged LoRA model: {merged_dir}")
    except Exception as exc:
        print(f"Skipping merged save because merge_and_unload failed: {exc}")

    if wandb is not None:
        wandb.finish()

    print(f"Done. Final checkpoint: {final_dir}")


if __name__ == "__main__":
    train(parse_args())
