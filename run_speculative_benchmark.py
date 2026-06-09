"""
Speculative decoding benchmark: Qwen2.5-7B-Instruct (target, INT4-NF4)
                              + Qwen2.5-1.5B-Instruct (draft, FP16)
Reports: baseline tok/s vs speculative tok/s, acceptance rate, speedup.
"""
import os, time, torch
os.environ["HF_HOME"] = "/storage/gxg8313/hf"
os.environ["TRANSFORMERS_CACHE"] = "/storage/gxg8313/hf/hub"

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

TARGET_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DRAFT_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda:0"
MAX_NEW_TOKENS = 128
NUM_SPECULATE  = 4          # draft tokens per step
WARMUP_RUNS    = 2
BENCH_RUNS     = 8

PROMPTS = [
    "Explain the transformer attention mechanism in detail.",
    "What are the key differences between BERT and GPT architectures?",
    "Describe how gradient descent optimizes neural network weights.",
    "What is the role of layer normalization in deep learning models?",
    "Explain the concept of tokenization in natural language processing.",
    "How does the FAISS library enable fast similarity search at scale?",
    "What is speculative decoding and how does it speed up inference?",
    "Describe the differences between supervised and reinforcement learning.",
]


def load_target(model_name):
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=cfg, device_map=DEVICE, trust_remote_code=False
    )
    model.eval()
    return model


def load_draft(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=DEVICE, trust_remote_code=False
    )
    model.eval()
    return model


def benchmark_baseline(model, tokenizer, prompts, n_runs, max_new_tokens):
    total_tokens = 0
    total_time = 0.0
    for i in range(n_runs):
        prompt = prompts[i % len(prompts)]
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        n_new = out.shape[1] - inputs["input_ids"].shape[1]
        total_tokens += n_new
        total_time += elapsed
    return total_tokens / total_time  # tok/s


def benchmark_speculative(target, draft, tokenizer, draft_tokenizer, prompts, n_runs, max_new_tokens, num_speculate):
    total_tokens = 0
    total_time = 0.0
    for i in range(n_runs):
        prompt = prompts[i % len(prompts)]
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = target.generate(
                **inputs,
                assistant_model=draft,
                tokenizer=tokenizer,
                assistant_tokenizer=draft_tokenizer,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                num_assistant_tokens=num_speculate,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        n_new = out.shape[1] - inputs["input_ids"].shape[1]
        total_tokens += n_new
        total_time += elapsed
    return total_tokens / total_time  # tok/s


def gpu_mem_gb():
    return torch.cuda.memory_allocated(DEVICE) / 1e9


print("=" * 60)
print("Speculative Decoding Benchmark")
print(f"Target : {TARGET_MODEL} (INT4-NF4)")
print(f"Draft  : {DRAFT_MODEL}  (FP16)")
print(f"Device : {DEVICE}  |  max_new_tokens={MAX_NEW_TOKENS}  |  K={NUM_SPECULATE}")
print("=" * 60)

print("\n[1/4] Loading target model (INT4-NF4)...")
tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=False)
target = load_target(TARGET_MODEL)
mem_target = gpu_mem_gb()
print(f"      Target loaded  {mem_target:.2f} GB VRAM")

print("[2/4] Loading draft model (FP16)...")
draft = load_draft(DRAFT_MODEL)
draft_tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL, trust_remote_code=False)
mem_both = gpu_mem_gb()
print(f"      Both loaded    {mem_both:.2f} GB VRAM")

print(f"\n[3/4] Baseline (target only) — {WARMUP_RUNS} warm-up + {BENCH_RUNS} bench runs...")
benchmark_baseline(target, tokenizer, PROMPTS, WARMUP_RUNS, MAX_NEW_TOKENS)
baseline_tps = benchmark_baseline(target, tokenizer, PROMPTS, BENCH_RUNS, MAX_NEW_TOKENS)
print(f"      Baseline throughput: {baseline_tps:.1f} tok/s")

print(f"\n[4/4] Speculative decoding (K={NUM_SPECULATE}) — {WARMUP_RUNS} warm-up + {BENCH_RUNS} bench runs...")
benchmark_speculative(target, draft, tokenizer, draft_tokenizer, PROMPTS, WARMUP_RUNS, MAX_NEW_TOKENS, NUM_SPECULATE)
spec_tps = benchmark_speculative(target, draft, tokenizer, draft_tokenizer, PROMPTS, BENCH_RUNS, MAX_NEW_TOKENS, NUM_SPECULATE)
print(f"      Speculative throughput: {spec_tps:.1f} tok/s")

speedup = spec_tps / baseline_tps
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Baseline (INT4-NF4, target only) : {baseline_tps:.1f} tok/s")
print(f"Speculative (INT4-NF4 + 1.5B draft): {spec_tps:.1f} tok/s")
print(f"Speedup                            : {speedup:.2f}x")
print(f"VRAM (target only)                 : {mem_target:.2f} GB")
print(f"VRAM (target + draft)              : {mem_both:.2f} GB")
print("=" * 60)
print(f"FINAL: baseline={baseline_tps:.1f} spec={spec_tps:.1f} speedup={speedup:.2f}x")

import json, datetime
result = {
    "date": datetime.date.today().isoformat(),
    "hardware": "NVIDIA A30",
    "target_model": TARGET_MODEL,
    "target_quant": "INT4-NF4",
    "draft_model": DRAFT_MODEL,
    "draft_quant": "FP16",
    "max_new_tokens": MAX_NEW_TOKENS,
    "num_speculate_K": NUM_SPECULATE,
    "bench_runs": BENCH_RUNS,
    "baseline_toks_per_sec": round(baseline_tps, 1),
    "speculative_toks_per_sec": round(spec_tps, 1),
    "speedup": round(speedup, 2),
    "vram_target_only_gb": round(mem_target, 2),
    "vram_target_plus_draft_gb": round(mem_both, 2),
}
out_path = "speculative_benchmark_results.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"Results saved to {out_path}")
