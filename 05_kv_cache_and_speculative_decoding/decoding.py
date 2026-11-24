"""
KV caching and speculative decoding, measured on GPT-2 models on a CPU.

An LLM writes one token at a time, and each new token attends to every token before it. Two
techniques attack that autoregressive loop:

1. KV caching. Without a cache, every step re-runs the model over the whole sequence, so a step
   gets slower as the text grows. With a cache, the keys and values of earlier tokens are kept, and
   each step only processes the newest token: memory is traded for compute.
2. Speculative decoding. A small draft model guesses the next k tokens; the large model checks all k
   in one forward pass and keeps the longest prefix it agrees with, plus one token of its own. With
   greedy decoding the output is exactly what the large model would have written alone; the question
   is only whether it arrives faster.

Both are implemented here from scratch, in plain loops, so every forward pass can be counted. The
models are GPT-2 Medium (355M parameters, the "large" model) and DistilGPT2 (82M, the draft); they
share a tokenizer, which speculative decoding needs. The first run downloads about 1.9 GB.

How to run
----------
    pip install -r requirements.txt
    python 05_kv_cache_and_speculative_decoding/decoding.py               # both experiments
    python 05_kv_cache_and_speculative_decoding/decoding.py --only kv-cache --new-tokens 100
    python 05_kv_cache_and_speculative_decoding/decoding.py --only speculative --draft-tokens 2 4 6

No API key is needed: everything runs locally.
"""

import argparse
import inspect
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

HERE = Path(__file__).resolve().parent
TARGET_MODEL = "gpt2-medium"
DRAFT_MODEL = "distilgpt2"

KV_PROMPT = ("The history of the printing press is a story about how ideas travel. Before movable type, "
             "every book was copied by hand, which made books rare and expensive. When printing spread "
             "across Europe in the fifteenth century,")

SPEC_PROMPTS = {
    "list": "The capital of France is Paris. The capital of Germany is Berlin. The capital of Spain is Madrid. "
            "The capital of Italy is",
    "story": "Once upon a time, in a small village by the sea, there lived an old fisherman who",
    "explainer": "A refrigerator keeps food cold by moving heat from the inside to the outside. It does this with",
}


# --- Small helpers ----------------------------------------------------------------------------

def load(name):
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()
    return model


def _cache_kwarg(model):
    """Newer transformers versions call the KV cache argument `cache`; older ones `past_key_values`."""
    return "cache" if "cache" in inspect.signature(model.forward).parameters else "past_key_values"


def forward(model, ids, cache=None):
    """One forward pass over `ids` (a 1-D list of token ids), appending to `cache` if one is given."""
    tensor = torch.tensor([ids])
    if cache is None:
        return model(tensor, use_cache=False).logits[0]
    return model(tensor, use_cache=True, **{_cache_kwarg(model): cache}).logits[0]


def trim(cache, keep):
    """Drop cached positions beyond `keep` (used when the large model rejects draft tokens)."""
    remove = cache.get_seq_length() - keep
    if remove > 0:
        cache.crop(-remove)


def kv_cache_bytes(cache):
    """Measured size of the keys and values held in a DynamicCache."""
    total = 0
    for layer in cache.layers:
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
    return total


# --- 1. KV caching ----------------------------------------------------------------------------

@torch.no_grad()
def greedy_no_cache(model, prompt_ids, new_tokens):
    """Every step runs the model over the whole sequence so far."""
    ids, step_ms = list(prompt_ids), []
    for _ in range(new_tokens):
        start = time.perf_counter()
        next_id = int(forward(model, ids)[-1].argmax())
        step_ms.append((time.perf_counter() - start) * 1000)
        ids.append(next_id)
    return ids[len(prompt_ids):], step_ms


@torch.no_grad()
def greedy_with_cache(model, prompt_ids, new_tokens):
    """Prefill the prompt once, then process only the newest token at each step."""
    cache, step_ms = DynamicCache(), []
    start = time.perf_counter()
    logits = forward(model, list(prompt_ids), cache)
    prefill_ms = (time.perf_counter() - start) * 1000
    ids = list(prompt_ids)
    for _ in range(new_tokens):
        start = time.perf_counter()
        next_id = int(logits[-1].argmax())
        ids.append(next_id)
        logits = forward(model, [next_id], cache)
        step_ms.append((time.perf_counter() - start) * 1000)
    return ids[len(prompt_ids):], step_ms, prefill_ms, cache


def run_kv_cache(tokenizer, model, new_tokens):
    config = model.config
    prompt_ids = tokenizer(KV_PROMPT).input_ids
    print(f"\n=== KV caching: {TARGET_MODEL}, a {len(prompt_ids)}-token prompt, {new_tokens} new tokens ===\n")
    greedy_with_cache(model, prompt_ids, 5)  # warm-up

    cached_out, cached_ms, prefill_ms, cache = greedy_with_cache(model, prompt_ids, new_tokens)
    plain_out, plain_ms = greedy_no_cache(model, prompt_ids, new_tokens)

    def window(values, first):
        part = values[:10] if first else values[-10:]
        return statistics.median(part)

    rows = [("without a cache", plain_ms, 0.0), ("with the KV cache", cached_ms, prefill_ms)]
    print("Time per new token (median), and in total:")
    print(f"{'':20s} {'first 10 tokens':>16s} {'last 10 tokens':>15s} {'total':>9s}")
    for label, ms, prefill in rows:
        print(f"{label:20s} {window(ms, True):12.1f} ms {window(ms, False):11.1f} ms "
              f"{(sum(ms) + prefill) / 1000:7.1f} s")
    speedup = (sum(plain_ms)) / (sum(cached_ms) + prefill_ms)
    print(f"\nSpeed-up from the cache: {speedup:.1f}x   identical text: {cached_out == plain_out}")

    per_token = 2 * config.n_layer * config.n_embd * 4  # keys + values, every layer, fp32
    measured = kv_cache_bytes(cache)
    length = cache.get_seq_length()
    print(f"\nKV cache size = 2 (keys, values) x {config.n_layer} layers x {config.n_embd} values x 4 bytes "
          f"= {per_token / 1024:.0f} KiB per token")
    print(f"Measured after {length} tokens: {measured / 2**20:.1f} MiB (formula: {per_token * length / 2**20:.1f} MiB). "
          f"At the model's full {config.n_positions}-token context: {per_token * config.n_positions / 2**20:.0f} MiB "
          "per sequence.")
    print("\nThe text it wrote:\n" + tokenizer.decode(cached_out).strip())
    return {"model": TARGET_MODEL, "prompt_tokens": len(prompt_ids), "new_tokens": new_tokens,
            "no_cache_ms_per_token": {"first10": window(plain_ms, True), "last10": window(plain_ms, False)},
            "cache_ms_per_token": {"first10": window(cached_ms, True), "last10": window(cached_ms, False)},
            "no_cache_total_s": sum(plain_ms) / 1000, "cache_total_s": (sum(cached_ms) + prefill_ms) / 1000,
            "speedup": speedup, "identical": cached_out == plain_out,
            "kv_bytes_per_token": per_token, "kv_bytes_measured": measured, "kv_tokens": length,
            "no_cache_step_ms": plain_ms, "cache_step_ms": cached_ms}


# --- 2. Speculative decoding --------------------------------------------------------------------

@torch.no_grad()
def speculative(target, draft, prompt_ids, new_tokens, k):
    """Greedy speculative decoding with KV caches on both models.

    Invariant at the top of each round: both caches hold every token except the last one.
    """
    t_cache, d_cache = DynamicCache(), DynamicCache()
    ids = list(prompt_ids)
    forward(target, ids[:-1], t_cache)
    forward(draft, ids[:-1], d_cache)
    stats = {"rounds": 0, "proposed": 0, "accepted": 0, "target_passes": 1, "draft_passes": 1}
    while len(ids) - len(prompt_ids) < new_tokens:
        stats["rounds"] += 1
        base = len(ids)
        # The draft proposes k tokens, one at a time.
        proposal, last = [], ids[-1]
        for _ in range(k):
            last = int(forward(draft, [last], d_cache)[-1].argmax())
            proposal.append(last)
            stats["draft_passes"] += 1
        # The large model scores the last accepted token and all k proposals in ONE pass.
        logits = forward(target, [ids[-1]] + proposal, t_cache)
        stats["target_passes"] += 1
        choices = logits.argmax(-1).tolist()  # what the large model would write after each position
        accepted = 0
        while accepted < k and choices[accepted] == proposal[accepted]:
            accepted += 1
        stats["proposed"] += k
        stats["accepted"] += accepted
        ids += proposal[:accepted] + [choices[accepted]]  # the agreed prefix + one token from the large model
        # Keep only the positions that are now part of the text (everything but the new last token).
        trim(t_cache, base + accepted)
        trim(d_cache, base + accepted)
        if accepted == k:  # the draft never saw its own last proposal; add it so the caches stay in step
            forward(draft, [proposal[-1]], d_cache)
            stats["draft_passes"] += 1
    return ids[len(prompt_ids):len(prompt_ids) + new_tokens], stats


def timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start


def run_speculative(tokenizer, target, draft, new_tokens, draft_tokens, reps):
    print(f"\n=== Speculative decoding: {DRAFT_MODEL} drafts, {TARGET_MODEL} verifies; {new_tokens} new tokens ===\n")
    greedy_with_cache(target, tokenizer("Warm up").input_ids, 3)
    greedy_with_cache(draft, tokenizer("Warm up").input_ids, 3)
    results = []
    for name, prompt in SPEC_PROMPTS.items():
        prompt_ids = tokenizer(prompt).input_ids
        base_times, base_out = [], None
        for _ in range(reps):
            (out, _, _, _), seconds = timed(greedy_with_cache, target, prompt_ids, new_tokens)
            base_times.append(seconds)
            base_out = out
        base = statistics.median(base_times)
        row = {"prompt": name, "baseline_s": base, "runs": []}
        print(f"[{name}] large model alone: {base:.2f} s ({new_tokens / base:.1f} tokens/s)")
        for k in draft_tokens:
            times, out, stats = [], None, None
            for _ in range(reps):
                (out, stats), seconds = timed(speculative, target, draft, prompt_ids, new_tokens, k)
                times.append(seconds)
            seconds = statistics.median(times)
            rate = stats["accepted"] / stats["proposed"]
            per_pass = new_tokens / stats["target_passes"]
            same = out == base_out
            row["runs"].append({"k": k, "seconds": seconds, "speedup": base / seconds, "acceptance": rate,
                                "tokens_per_target_pass": per_pass, "target_passes": stats["target_passes"],
                                "identical": same})
            print(f"   k={k}: {seconds:5.2f} s  speed-up {base / seconds:4.2f}x  draft tokens accepted {rate:4.0%}  "
                  f"{per_pass:4.2f} tokens per large-model pass  identical text: {same}")
        print("   text: " + tokenizer.decode(base_out).strip().replace("\n", " ")[:160] + "...\n")
        results.append(row)
    return results


# --- Command line -------------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="KV caching and speculative decoding on GPT-2 models (CPU).")
    p.add_argument("--only", choices=["kv-cache", "speculative"], help="run one experiment")
    p.add_argument("--new-tokens", type=int, default=200, help="tokens to generate in the KV-cache test (default 200)")
    p.add_argument("--spec-tokens", type=int, default=80, help="tokens per prompt in the speculative test (default 80)")
    p.add_argument("--draft-tokens", type=int, nargs="+", default=[2, 4, 6], help="values of k to try (default 2 4 6)")
    p.add_argument("--reps", type=int, default=3, help="timed repetitions per setting; the median is reported")
    p.add_argument("--threads", type=int, help="CPU threads for PyTorch (default: PyTorch's choice)")
    p.add_argument("--save", help="write the measurements to this JSON file")
    p.add_argument("--plot", help="save a chart of the results to this PNG file")
    return p.parse_args(argv)


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [key for key in ("kv_cache", "speculative") if key in results]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.2), squeeze=False)
    for ax, key in zip(axes[0], panels):
        if key == "kv_cache":
            kv = results["kv_cache"]
            ax.plot(range(1, len(kv["no_cache_step_ms"]) + 1), kv["no_cache_step_ms"], label="without a cache")
            ax.plot(range(1, len(kv["cache_step_ms"]) + 1), kv["cache_step_ms"], label="with the KV cache")
            ax.set_xlabel("new token number")
            ax.set_ylabel("ms for this token")
            ax.set_title(f"{kv['model']}: time per token as the text grows")
            ax.set_ylim(bottom=0)
            ax.legend()
        else:
            for row in results["speculative"]:
                ks = [r["k"] for r in row["runs"]]
                ax.plot(ks, [r["speedup"] for r in row["runs"]], marker="o", label=row["prompt"])
            ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
            ax.set_xticks(ks)
            ax.set_xlabel("draft tokens per round (k)")
            ax.set_ylabel("speed-up over the large model alone")
            ax.set_title("Speculative decoding (above 1 = faster)")
            ax.legend(title="prompt")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"Saved the chart to {path}")


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    print(f"PyTorch {torch.__version__} on CPU, {torch.get_num_threads()} threads")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    target = load(TARGET_MODEL)
    results = {"threads": torch.get_num_threads()}
    if args.only in (None, "kv-cache"):
        results["kv_cache"] = run_kv_cache(tokenizer, target, args.new_tokens)
    if args.only in (None, "speculative"):
        draft = load(DRAFT_MODEL)
        results["speculative"] = run_speculative(tokenizer, target, draft, args.spec_tokens, args.draft_tokens,
                                                 args.reps)
    if args.save:
        Path(args.save).write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"Saved the measurements to {args.save}")
    if args.plot:
        plot(results, args.plot)


if __name__ == "__main__":
    main()
