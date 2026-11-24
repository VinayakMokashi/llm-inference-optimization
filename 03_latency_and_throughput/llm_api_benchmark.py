"""
Latency, throughput and cost of an LLM API, measured.

What an application built on a hosted model can measure, and control, about speed and cost:

1. anatomy       30 requests, one after another. For each: time to the first token (the model's hidden
                 reasoning included), time to the first token of the answer, and the total; plus the
                 provider's own timings (queue, reading the prompt, writing the answer) when it reports
                 them. Reported as mean and percentiles (p50, p95, p99): averages hide the slow ones.
2. input-length  Prompts of about 1,000 to 32,000 tokens, made unique so the provider's prompt cache can't
                 help. How do prefill time (reading the prompt) and time to first token grow with length?
3. concurrency   The same 16 requests with 1, 2, 4, 8 and 16 in flight at once: total time, requests and
                 tokens per second, and each request's latency.
4. prompt-cache  One long shared prefix with five different questions: does the provider reuse the prefix,
                 and how much faster is reading the prompt when it does?
5. cost          The per-request cost formula, 1.25 x (input tokens x input price + output tokens x output
                 price), with the measured token counts and example prices, scaled to a user base.

The model is gpt-oss-120b: on Cerebras when CEREBRAS_API_KEY is set, otherwise on Groq.

How to run
----------
    pip install -r requirements.txt
    python 03_latency_and_throughput/llm_api_benchmark.py                       # everything (~5 minutes)
    python 03_latency_and_throughput/llm_api_benchmark.py --only concurrency
    python 03_latency_and_throughput/llm_api_benchmark.py --provider groq --quick  # smaller, for a free tier

A full run sends about 230,000 tokens in 133 requests, most of them in the input-length test. --quick sends
far fewer, but on a free tier the rate limits dominate its timings.
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")

PRICE_PER_M = {"input": 0.35, "output": 0.75}  # USD per million tokens: EXAMPLE rates, check your provider
OVERHEAD = 1.25  # retries, system prompts and framework overhead, as a rule of thumb

TOPICS = ["how a refrigerator keeps food cold", "why the sky is blue", "how a credit card payment is approved",
          "what a load balancer does", "how vaccines train the immune system", "why bread rises",
          "how GPS finds your position", "what inflation is", "how noise-cancelling headphones work",
          "why leaves change colour in autumn", "how a search engine ranks pages", "what a mortgage is",
          "how solar panels make electricity", "why airplanes can fly", "how a password manager works",
          "what compound interest is", "how an electric car battery is charged", "why we have seasons",
          "how a barcode scanner works", "what a database index is", "how coffee is decaffeinated",
          "why the moon has phases", "how a heat pump heats a house", "what a VPN does",
          "how an elevator knows when to stop", "why ice floats", "how a microwave oven heats food",
          "what a carbon footprint is", "how fibre-optic cables carry data", "why cats purr"]

PRODUCTS = ["a stainless steel water bottle", "wireless earbuds with noise cancelling", "a bamboo cutting board",
            "a 27-inch 4K monitor", "trail running shoes", "a cast iron skillet", "a mechanical keyboard",
            "an electric toothbrush", "a yoga mat", "a robot vacuum cleaner", "a leather laptop sleeve",
            "a smart LED desk lamp", "a hard-shell carry-on suitcase", "a cold brew coffee maker",
            "a kids' bicycle helmet", "a portable power bank"]


# --- The API ------------------------------------------------------------------------------------------

def make_client(provider):
    if provider == "auto":
        provider = "cerebras" if os.environ.get("CEREBRAS_API_KEY") else "groq"
    if provider == "cerebras":
        return "Cerebras", OpenAI(base_url="https://api.cerebras.ai/v1", api_key=os.environ["CEREBRAS_API_KEY"],
                                  max_retries=10), "gpt-oss-120b"
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("No API key found: copy .env.example to .env at the repository root and add CEREBRAS_API_KEY "
                 "or GROQ_API_KEY.")
    return "Groq", OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"],
                          max_retries=10), "openai/gpt-oss-120b"


def call(client, model, messages, max_tokens):
    """One streamed request. Returns client-side timings, the provider's timings and token counts."""
    start = time.perf_counter()
    first_token = first_answer = None
    usage, server = None, {}
    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True, stream_options={"include_usage": True},
        temperature=0, max_tokens=max_tokens, reasoning_effort="low")
    for chunk in stream:
        now = time.perf_counter() - start
        extra = chunk.model_extra or {}
        if isinstance(extra.get("time_info"), dict):  # Cerebras reports its timings here
            server = extra["time_info"]
        if chunk.usage:
            usage = chunk.usage
            usage_extra = usage.model_extra or {}
            if "prompt_time" in usage_extra:  # Groq reports them inside usage
                server = usage_extra
        if chunk.choices:
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
            if (reasoning or delta.content) and first_token is None:
                first_token = now
            if delta.content and first_answer is None:
                first_answer = now
    total = time.perf_counter() - start
    details = getattr(usage, "prompt_tokens_details", None) if usage else None
    out_details = getattr(usage, "completion_tokens_details", None) if usage else None
    return {
        "ttft_s": first_token, "first_answer_s": first_answer, "total_s": total,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "reasoning_tokens": getattr(out_details, "reasoning_tokens", None) if out_details else None,
        "cached_tokens": getattr(details, "cached_tokens", None) if details else None,
        "server_queue_s": server.get("queue_time"), "server_prefill_s": server.get("prompt_time"),
        "server_decode_s": server.get("completion_time"), "server_total_s": server.get("total_time"),
    }


def pct(values, q):
    """Percentile q (0-100) with linear interpolation."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return float("nan")
    k = (len(values) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def summary(values):
    values = [v for v in values if v is not None]
    return {"mean": statistics.mean(values), "p50": pct(values, 50), "p95": pct(values, 95), "p99": pct(values, 99),
            "max": max(values)}


# --- 1. Anatomy and percentiles --------------------------------------------------------------------------

def run_anatomy(client, model, n):
    print(f"\n=== 1. Where the time goes: {n} short requests, one after another ===\n")
    rows = []
    for topic in TOPICS[:n]:
        rows.append(call(client, model, [{"role": "user", "content": f"In two sentences, explain {topic}."}], 600))
    stats = {key: summary([r[key] for r in rows]) for key in
             ("ttft_s", "first_answer_s", "total_s", "server_queue_s", "server_prefill_s", "server_decode_s",
              "server_total_s")
             if any(r[key] is not None for r in rows)}
    labels = {"ttft_s": "first token (incl. reasoning)", "first_answer_s": "first token of the answer",
              "total_s": "whole response (client)", "server_queue_s": "  provider: queue",
              "server_prefill_s": "  provider: reading the prompt", "server_decode_s": "  provider: writing the answer",
              "server_total_s": "  provider: total"}
    print(f"{'seconds':32s} {'mean':>7s} {'p50':>7s} {'p95':>7s} {'p99':>7s} {'max':>7s}")
    for key, s in stats.items():
        print(f"{labels[key]:32s} {s['mean']:7.3f} {s['p50']:7.3f} {s['p95']:7.3f} {s['p99']:7.3f} {s['max']:7.3f}")
    out_tokens = [r["completion_tokens"] for r in rows]
    decode = [r["completion_tokens"] / r["server_decode_s"] for r in rows if r["server_decode_s"]]
    reasoning = [r["reasoning_tokens"] for r in rows if r["reasoning_tokens"] is not None]
    print(f"\nOutput: {statistics.mean(out_tokens):.0f} tokens per response on average"
          + (f", {statistics.mean(reasoning):.0f} of them hidden reasoning" if reasoning else "")
          + (f"; the provider writes about {statistics.median(decode):,.0f} tokens/s" if decode else ""))
    if "server_total_s" in stats:
        network = [r["total_s"] - r["server_total_s"] for r in rows if r["server_total_s"]]
        print(f"Outside the provider's own timing (network, connection, streaming): median "
              f"{statistics.median(network):.3f} s of the {stats['total_s']['p50']:.3f} s median total")
    return {"rows": rows, "stats": stats}


# --- 2. Input length ---------------------------------------------------------------------------------------

def long_prompt(approx_tokens, nonce):
    """Numbered facts up to roughly `approx_tokens` tokens, starting with a unique id (no cache hits)."""
    rng = random.Random(nonce)
    lines = [f"Request {nonce}. Inventory records follow."]
    i, chars = 0, 0
    while chars < approx_tokens * 3.0:  # about 3 characters per token for this text
        i += 1
        line = (f"Record {i}: warehouse {rng.randint(1, 400)} holds {rng.randint(1, 999)} units of item "
                f"{rng.randint(1000, 9999)}, last counted on day {rng.randint(1, 365)}.")
        lines.append(line)
        chars += len(line) + 1
    target = rng.randint(1, i)
    question = f"\nHow many units does record {target} list? Reply with the number only."
    return "\n".join(lines) + question


def run_input_length(client, model, sizes, reps):
    print(f"\n=== 2. Input length: {', '.join(f'{s:,}' for s in sizes)} tokens, {reps} unique prompts each ===\n")
    rows = []
    for size in sizes:
        for _ in range(reps):
            result = call(client, model, [{"role": "user", "content": long_prompt(size, uuid.uuid4().hex)}], 300)
            result["target_tokens"] = size
            rows.append(result)
    print(f"{'input tokens':>12s} {'prefill (provider)':>19s} {'per 1k tokens':>14s} {'first token':>12s} "
          f"{'cost':>9s}")
    by_size = []
    for size in sizes:
        group = [r for r in rows if r["target_tokens"] == size]
        tokens = statistics.mean(r["prompt_tokens"] for r in group)
        prefill = statistics.median(r["server_prefill_s"] for r in group) if group[0]["server_prefill_s"] else None
        ttft = statistics.median(r["ttft_s"] for r in group)
        cost = OVERHEAD * (tokens * PRICE_PER_M["input"] + statistics.mean(r["completion_tokens"] for r in group)
                           * PRICE_PER_M["output"]) / 1e6
        by_size.append({"tokens": tokens, "prefill_s": prefill, "ttft_s": ttft, "cost": cost})
        prefill_txt = f"{prefill:15.3f} s" if prefill is not None else f"{'n/a':>17s}"
        per_k = f"{prefill / tokens * 1000 * 1000:10.2f} ms" if prefill is not None else f"{'':>13s}"
        print(f"{tokens:12,.0f} {prefill_txt:>19s} {per_k:>14s} {ttft:10.3f} s ${cost:8.5f}")
    first, last = by_size[0], by_size[-1]
    ratio = last["tokens"] / first["tokens"]
    print(f"\n{ratio:.0f}x the input tokens: first token {last['ttft_s'] / first['ttft_s']:.1f}x later"
          + (f", prefill {last['prefill_s'] / first['prefill_s']:.1f}x longer" if first["prefill_s"] else "")
          + f", cost {last['cost'] / first['cost']:.1f}x (billing is per token: linear)")
    return {"rows": rows, "by_size": by_size}


# --- 3. Concurrency ---------------------------------------------------------------------------------------

def copywriting(product):
    return [{"role": "system", "content": "You are an e-commerce copywriting assistant."},
            {"role": "user", "content": f"Write a concise SEO-friendly product title, three feature bullets and a "
                                        f"two-sentence description for {product}."}]


def run_concurrency(client, model, levels, n_requests):
    print(f"\n=== 3. Concurrency: the same {n_requests} requests, 1 to {max(levels)} in flight ===\n")
    products = (PRODUCTS * ((n_requests // len(PRODUCTS)) + 1))[:n_requests]
    results = []
    print(f"{'in flight':>9s} {'total time':>11s} {'requests/s':>11s} {'output tokens/s':>16s} "
          f"{'latency p50':>12s} {'latency p95':>12s}")
    for level in levels:
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=level) as pool:
            rows = list(pool.map(lambda p: call(client, model, copywriting(p), 800), products))
        wall = time.perf_counter() - start
        tokens = sum(r["completion_tokens"] for r in rows)
        latencies = [r["total_s"] for r in rows]
        results.append({"concurrency": level, "wall_s": wall, "req_per_s": n_requests / wall,
                        "tokens_per_s": tokens / wall, "p50_s": pct(latencies, 50), "p95_s": pct(latencies, 95),
                        "rows": rows})
        print(f"{level:9d} {wall:9.2f} s {n_requests / wall:11.2f} {tokens / wall:16,.0f} "
              f"{pct(latencies, 50):10.2f} s {pct(latencies, 95):10.2f} s")
    first, last = results[0], results[-1]
    print(f"\nFrom {first['concurrency']} to {last['concurrency']} in flight: "
          f"{last['req_per_s'] / first['req_per_s']:.1f}x the requests per second; "
          f"median latency {first['p50_s']:.2f} s -> {last['p50_s']:.2f} s")
    return results


# --- 4. Prompt cache ---------------------------------------------------------------------------------------

def run_prompt_cache(client, model, prefix_tokens, n_questions):
    print(f"\n=== 4. The provider's prompt cache: a {prefix_tokens:,}-token shared prefix, "
          f"{n_questions} different questions ===\n")
    nonce = uuid.uuid4().hex  # new for every run, so the first call is a guaranteed miss
    prefix = long_prompt(prefix_tokens, nonce).rsplit("\n", 1)[0]
    rows = []
    for i in range(n_questions):
        question = f"How many units does record {i * 7 + 3} list? Reply with the number only."
        rows.append(call(client, model, [{"role": "system", "content": prefix},
                                         {"role": "user", "content": question}], 300))
    print(f"{'call':>4s} {'input tokens':>13s} {'cached':>8s} {'prefill (provider)':>19s} {'first token':>12s}")
    for i, r in enumerate(rows, 1):
        cached = "n/a" if r["cached_tokens"] is None else f"{r['cached_tokens']:,}"
        prefill = "n/a" if r["server_prefill_s"] is None else f"{r['server_prefill_s']:.3f} s"
        print(f"{i:4d} {r['prompt_tokens']:13,d} {cached:>8s} {prefill:>19s} {r['ttft_s']:10.3f} s")
    hits = [r for r in rows[1:] if r["cached_tokens"]]
    if hits and rows[0]["server_prefill_s"]:
        print(f"\nLater calls reused {statistics.mean(r['cached_tokens'] / r['prompt_tokens'] for r in hits):.0%} of "
              f"the prompt; prefill {rows[0]['server_prefill_s']:.3f} s -> "
              f"{statistics.median(r['server_prefill_s'] for r in hits):.3f} s (median)")
    elif not hits:
        print("\nThis provider reported no cached tokens for this model.")
    return rows


# --- 5. Cost --------------------------------------------------------------------------------------------------

def run_cost(results):
    rows = [r for level in results.get("concurrency", []) for r in level["rows"]] or results["anatomy"]["rows"]
    tokens_in = statistics.mean(r["prompt_tokens"] for r in rows)
    tokens_out = statistics.mean(r["completion_tokens"] for r in rows)
    per_query = OVERHEAD * (tokens_in * PRICE_PER_M["input"] + tokens_out * PRICE_PER_M["output"]) / 1e6
    users, per_user = 10_000, 5
    daily = users * per_user * per_query
    print(f"\n=== 5. Cost, at EXAMPLE prices of ${PRICE_PER_M['input']} (input) and ${PRICE_PER_M['output']} "
          f"(output) per million tokens ===\n")
    print(f"Per request: {OVERHEAD} x ({tokens_in:.0f} input x ${PRICE_PER_M['input']}/M + {tokens_out:.0f} output x "
          f"${PRICE_PER_M['output']}/M) = ${per_query:.6f}")
    print(f"{users:,} users x {per_user} requests a day: ${daily:,.2f} a day, about ${daily * 30:,.0f} a month")
    return {"tokens_in": tokens_in, "tokens_out": tokens_out, "per_query": per_query, "daily": daily}


# --- Chart and command line ------------------------------------------------------------------------------------

def plot(results, path, provider):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    if "anatomy" in results:
        ax = axes[0][0]
        totals = [r["total_s"] for r in results["anatomy"]["rows"]]
        ax.hist(totals, bins=15, color="tab:blue", alpha=0.8)
        for label, value, style in (("mean", statistics.mean(totals), "-"), ("p50", pct(totals, 50), "--"),
                                    ("p95", pct(totals, 95), ":")):
            ax.axvline(value, color="black", linestyle=style, label=f"{label} {value:.2f} s")
        ax.set_title("Whole-response latency, 30 requests")
        ax.set_xlabel("seconds")
        ax.legend()
    if "input_length" in results:
        ax = axes[0][1]
        by_size = results["input_length"]["by_size"]
        xs = [b["tokens"] for b in by_size]
        ax.plot(xs, [b["ttft_s"] for b in by_size], marker="o", label="first token (client)")
        if by_size[0]["prefill_s"] is not None:
            ax.plot(xs, [b["prefill_s"] for b in by_size], marker="o", label="prefill (provider)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x / 1000:.0f}k" for x in xs])
        ax.minorticks_off()
        ax.set_title("Time vs input length")
        ax.set_xlabel("input tokens")
        ax.set_ylabel("seconds")
        ax.legend()
    if "concurrency" in results:
        ax = axes[1][0]
        levels = [r["concurrency"] for r in results["concurrency"]]
        ax.plot(levels, [r["req_per_s"] for r in results["concurrency"]], marker="o", label="requests/s")
        ax.set_xscale("log", base=2)
        ax.set_xticks(levels)
        ax.set_xticklabels([str(level) for level in levels])
        ax.minorticks_off()
        ax.set_xlabel("requests in flight")
        ax.set_ylabel("requests per second")
        ax2 = ax.twinx()
        ax2.plot(levels, [r["p95_s"] for r in results["concurrency"]], marker="s", color="tab:orange",
                 label="p95 latency")
        ax2.set_ylabel("p95 latency (s)")
        ax.set_title("Throughput and latency vs concurrency")
        lines = ax.get_legend_handles_labels()
        lines2 = ax2.get_legend_handles_labels()
        ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="upper left")
    if "prompt_cache" in results:
        ax = axes[1][1]
        rows = results["prompt_cache"]
        values = [r["server_prefill_s"] if r["server_prefill_s"] is not None else r["ttft_s"] for r in rows]
        ax.bar(range(1, len(rows) + 1), values, color=["tab:red"] + ["tab:green"] * (len(rows) - 1))
        ax.set_title("Prompt cache: first call (miss) vs later calls")
        ax.set_xlabel("call")
        ax.set_ylabel("prefill seconds (provider)" if rows[0]["server_prefill_s"] is not None else "first token (s)")
    fig.suptitle(f"gpt-oss-120b on {provider}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"Saved the chart to {path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Measure latency, throughput and cost of an LLM API.")
    p.add_argument("--provider", choices=["auto", "cerebras", "groq"], default="auto",
                   help="auto = Cerebras if its key is set, otherwise Groq")
    p.add_argument("--only", choices=["anatomy", "input-length", "concurrency", "prompt-cache"],
                   help="run one experiment (the cost estimate uses whichever ran)")
    p.add_argument("--quick", action="store_true", help="smaller runs, for a free tier's rate limits")
    p.add_argument("--save", help="write all measurements to this JSON file")
    p.add_argument("--plot", help="save a chart to this PNG file")
    return p.parse_args(argv)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    provider, client, model = make_client(args.provider)
    quick = args.quick
    print(f"Model: {model} on {provider}" + (" (quick mode)" if quick else ""))
    results = {"provider": provider, "model": model}
    if args.only in (None, "anatomy"):
        results["anatomy"] = run_anatomy(client, model, 10 if quick else 30)
    if args.only in (None, "input-length"):
        sizes = [1000, 2000, 4000] if quick else [1000, 2000, 4000, 8000, 16000, 32000]
        results["input_length"] = run_input_length(client, model, sizes, 2 if quick else 3)
    if args.only in (None, "concurrency"):
        levels = [1, 2, 4] if quick else [1, 2, 4, 8, 16]
        results["concurrency"] = run_concurrency(client, model, levels, 8 if quick else 16)
    if args.only in (None, "prompt-cache"):
        results["prompt_cache"] = run_prompt_cache(client, model, 3000 if quick else 6000, 5)
    if "anatomy" in results or "concurrency" in results:
        results["cost"] = run_cost(results)
    if args.save:
        Path(args.save).write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"\nSaved the measurements to {args.save}")
    if args.plot:
        plot(results, args.plot, provider)


if __name__ == "__main__":
    main()
