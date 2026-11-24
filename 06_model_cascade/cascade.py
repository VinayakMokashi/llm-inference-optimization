"""
A model cascade: answer with the small local model when it is confident, call the LLM when it is not.

The distillation notebook trains a 50,000-parameter student that copies an LLM's intent labels and
runs about 40 times faster, but agrees with the LLM on only about 72% of unseen queries. A cascade
uses both: the student answers every query first, and only queries where its top probability is
below a threshold are sent to the LLM. The threshold sets the trade-off between cost and quality.

This script needs no API calls: the LLM's labels for all 116 queries are already saved in
01_model_distillation/teacher_labels.json, so "sending a query to the LLM" means using its saved label
(along with that call's recorded latency and tokens). The student's predictions come from
cross-validation, so every query is scored by a student that never trained on it.

How to run
----------
    pip install -r requirements.txt
    python 06_model_cascade/cascade.py
    python 06_model_cascade/cascade.py --thresholds 0.5 0.6 0.7 --plot cascade.png
"""

import argparse
import json
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
warnings.filterwarnings("ignore", message="IProgress not found")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = Path(__file__).resolve().parent
LABELS_FILE = HERE.parent / "01_model_distillation" / "teacher_labels.json"
INTENTS = ["navigational", "informational", "transactional", "comparison", "support"]
PRICE_PER_M = {"input": 0.35, "output": 0.75}  # USD per million tokens: EXAMPLE rates, check your provider

# The distillation notebook's settings.
SEED, T, FLOOR, EPOCHS, BATCH, LR = 42, 3.0, 0.01, 60, 16, 1e-3


def make_student():
    torch.manual_seed(SEED)
    return nn.Sequential(nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, len(INTENTS)))


def distillation_loss(logits, teacher_probs):
    """KL divergence at temperature T, with the teacher's probabilities floored at 1% before softening."""
    soft = torch.softmax(torch.log(teacher_probs.clamp_min(FLOOR)) / T, dim=-1)
    return F.kl_div(F.log_softmax(logits / T, dim=-1), soft, reduction="batchmean") * T * T


def train(X, P, idx):
    model = make_student()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    order = torch.Generator().manual_seed(SEED)
    idx = torch.tensor(idx)
    for _ in range(EPOCHS):
        model.train()
        for batch in idx[torch.randperm(len(idx), generator=order)].split(BATCH):
            loss = distillation_loss(model(X[batch]), P[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def out_of_fold_probs(X, P, folds=4, shuffles=3):
    """One array per shuffle: each query's probabilities from the student that did not train on it."""
    runs = []
    for shuffle in range(shuffles):
        probs = np.zeros((len(X), len(INTENTS)))
        parts = np.array_split(np.random.default_rng(shuffle).permutation(len(X)), folds)
        for k in range(folds):
            trained_on = np.concatenate([p for j, p in enumerate(parts) if j != k])
            model = train(X, P, trained_on)
            with torch.no_grad():
                probs[parts[k]] = torch.softmax(model(X[parts[k]]), dim=-1).numpy()
        runs.append(probs)
    return runs


def sweep(prob_runs, teacher_top, teacher_ms, teacher_cost, student_ms, thresholds, rng_seed=0):
    """Escalate the queries whose top student probability is below each threshold.

    Every number is computed for each cross-validation shuffle separately (one student per query, as
    in deployment) and then averaged over the shuffles.
    """
    rows = []
    for threshold in thresholds:
        per_run = []
        for probs in prob_runs:
            confidence, student_right = probs.max(axis=1), probs.argmax(axis=1) == teacher_top
            escalate = confidence < threshold
            kept = ~escalate
            rng = np.random.default_rng(rng_seed)  # the same number of escalations, chosen at random
            per_run.append({
                "sent_to_llm": escalate.mean(),
                "agreement": np.where(escalate, True, student_right).mean(),
                "agreement_if_random": np.mean([np.where(rng.permutation(escalate), True, student_right).mean()
                                                for _ in range(200)]),
                "student_right_when_kept": student_right[kept].mean() if kept.any() else None,
            })
        mean = {key: float(np.mean([run[key] for run in per_run])) for key in per_run[0]
                if key != "student_right_when_kept"}
        kept_rates = [run["student_right_when_kept"] for run in per_run if run["student_right_when_kept"] is not None]
        mean["student_right_when_kept"] = float(np.mean(kept_rates)) if kept_rates else None
        rows.append({"threshold": threshold, **mean,
                     "cost_per_1k": mean["sent_to_llm"] * 1000 * teacher_cost,
                     "mean_latency_ms": student_ms + mean["sent_to_llm"] * teacher_ms})
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="A student-then-LLM cascade, simulated with saved LLM labels.")
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01],
                   help="escalate when the student's top probability is below this (0 = never, above 1 = always)")
    p.add_argument("--save", help="write the results to this JSON file")
    p.add_argument("--plot", help="save a chart to this PNG file")
    return p.parse_args(argv)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    from sentence_transformers import SentenceTransformer

    rows = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    queries = [r["query"] for r in rows]
    P = torch.tensor([[r[i] for i in INTENTS] for r in rows], dtype=torch.float32)
    teacher_top = P.numpy().argmax(axis=1)
    teacher_ms = statistics.mean(r["latency_s"] for r in rows) * 1000
    teacher_cost = statistics.mean(r["prompt_tokens"] * PRICE_PER_M["input"] + r["completion_tokens"]
                                   * PRICE_PER_M["output"] for r in rows) / 1e6

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    X = torch.tensor(encoder.encode(queries), dtype=torch.float32)
    start = time.perf_counter()
    prob_runs = out_of_fold_probs(X, P)
    print(f"{len(queries)} queries; cross-validated student predictions from 12 trainings "
          f"({time.perf_counter() - start:.0f} s), averaged over 3 shuffles")

    student = train(X, P, np.arange(len(queries)))  # the deployed student, for timing only
    encoder.encode("warm-up")
    times = []
    for q in queries[:30]:
        t0 = time.perf_counter()
        with torch.no_grad():
            student(torch.tensor(encoder.encode([q]), dtype=torch.float32))
        times.append((time.perf_counter() - t0) * 1000)
    student_ms = statistics.median(times)
    print(f"Student: {student_ms:.1f} ms per query on this machine. LLM: {teacher_ms:.0f} ms on average per call "
          f"(recorded), ${teacher_cost * 1000:.3f} per 1,000 calls at example rates.\n")

    results = sweep(prob_runs, teacher_top, teacher_ms, teacher_cost, student_ms, args.thresholds)
    print(f"{'escalate below':>14s} {'sent to LLM':>12s} {'agreement':>10s} {'(random)':>9s} "
          f"{'student right when kept':>24s} {'$ per 1k':>9s} {'mean latency':>13s}")
    for r in results:
        label = "never" if r["threshold"] <= 0 else ("always" if r["threshold"] > 1 else f"{r['threshold']:.2f}")
        kept = "-" if r["student_right_when_kept"] is None else f"{r['student_right_when_kept']:.0%}"
        print(f"{label:>14s} {r['sent_to_llm']:12.0%} {r['agreement']:10.0%} {r['agreement_if_random']:9.0%} "
              f"{kept:>24s} {r['cost_per_1k']:9.3f} {r['mean_latency_ms']:10.0f} ms")
    print("\nagreement = the cascade's answer matches the LLM's; (random) = escalating the same number of queries at random.")

    if args.save:
        Path(args.save).write_text(json.dumps({"student_ms": student_ms, "teacher_ms": teacher_ms,
                                               "teacher_cost_per_query": teacher_cost, "rows": results},
                                              indent=1), encoding="utf-8")
        print(f"Saved the results to {args.save}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        xs = [r["sent_to_llm"] * 100 for r in results]
        ax.plot(xs, [r["agreement"] * 100 for r in results], marker="o", label="escalate when the student is unsure")
        ax.plot(xs, [r["agreement_if_random"] * 100 for r in results], linestyle="--", label="escalate at random")
        for r, x in zip(results, xs):
            if 0 < r["threshold"] <= 1:
                ax.annotate(f"{r['threshold']:.1f}", (x, r["agreement"] * 100), textcoords="offset points",
                            xytext=(4, -12), fontsize=8)
        ax.set_xlabel("% of queries sent to the LLM (cost)")
        ax.set_ylabel("% matching the LLM's answer")
        ax.set_title("Student-then-LLM cascade")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print(f"Saved the chart to {args.plot}")


if __name__ == "__main__":
    main()
