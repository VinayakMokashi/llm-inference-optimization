"""
Quantization: fewer bits per weight, measured on a CPU.

A model's weights are usually stored as 32-bit floats. Quantization stores them with fewer bits
(16, 8, 4 ...), so the model takes less memory and, on hardware with fast low-precision arithmetic,
runs faster. The price is rounding error. Three experiments:

1. precision   How fast is one matrix multiplication in FP32, BF16 and FP16 on THIS machine?
               Lower precision only helps when the hardware has fast kernels for it.
2. bits        How much quality is left at 8, 4, 3 and 2 bits? DistilGPT2's transformer weights are
               rounded to each width ("fake" quantization: round, then store back as floats) with
               one scale per tensor, per output channel, or per group of 128 weights, and the
               model's perplexity on a fixed text is measured.
3. int8        Real int8 kernels: PyTorch's dynamic quantization applied to all-MiniLM-L6-v2, the
               sentence encoder used by the distillation and cache notebooks. Size on disk, speed,
               and whether its embeddings (and the response cache's decisions) change.

How to run
----------
    pip install -r requirements.txt
    python 04_quantization/quantization.py                  # all three experiments
    python 04_quantization/quantization.py --only bits

No API key is needed. The first run downloads DistilGPT2 (about 350 MB) and all-MiniLM-L6-v2 (90 MB).
"""

import argparse
import copy
import io
import json
import math
import os
import re
import statistics
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
warnings.filterwarnings("ignore", message="IProgress not found")

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
LABELS_FILE = HERE.parent / "01_model_distillation" / "teacher_labels.json"

# A short original text for measuring perplexity (how surprised the model is by each next token).
EVAL_TEXT = """Maps have always been arguments about what matters. A medieval map of the world might place
Jerusalem at its centre and fill the edges with monsters, because its makers cared more about meaning than
distance. Sailors in the Mediterranean drew portolan charts instead: webs of straight lines that told a pilot
which compass heading would carry the ship from one harbour to the next. Those charts were practical, and they
were remarkably accurate along the coasts that merchants used every week, but they said almost nothing about the
land behind the shore.
The great change came when map makers began to measure the earth rather than describe it. Surveyors carried
chains and brass instruments across fields and hills, measuring angles between church towers and hilltops and
building triangles that could be stitched together into a single network. A national survey could take decades.
In return it produced maps on which every road, river and village sat in its true position, and on which a
distance could be read with a ruler.
Printing made such maps cheap enough to sell to travellers, and railways gave travellers a reason to buy them.
Timetables, guidebooks and folding maps became ordinary objects. Cities published street plans with numbered
houses so that letters, taxis and fire engines could find an address. Each improvement made the map a little less
like a picture and a little more like a tool.
Today most people carry a map that redraws itself as they walk. It knows where they are, suggests where they might
want to go and estimates how long the journey will take. Yet the old questions have not gone away. Someone still
decides which shops appear on the screen, which roads are shown as fast, and which neighbourhoods are labelled at
all. A map is still an argument about what matters, even when it is written in code."""

# Labelled question pairs from the response-cache notebook: the same person, and a different person.
SAME_PERSON = [("Tell me about Elon", "Elon Musk"), ("Info on Curie", "Marie Curie"), ("Who is Elon Musk?", "Elon Musk"),
               ("Musk", "Elon Musk"), ("Tell me about Jensen", "Jensen Huang"), ("info on huang", "Jensen Huang"),
               ("Tesla founder", "Elon Musk"), ("Nvidia CEO", "Jensen Huang")]
DIFFERENT_PERSON = [("Pierre Curie", "Marie Curie"), ("Marie Curie's daughter", "Marie Curie"),
                    ("Elon Musk's mother", "Elon Musk"), ("Jensen Ackles", "Jensen Huang"),
                    ("Virat Kohli", "Sachin Tendulkar"), ("Venus Williams", "Serena Williams"),
                    ("Michelle Obama", "Barack Obama"), ("Steve Wozniak", "Steve Jobs")]
FILLER = {"tell", "me", "about", "info", "information", "on", "who", "is", "was", "what", "do", "you", "know",
          "the", "a", "an", "of", "please", "give", "some", "biography", "bio", "profile", "can"}
CACHE_THRESHOLD = 0.60


def core(text):
    """The cache notebook's preprocessing: lower case, no punctuation or "'s", no filler words."""
    text = re.sub(r"'s\b", "", text.lower().replace("’", "'"))
    words = [w for w in re.sub(r"[^a-z0-9 ]+", " ", text).split() if w not in FILLER]
    return " ".join(words) or text


def median_ms(fn, reps):
    fn()  # warm-up
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


# --- 1. Matrix multiplication by precision --------------------------------------------------------

def run_precision(size, reps):
    print(f"\n=== 1. One {size} x {size} matrix multiplication, by number format ===\n")
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    rows = []
    for device in devices:
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            a = torch.randn(size, size, device=device).to(dtype)
            b = torch.randn(size, size, device=device).to(dtype)

            def matmul():
                a @ b
                if device == "cuda":
                    torch.cuda.synchronize()

            slow = dtype != torch.float32 and device == "cpu"
            ms = median_ms(matmul, 2 if slow else reps)
            rows.append({"device": device, "dtype": str(dtype).replace("torch.", ""), "ms": ms})
    for device in devices:
        base = next(r["ms"] for r in rows if r["device"] == device and r["dtype"] == "float32")
        for r in (r for r in rows if r["device"] == device):
            r["vs_fp32"] = base / r["ms"]
            note = "faster" if r["vs_fp32"] > 1 else "slower"
            factor = r["vs_fp32"] if r["vs_fp32"] > 1 else 1 / r["vs_fp32"]
            print(f"{device:5s} {r['dtype']:9s} {r['ms']:10.1f} ms   {factor:6.1f}x {note if r['dtype'] != 'float32' else ''}")
    return rows


# --- 2. Quality at fewer bits (fake quantization of DistilGPT2) --------------------------------------

def quantize_dequantize(weight, bits, granularity, group_size=128):
    """Symmetric absmax quantization of a Conv1D weight (shape: inputs x outputs), returned as floats.

    per-tensor: one scale for the whole matrix; per-channel: one per output; group: one per output and
    per block of `group_size` inputs. Values are rounded to the integers -qmax..qmax.
    """
    qmax = 2 ** (bits - 1) - 1
    w = weight.detach()
    if granularity == "per-tensor":
        scale = w.abs().max().clamp(min=1e-12) / qmax
        return (w / scale).round().clamp(-qmax, qmax) * scale, 1
    if granularity == "per-channel":
        scale = w.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / qmax
        return (w / scale).round().clamp(-qmax, qmax) * scale, w.shape[1]
    inputs, outputs = w.shape
    pad = (-inputs) % group_size
    padded = torch.nn.functional.pad(w, (0, 0, 0, pad))
    groups = padded.reshape(-1, group_size, outputs)
    scale = groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / qmax
    q = (groups / scale).round().clamp(-qmax, qmax) * scale
    return q.reshape(-1, outputs)[:inputs], scale.numel()


def perplexity(model, ids):
    with torch.no_grad():
        return math.exp(model(ids, labels=ids).loss.item())


def run_bits(bit_widths):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.pytorch_utils import Conv1D

    name = "distilgpt2"
    print(f"\n=== 2. Quality at fewer bits: {name}'s transformer weights, perplexity on a fixed text ===\n")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).eval()
    ids = tokenizer(" ".join(EVAL_TEXT.split()), return_tensors="pt").input_ids
    layers = [m for m in model.modules() if isinstance(m, Conv1D)]
    originals = [layer.weight.data.clone() for layer in layers]
    n_weights = sum(w.numel() for w in originals)
    n_total = sum(p.numel() for p in model.parameters())
    base = perplexity(model, ids)
    print(f"{len(layers)} weight matrices, {n_weights / 1e6:.1f}M of the model's {n_total / 1e6:.1f}M parameters "
          f"(the token embeddings stay in FP32). Text: {ids.shape[1]} tokens.\n")
    print(f"{'bits':>4s} {'scales':12s} {'perplexity':>10s} {'change':>8s} {'size of these weights':>22s}")
    print(f"{32:>4d} {'(FP32)':12s} {base:10.2f} {'':>8s} {n_weights * 4 / 2**20:18.1f} MB")
    rows = [{"bits": 32, "granularity": "fp32", "perplexity": base, "mb": n_weights * 4 / 2**20}]
    for bits in bit_widths:
        for granularity in ("per-tensor", "per-channel", "group-128"):
            n_scales = 0
            for layer, original in zip(layers, originals):
                method = "group" if granularity == "group-128" else granularity
                layer.weight.data, count = quantize_dequantize(original, bits, method)
                n_scales += count
            ppl = perplexity(model, ids)
            mb = (n_weights * bits + n_scales * 16) / 8 / 2**20  # 16-bit scales
            rows.append({"bits": bits, "granularity": granularity, "perplexity": ppl, "mb": mb})
            print(f"{bits:>4d} {granularity:12s} {ppl:10.2f} {(ppl / base - 1) * 100:+7.1f}% {mb:18.1f} MB")
        print()
    for layer, original in zip(layers, originals):
        layer.weight.data = original
    return {"model": name, "tokens": ids.shape[1], "weights_quantized": n_weights, "rows": rows}


# --- 3. Real int8 kernels on the sentence encoder ------------------------------------------------------

def state_dict_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / 2**20


def run_int8(reps):
    from sentence_transformers import SentenceTransformer
    from torch.ao.quantization import per_channel_dynamic_qconfig

    print("\n=== 3. Real int8: PyTorch dynamic quantization of all-MiniLM-L6-v2 ===\n")
    fp32 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    models = {"fp32": fp32}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch.ao.quantization prints deprecation notices (torchao is its successor)
        models["int8, one scale per matrix"] = torch.ao.quantization.quantize_dynamic(
            copy.deepcopy(fp32), {torch.nn.Linear}, dtype=torch.qint8)
        models["int8, one scale per channel"] = torch.ao.quantization.quantize_dynamic(
            copy.deepcopy(fp32), {torch.nn.Linear: per_channel_dynamic_qconfig}, dtype=torch.qint8)
    queries = [row["query"] for row in json.loads(LABELS_FILE.read_text(encoding="utf-8"))]
    reference = fp32.encode(queries, normalize_embeddings=True)

    def pair_scores(model, pairs):
        x = model.encode([core(p) for p, _ in pairs], normalize_embeddings=True)
        y = model.encode([core(q) for _, q in pairs], normalize_embeddings=True)
        return (x * y).sum(axis=1)

    base_scores = {label: pair_scores(fp32, pairs) for label, pairs in
                   (("same person", SAME_PERSON), ("different person", DIFFERENT_PERSON))}
    rows = {}
    for name, model in models.items():
        row = {"size_mb": state_dict_mb(model),
               "batch_ms": median_ms(lambda m=model: m.encode(queries, batch_size=32), reps),
               "single_ms": median_ms(lambda m=model: m.encode(queries[0]), reps * 10)}
        if name != "fp32":
            cosines = (reference * model.encode(queries, normalize_embeddings=True)).sum(axis=1)
            row.update(cosine_mean=float(cosines.mean()), cosine_min=float(cosines.min()), flips=[], pair_change=0.0)
            for label, pairs in (("same person", SAME_PERSON), ("different person", DIFFERENT_PERSON)):
                scores = pair_scores(model, pairs)
                row["pair_change"] = max(row["pair_change"], float(abs(scores - base_scores[label]).max()))
                for (p, q), x, y in zip(pairs, base_scores[label], scores):
                    if (x >= CACHE_THRESHOLD) != (y >= CACHE_THRESHOLD):
                        row["flips"].append(f"{p!r} vs {q!r} ({label}): {x:.3f} -> {y:.3f}")
        rows[name] = row

    base = rows["fp32"]
    print(f"{'':28s} {'size':>8s} {'116 queries':>12s} {'1 query':>9s} {'similarity to fp32 (mean / lowest)':>36s}")
    for name, row in rows.items():
        similarity = "" if name == "fp32" else f"{row['cosine_mean']:.3f} / {row['cosine_min']:.3f}"
        print(f"{name:28s} {row['size_mb']:5.1f} MB {row['batch_ms']:9.0f} ms {row['single_ms']:6.1f} ms {similarity:>36s}")
    for name, row in rows.items():
        if name == "fp32":
            continue
        print(f"\n{name}: {base['size_mb'] / row['size_mb']:.1f}x smaller; speed {base['batch_ms'] / row['batch_ms']:.2f}x "
              f"on a batch, {base['single_ms'] / row['single_ms']:.2f}x on one query (above 1 = faster)")
        print(f"   cache pair scores change by up to {row['pair_change']:.3f}; decisions at the {CACHE_THRESHOLD} "
              f"threshold that flip: {len(row['flips'])} of {len(SAME_PERSON) + len(DIFFERENT_PERSON)}")
        for flip in row["flips"]:
            print("      " + flip)
    return rows


# --- Command line ------------------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Quantization experiments on a CPU (no API key needed).")
    p.add_argument("--only", choices=["precision", "bits", "int8"], help="run one experiment")
    p.add_argument("--size", type=int, default=1024, help="matrix size for the precision test (default 1024)")
    p.add_argument("--bits", type=int, nargs="+", default=[8, 4, 3, 2], help="bit widths to test (default 8 4 3 2)")
    p.add_argument("--reps", type=int, default=10, help="timed repetitions; the median is reported")
    p.add_argument("--save", help="write the measurements to this JSON file")
    return p.parse_args(argv)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    torch.manual_seed(0)
    print(f"PyTorch {torch.__version__}, {torch.get_num_threads()} CPU threads"
          + (f", GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ", no GPU"))
    results = {}
    if args.only in (None, "precision"):
        results["precision"] = run_precision(args.size, args.reps)
    if args.only in (None, "bits"):
        results["bits"] = run_bits(args.bits)
    if args.only in (None, "int8"):
        results["int8"] = run_int8(args.reps)
    if args.save:
        Path(args.save).write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"\nSaved the measurements to {args.save}")


if __name__ == "__main__":
    main()
