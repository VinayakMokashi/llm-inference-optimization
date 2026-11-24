# LLM Inference Optimization

Two ways to make an LLM application faster and cheaper, each as a runnable notebook saved with its outputs:

| Notebook | The idea | What it measures |
| --- | --- | --- |
| [`01_model_distillation/`](01_model_distillation/) | A large LLM labels data **once**; a tiny local network learns to copy it, then answers in milliseconds with no API | Agreement with the teacher on unseen queries, speed, cost, and whether soft labels beat hard labels |
| [`02_response_cache/`](02_response_cache/) | Keep the answers to questions already asked, and reply from the cache when the same question, or one that means the same, comes in again | Hit rate, time and tokens saved, how long answers should live, and what counts as "the same question" |

Both use `gpt-oss-120b` through an OpenAI-compatible API: [Cerebras](https://cloud.cerebras.ai) when its key is set, otherwise [Groq](https://console.groq.com)'s free tier.

## Repository structure

```
llm-inference-optimization/
├── 01_model_distillation/
│   ├── model_distillation.ipynb   # teacher labels, student network, training, evaluation, speed and cost (with saved outputs)
│   └── teacher_labels.json        # the teacher's labels for the 116 queries, so re-runs don't pay to label again
├── 02_response_cache/
│   └── response_cache.ipynb       # exact + semantic cache as a LangGraph agent, time to live, similarity tests, traffic simulation (with saved outputs)
├── requirements.txt
├── .env.example                   # template for your API key
└── LICENSE
```

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/VinayakMokashi/llm-inference-optimization.git
   cd llm-inference-optimization
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   This installs PyTorch (through `sentence-transformers`), so it can take a few minutes. The first run also downloads the `all-MiniLM-L6-v2` embedding model (about 90 MB). Everything runs on a CPU.

4. **Add an API key.** Copy `.env.example` to `.env` at the repository root and fill in one key:

   ```bash
   cp .env.example .env
   ```

   ```
   CEREBRAS_API_KEY=your_cerebras_api_key_here   # used first when it is set
   GROQ_API_KEY=your_groq_api_key_here           # used when there is no Cerebras key
   ```

   Both notebooks find `.env` at the repository root by themselves. It is git-ignored and never committed.

## How to run

```bash
jupyter notebook 01_model_distillation/model_distillation.ipynb
jupyter notebook 02_response_cache/response_cache.ipynb
```

Or open them in VS Code and choose the virtual environment as the kernel, or run one from start to finish without opening it:

```bash
jupyter nbconvert --to notebook --execute --inplace 02_response_cache/response_cache.ipynb
```

Both notebooks are saved with their outputs, so you can read every result on GitHub without running anything.

| Notebook | Model calls per run | Time |
| --- | --- | --- |
| Model distillation | 8: the labels are loaded from `teacher_labels.json`. Delete that file to label again (about 125 calls, ~45,000 tokens) | about 1 minute |
| Response cache | about 15 (roughly 6,000 tokens); the traffic simulation uses a fake model and makes none | about 1 minute |

## 1. Model distillation

A large model is good at classifying search queries by intent, but every call costs money and takes about half a second. Distillation uses the large model as a **teacher** that labels a set of queries once; a small **student** network then learns to copy those labels and runs locally from then on.

### How it works

```
116 search queries ─► teacher: gpt-oss-120b returns a probability for each intent (JSON) ─► teacher_labels.json
        │
        └─► all-MiniLM-L6-v2 (frozen, 384 numbers per query) ─► student: 384 → 128 → 5 ─► intent probabilities
                                                                trained to match the teacher (KL divergence, T = 3)
```

- **The task:** five intents a shop's search box might route on: navigational ("amazon login"), informational ("how does a microwave work"), transactional ("buy iphone 15 pro"), comparison ("nike vs adidas running shoes") and support ("track my order"). The queries are written without answers; the teacher provides every label.
- **The teacher:** asked for a probability for every intent, in a fixed JSON shape. Code checks each reply, rescales the five numbers to sum to 1, and asks again if a reply is unusable. The labels are saved, so the expensive step runs once.
- **Soft labels:** the student learns the teacher's full distribution ("40% informational, 40% transactional, 15% comparison"), not just its top pick.
- **The loss:** KL divergence between the student's and the teacher's distributions, both flattened with a temperature of 3 (and scaled by T²) so the small probabilities count too. Teacher probabilities below 1% are treated as 1% first: the teacher writes probabilities in steps of about 1%, and a tinier floor turned that rounding into large differences after flattening.
- **A fair test:** a quarter of the queries (29) are held out and never trained on. Section 10 also uses 4-fold cross-validation, repeated with 3 shuffles, because on 29 queries one query moves agreement by 3.4 points.

### Results

From the run saved in the [notebook](01_model_distillation/model_distillation.ipynb):

| | Teacher (API) | Student (local) |
| --- | --- | --- |
| Agrees with the teacher's top intent, unseen queries | — | **72%** (98% on its training queries; always guessing the most common intent: 24%) |
| Jensen-Shannon distance to the teacher, unseen queries | — | 0.33 |
| Time per query (median) | 471 ms | **12 ms**, of which 0.04 ms is the network and the rest the sentence encoder |
| Cost | about $0.16 per 1,000 queries at example rates | no API fee |
| Size | `gpt-oss-120b` (about 117B parameters) behind an API | 49,925 parameters, plus the 22.7M-parameter frozen encoder (about 90 MB) |

| Student trained on (cross-validated) | Agreement | Jensen-Shannon distance |
| --- | --- | --- |
| Soft labels, T = 3 | 70% | 0.347 |
| Soft labels, T = 1 | 72% | 0.341 |
| Hard labels (top intent only) | **77%** | 0.342 |

### What the results showed

- **Test on held-out data.** 98% agreement on the training queries against 72% on unseen ones: with 87 training queries the student partly memorises. Most misses are short queries such as "wikipedia" or "promo code not working".
- **Soft labels did not beat hard labels here.** The teacher's probabilities are written guesses, mostly round numbers with ties like 40/40, not the model's internal probabilities that classic distillation uses; and the data is small. This test can't say whether soft labels would win with thousands of real queries.
- **The student copies the teacher, mistakes included, and the teacher is not fully consistent:** "Samsung vs iPhone camera" got 70% comparison in one run and 80% in another, both at temperature 0.
- **"Tiny" describes the network, not the deployment.** Nearly all of the student's time and size is the sentence encoder; it still runs comfortably on an ordinary CPU.

## 2. Response cache

In most LLM applications a few questions make up most of the traffic. A response cache keeps each answer and serves it again when the same question, or one that means the same thing, comes back. The demo agent writes a short profile of a person, with two caches in front of the model, built as a [LangGraph](https://langchain-ai.github.io/langgraph/) graph:

```
query → check exact cache ──hit──→ reply
            │ miss
        check semantic cache ──hit──→ reply
            │ miss
        call the model → store the answer → reply
```

### How it works

- **Exact cache:** the key is the normalised text (lower case, no punctuation, no "'s"), so "Marie Curie!" and "marie curie" match. A dictionary lookup.
- **Semantic cache:** filler words ("tell me about", "who is") are stripped, the rest is embedded with `all-MiniLM-L6-v2`, and the closest cached question counts as a match above a cosine similarity of **0.60**.
- **Time to live:** each entry expires after `ttl_seconds` (24 hours by default; 5 seconds in the demo).
- **Guard (optional):** a semantic match is rejected if the new question has a meaningful word the cached one doesn't ("Pierre", "daughter", "mother").
- **`answer_fn`** is the model call; the traffic simulation swaps in a fake one, so it runs for free.

### Results

From the run saved in the [notebook](02_response_cache/response_cache.ipynb):

| Test | Result |
| --- | --- |
| 7 questions, with a repeat, a lowercase repeat and two rephrasings | **3 model calls**; 1.9 s instead of about 4.1 s; 694 tokens instead of about 1,600. An exact hit takes about 1.5 ms, a semantic hit 11-15 ms, a model call 500-700 ms |
| Time to live of 5 seconds | A repeat within 5 seconds came from the cache; after 6 seconds the model was called again |
| 5 similar-looking questions about other people, no guard | **3 got someone else's profile**: "Pierre Curie" and "Marie Curie's daughter" got Marie Curie's, "Elon Musk's mother" got Elon Musk's |
| Thresholds 0.5-0.9 on 8 same-person and 8 different-person pairs | At 0.6: 6 right matches and 6 wrong ones. At 0.8: 1 wrong, but only 3 right. With the guard: 0 wrong at every threshold, 6 right at 0.5-0.6 |
| 300 simulated requests (a few people far more popular than the rest), fake model | Guard on: **91% answered from the cache** (27 model calls), 0 wrong answers. Guard off: 93%, but 5 wrong: every request about Virat Kohli got Sachin Tendulkar's profile |

### What the results showed

- **Most savings come from the semantic cache:** 196 semantic hits against 77 exact ones, because people phrase the same question in many ways.
- **Deciding what counts as "similar" is the hard part.** Some wrong pairs score higher than right ones ("Venus Williams" vs "Serena Williams" 0.80; "Tell me about Elon" vs "Elon Musk" 0.62), so no threshold alone separates them. Test on labelled pairs and add a rule that fits the application.
- **Set the time to live by how fast the facts change,** and never share personal answers (orders, accounts) across users.
- **In production** the cache would live in Redis or a vector database, and its key would include the model and prompt version, so a prompt change doesn't serve old answers.

## Tech stack

- Python 3.10+ (tested on 3.12)
- `openai` SDK, pointed at Cerebras or Groq (both OpenAI-compatible), and `python-dotenv`
- `sentence-transformers` (`all-MiniLM-L6-v2`) and PyTorch for the embeddings and the student network
- NumPy and SciPy (Jensen-Shannon distance), pandas and matplotlib
- LangGraph for the cache agent's graph
- Jupyter to run the notebooks

## Limitations

- **Small, hand-written data.** 116 search queries and one teacher run for distillation; a dozen questions, 16 labelled pairs and a simulated stream of requests about 20 people for the cache. Both show the method and its pitfalls, not production numbers.
- **Written probabilities, not logits.** The teacher's soft labels are numbers the model writes in its reply. An open model run locally could expose its real output probabilities instead.
- **A simple cache.** An in-memory dictionary, compared against every entry on each lookup; the guard is a rule for name lookups and would need rethinking for other applications; descriptions such as "Nvidia CEO" never match a name.
- **Example prices.** Token rates in the distillation notebook are illustrative; check your provider's current prices.

## License

Released under the [MIT License](LICENSE).
