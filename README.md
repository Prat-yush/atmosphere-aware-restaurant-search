# AmbiSense

Evidence-backed atmosphere search over Yelp restaurant reviews (Philadelphia).
CSCE 670 course project, Texas A&M, Fall 2026.

Type a query such as *"quiet but not boring, good for a first date"* and get restaurants
ranked by review-sentence evidence, with supporting and contradicting sentences shown
for each result.

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # includes the spaCy en_core_web_sm wheel
pytest
```

## Data

This project uses the [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/).
By downloading it you accept the Yelp Dataset Terms of Use (non-commercial and academic use,
no redistribution). `data/` is gitignored. Never commit raw or derived data.

```bash
mkdir -p data/download data/raw
curl -L -C - -o data/download/Yelp-JSON.zip \
  https://business.yelp.com/external-assets/files/Yelp-JSON.zip      # ~4.3 GB
unzip -j data/download/Yelp-JSON.zip '*.tar' -d data/download          # yelp_dataset.tar
tar -xf data/download/yelp_dataset.tar -C data/raw \
  yelp_academic_dataset_business.json yelp_academic_dataset_review.json
rm -r data/download   # optional, frees ~8.6 GB
```

The result should be:

```
data/raw/yelp_academic_dataset_business.json
data/raw/yelp_academic_dataset_review.json   # ~5 GB, always streamed
```

## Pipeline

Each step is idempotent. It skips outputs that already exist unless you pass `--force`.

| step | command | output |
|---|---|---|
| 1. filter + sentence extraction | `python -m src.prep` | `data/proc/businesses.parquet`, `data/proc/sentences.parquet` |
| 2. indexes | `python -m src.index` | `data/proc/bm25.pkl`, `data/proc/dense.faiss`, `data/proc/index_ids.parquet`, `data/proc/bm25_restaurants.pkl` |
| 3. NLI polarity | `python -m src.nli` | `data/proc/polarity.parquet` |

Then query from the command line:

```bash
python -m src.parse "quiet but not boring, good for a first date"   # show parsed targets
python -m src.rank  "quiet but not boring, good for a first date" --system full --top 10
```

For a quick sample run, use `python -m src.prep --max-reviews 50000`. It writes
`data/proc/sentences.sample50000.parquet` and leaves the full output path alone.

### Step 1 details

- Restaurants: `city` equals `philadelphia` after whitespace and case normalization,
  `Restaurants` is one of the categories, and `review_count >= 20`.
- Reviews are streamed line by line. Each review is split at line breaks, then into
  sentences with spaCy `en_core_web_sm` (tokenizer plus the rule-based `sentencizer`; no parser).
- A sentence is kept if it contains at least one term from `lexicon.yaml`. Matching is
  case-insensitive and whole-word. `exclude` phrases (such as "dim sum") are removed
  before matching.
- `sent_id` is `<review_id>_<sentence index within review>`, which stays stable across runs.

### Step 2 details

- BM25 (`rank_bm25.BM25Okapi`): lowercase word tokens, with contractions kept whole
  ("wasn't"). No stemming and no stopword removal.
- Dense: `all-MiniLM-L6-v2` embeddings, L2-normalized, in a FAISS `IndexFlatIP`. Inner
  product equals cosine similarity, and search is exact.
- Row *i* of both indexes is row *i* of `index_ids.parquet` (`sent_id`, `business_id`).
  `load_indexes()` refuses to load misaligned files.
- Restaurant-level BM25 for the `bm25_baseline` system: each restaurant's reviews are
  concatenated into one document (~74M tokens in all). The counts are kept in a sparse
  matrix and scored with rank_bm25's BM25Okapi formula; a test checks it gives the same
  scores as rank_bm25.

### Step 3 details

- `cross-encoder/nli-deberta-v3-small`. The premise is the review sentence and the
  hypothesis is each pole's hypothesis from `lexicon.yaml`. Stored value:
  `p = P(entailment) - P(contradiction)`, in [-1, 1], one column per pole (`noise.quiet`, ...).
- Hypotheses are short, in the past tense, with an aspect subject ("The lighting was dim.").
  The first wording tried ("This restaurant has dim lighting.") scored clearly supporting
  sentences as neutral, because reviews rarely say "restaurant".
- Checkpointed in 4,096-sentence chunks under `data/proc/polarity_chunks/`; rerunning
  resumes. About 1.5 h on an M3 (MPS); it also runs on CUDA (fp16) or CPU.

### Step 4: ranking (`src/rank.py`)

Per parsed target (aspect, pole):

1. Retrieval text = the target pole's hypothesis plus the matched query term. BM25 and
   dense search run separately, **restricted to sentences whose lexicon match includes the
   target's aspect**, and are fused with reciprocal rank fusion (k_rrf = 60). The top 1,000
   fused sentences are kept. The aspect restriction is there because the NLI model tends to
   call off-topic sentences "contradiction".
2. Group by restaurant and keep the top k = 10 sentences: the evidence set E_r.
3. Aggregate over **retrieved evidence only**. The denominator is E_r, never all reviews:
   `w_i = sim_i / sum_{j in E_r} sim_j` (sim = fused RRF score), `raw_r = sum_i w_i p_i`,
   `score_r = (n_r raw_r + m mu) / (n_r + m)` with m = 4, and mu = the mean raw_r over
   restaurants with evidence for that target.
4. Final score = mean over targets. A restaurant with no evidence for a target gets mu
   for it. Only restaurants with evidence for at least one target are ranked.

Systems: `full`, `no_polarity` (every p_i = +1, same mu), `bm25_baseline`.

### Step 5: evaluation (`src/eval.py`)

Queries are written by hand in `queries/test.jsonl` (30) and `queries/dev.jsonl` (6), one JSON
object per line:

```json
{"qid": "t01", "text": "quiet but not boring, good for a first date", "category": "mixed"}
```

`category` is one of `single`, `mixed`, `negation`, `contrast`.

```bash
python -m src.eval pool               # data/eval/pool.csv: blind pool of top-10 from all systems
#   grade it: grade = 0 (not relevant), 1 (partly), 2 (fully)
python -m src.eval import-grades      # -> queries/qrels.tsv (ids and grades only; committed)
python -m src.eval metrics            # nDCG@10, P@5 by system and category; popularity check
python -m src.eval polarity-sample    # data/eval/polarity_sample.csv: 100 sentences
#   label it: label = pos, neg or none (for the row's aspect)
python -m src.eval polarity-accuracy  # NLI accuracy against your labels
python -m src.eval tune               # k, m, mu grid on dev queries only
```

`data/eval/` holds Yelp text, so it is gitignored. Back up your graded CSVs yourself.

### macOS note

torch and faiss-cpu each bundle their own OpenMP runtime, and loading both in one process
aborts. On macOS, `src/__init__.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1`
before anything else is imported. The test suite checks that faiss results still match
numpy exactly after torch has run. Linux and Colab are unaffected.

## Layout

```
lexicon.yaml      frozen aspect lexicon (6 aspects x 2 poles, NLI hypotheses)
src/config.py     paths, hyperparameters, model names
src/prep.py       step 1
src/index.py      step 2
src/nli.py        step 3
src/parse.py      query parser
src/rank.py       step 4
src/eval.py       step 5
tests/            pytest suite
queries/          hand-written queries and qrels (not generated)
```
