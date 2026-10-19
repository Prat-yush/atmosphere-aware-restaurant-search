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
| 2. BM25 + dense index | `python -m src.index` | *(next)* |
| 3. NLI polarity | `python -m src.nli` | `data/proc/polarity.parquet` *(week 2)* |

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

## Layout

```
lexicon.yaml      frozen aspect lexicon (6 aspects x 2 poles, NLI hypotheses)
src/config.py     paths, hyperparameters, model names
src/prep.py       step 1
tests/            pytest suite
queries/          hand-written queries and qrels (not generated)
```
