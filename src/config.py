"""Single source of configuration: paths, hyperparameters, model names."""

from pathlib import Path

SEED = 42

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "proc"
QUERIES_DIR = ROOT / "queries"

LEXICON_PATH = ROOT / "lexicon.yaml"
RAW_BUSINESS_PATH = RAW_DIR / "yelp_academic_dataset_business.json"
RAW_REVIEW_PATH = RAW_DIR / "yelp_academic_dataset_review.json"

BUSINESSES_PATH = PROC_DIR / "businesses.parquet"
SENTENCES_PATH = PROC_DIR / "sentences.parquet"
POLARITY_PATH = PROC_DIR / "polarity.parquet"
BM25_PATH = PROC_DIR / "bm25.pkl"
DENSE_INDEX_PATH = PROC_DIR / "dense.faiss"
INDEX_IDS_PATH = PROC_DIR / "index_ids.parquet"  # row i of both indexes -> sent_id
POLARITY_CHUNK_DIR = PROC_DIR / "polarity_chunks"  # nli.py checkpoints

# --- Business filter -----------------------------------------------------
TARGET_CITY = "philadelphia"  # compared after strip + casefold
TARGET_CATEGORY = "Restaurants"
MIN_REVIEW_COUNT = 20

# --- Sentence splitting --------------------------------------------------
SPACY_MODEL = "en_core_web_sm"
SPACY_BATCH_SIZE = 1000

# --- Retrieval / ranking -------------------------------------------------
K_RRF = 60  # reciprocal rank fusion constant
TOP_N_SENTENCES = 1000  # fused candidates kept per target
K_PER_RESTAURANT = 10  # evidence sentences kept per restaurant
M_SHRINK = 4  # pseudo-count for shrinkage toward global mean

# --- Models --------------------------------------------------------------
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
EMBED_BATCH_SIZE = 256
EMBED_DEVICE = None  # None = sentence-transformers auto-select (cuda > mps > cpu)
NLI_BATCH_SIZE = 64  # pairs per forward pass; fastest on M3 MPS (337 pairs/s vs 241 at 128)
NLI_MAX_LENGTH = 128  # tokens; covers >99% of kept sentences plus a hypothesis
NLI_CHUNK_SIZE = 4096  # sentences per checkpoint file
