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
