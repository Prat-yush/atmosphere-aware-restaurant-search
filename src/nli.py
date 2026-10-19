"""Step 3: NLI polarity of every kept sentence against every aspect-pole hypothesis.

p = P(entailment) - P(contradiction), in [-1, 1], from cross-encoder/nli-deberta-v3-small.
Output data/proc/polarity.parquet: sent_id plus one float32 column per pole, named
"<aspect>.<pole_id>" (e.g. "noise.quiet"), in sentences.parquet row order.

Work is checkpointed in chunks under data/proc/polarity_chunks/; an interrupted run
resumes from the last finished chunk. Runs on CUDA (Colab), MPS or CPU.

Usage:
    python -m src.nli          # compute (resumes if chunks exist)
    python -m src.nli --force  # discard checkpoints and output, recompute
"""

from __future__ import annotations

import argparse
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.prep import load_lexicon

Scorer = Callable[[Sequence[str]], np.ndarray]  # texts -> (len(texts), n_poles) polarity


# --- Hypotheses --------------------------------------------------------------

def pole_column(aspect: str, pole_id: str) -> str:
    return f"{aspect}.{pole_id}"


def load_hypotheses(lexicon: dict) -> list[tuple[str, str]]:
    """[(column, hypothesis)] in lexicon order: positive then negative pole per aspect."""
    return [
        (pole_column(aspect, poles[side]["id"]), poles[side]["hypothesis"])
        for aspect, poles in lexicon["aspects"].items()
        for side in ("positive", "negative")
    ]


# --- Model -------------------------------------------------------------------

def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def label_indices(id2label: dict) -> tuple[int, int]:
    """(entailment index, contradiction index) from the model config, not hard-coded."""
    by_name = {str(v).lower(): int(k) for k, v in id2label.items()}
    return by_name["entailment"], by_name["contradiction"]


class NLIScorer:
    """Scores premise sentences against a fixed list of hypotheses."""

    def __init__(
        self,
        hypotheses: Sequence[str],
        model_name: str = config.NLI_MODEL,
        device: str | None = None,
        batch_size: int = config.NLI_BATCH_SIZE,
        max_length: int = config.NLI_MAX_LENGTH,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.manual_seed(config.SEED)
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        self.entail, self.contra = label_indices(self.model.config.id2label)
        self.hypotheses = list(hypotheses)
        self.batch_size = batch_size
        self.max_length = max_length

    def score_pairs(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        """p for each (premise, hypothesis) pair, batched in length order to cut padding."""
        import torch

        order = np.argsort([len(p) for p in premises], kind="stable")
        out = np.empty(len(premises), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                enc = self.tokenizer(
                    [premises[i] for i in idx],
                    [hypotheses[i] for i in idx],
                    padding=True,
                    truncation="only_first",
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                probs = self.model(**enc).logits.float().softmax(dim=-1)
                out[idx] = (probs[:, self.entail] - probs[:, self.contra]).cpu().numpy()
        return out

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        n, h = len(texts), len(self.hypotheses)
        premises = [t for t in texts for _ in range(h)]
        hyps = self.hypotheses * n
        return self.score_pairs(premises, hyps).reshape(n, h)


# --- Checkpointed run --------------------------------------------------------

def chunk_path(chunk_dir: Path, i: int) -> Path:
    return chunk_dir / f"chunk_{i:05d}.parquet"


def score_chunk(sent_ids: Sequence[str], texts: Sequence[str], columns: list[str], scorer: Scorer) -> pd.DataFrame:
    scores = scorer(texts)
    if scores.shape != (len(texts), len(columns)):
        raise ValueError(f"scorer returned {scores.shape}, expected {(len(texts), len(columns))}")
    df = pd.DataFrame(np.clip(scores, -1.0, 1.0).astype(np.float32), columns=columns)
    df.insert(0, "sent_id", list(sent_ids))
    return df


def assemble(chunk_dir: Path, n_chunks: int, expected_ids: Sequence[str]) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(chunk_path(chunk_dir, i)) for i in range(n_chunks)], ignore_index=True)
    if df["sent_id"].tolist() != list(expected_ids):
        raise ValueError("checkpoint chunks do not match sentences.parquet; rerun with --force")
    return df


def run(
    sentences_path: Path,
    out_path: Path,
    chunk_dir: Path,
    scorer_factory: Callable[[list[str]], Scorer] | None = None,
    chunk_size: int = config.NLI_CHUNK_SIZE,
    force: bool = False,
    lexicon_path: Path = config.LEXICON_PATH,
) -> None:
    if force:
        shutil.rmtree(chunk_dir, ignore_errors=True)
        out_path.unlink(missing_ok=True)
    if out_path.exists():
        print(f"skip: {out_path} exists (use --force to recompute)")
        return

    hyps = load_hypotheses(load_lexicon(lexicon_path))
    columns = [c for c, _ in hyps]
    sentences = pd.read_parquet(sentences_path, columns=["sent_id", "text"])
    n = len(sentences)
    n_chunks = (n + chunk_size - 1) // chunk_size
    chunk_dir.mkdir(parents=True, exist_ok=True)

    todo = [i for i in range(n_chunks) if not chunk_path(chunk_dir, i).exists()]
    print(f"{n:,} sentences x {len(columns)} hypotheses = {n * len(columns):,} pairs; "
          f"{n_chunks - len(todo)}/{n_chunks} chunks already done")

    if todo:
        if scorer_factory is None:
            scorer_factory = lambda h: NLIScorer(h)  # noqa: E731
        scorer = scorer_factory([h for _, h in hyps])
        t0 = time.time()
        for k, i in enumerate(todo, 1):
            part = sentences.iloc[i * chunk_size : (i + 1) * chunk_size]
            df = score_chunk(part["sent_id"].tolist(), part["text"].tolist(), columns, scorer)
            tmp = chunk_path(chunk_dir, i).with_suffix(".tmp")
            df.to_parquet(tmp, index=False)
            tmp.rename(chunk_path(chunk_dir, i))  # atomic: a killed run never leaves a half chunk
            rate = k * chunk_size * len(columns) / (time.time() - t0)
            eta = (len(todo) - k) * chunk_size * len(columns) / rate
            print(f"chunk {i + 1}/{n_chunks}  {rate:,.0f} pairs/s  eta {eta / 60:.1f} min", flush=True)

    out = assemble(chunk_dir, n_chunks, sentences["sent_id"].tolist())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"wrote {out_path} ({len(out):,} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="discard checkpoints and output, recompute")
    args = parser.parse_args()
    run(config.SENTENCES_PATH, config.POLARITY_PATH, config.POLARITY_CHUNK_DIR, force=args.force)


if __name__ == "__main__":
    main()
