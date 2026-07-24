#!/usr/bin/env python
"""BAREC 2026 sentence-level Strict Track regression baseline.

The complete workflow lives in this file: validation, Arabic D3Tok
preprocessing, distributed fine-tuning, model selection, inference, and
submission validation.  Edit :class:`Config` and run ``python train.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import inspect
import json
import logging
import math
import multiprocessing
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import uuid
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# ---------------------------------------------------------------------------
# 1. Centralized configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger("barec")
PREPROCESSING_VERSION = "barec-d3tok-v1"
TATWEEL = "\u0640"
ARABIC_DIACRITICS = frozenset(chr(code) for code in range(0x064B, 0x0653)) | {
    "\u0670"
}


@dataclass
class Config:
    """All user-editable paths and hyperparameters for the baseline."""

    # Data and output paths, resolved relative to this file.
    TRAIN_PATH: str = "data/barec-corpus-v1/train.csv"
    DEV_PATH: str = "data/barec-corpus-v1/dev.csv"
    TEST_PATH: str = "data/barec-corpus-v1/test.csv"
    OUTPUT_DIR: str = "outputs"
    CHECKPOINT_DIR: str = "outputs/checkpoints"
    CACHE_DIR: str = "cache"
    SUBMISSION_DIR: str = "outputs"

    # Columns.
    ID_COLUMN: str = "ID"
    TEXT_COLUMN: str = "Sentence"
    LABEL_COLUMN: str = "Readability_Level_19"

    # Model and Arabic preprocessing.
    MODEL_NAME: str = "CAMeL-Lab/readability-arabertv2-d3tok-reg"
    # The official regression checkpoint was trained on zero-based levels
    # 0..18, while this shared-task pipeline keeps labels on the 1..19 scale.
    MODEL_OUTPUT_OFFSET: float = 1.0
    MAX_LENGTH: int = 256
    DROPOUT: float = 0.1
    D3TOK_RESOURCE: str = "calima-msa-r13"
    AUTO_DOWNLOAD_CAMEL_DATA: bool = True
    CAMEL_DATA_PACKAGE: str = "light"
    FORCE_REPROCESS: bool = False
    PREPROCESS_NUM_WORKERS: int = 1

    # Training.
    NUM_EPOCHS: int = 5
    PER_DEVICE_BATCH_SIZE: int = 8
    EVAL_BATCH_SIZE: int = 16
    GRADIENT_ACCUMULATION_STEPS: int = 2
    ENCODER_LR: float = 2e-5
    HEAD_LR: float = 1e-4
    WEIGHT_DECAY: float = 0.01
    WARMUP_RATIO: float = 0.1
    MAX_GRAD_NORM: float = 1.0
    EARLY_STOPPING_PATIENCE: int = 2
    SEED: int = 42
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True
    USE_FP16: bool = True
    USE_WEIGHTED_SAMPLER: bool = False
    SAMPLER_ALPHA: float = 0.5
    SAMPLER_REPLACEMENT: bool = True
    DDP_TIMEOUT_MINUTES: int = 180
    LOG_EVERY_N_STEPS: int = 50
    RESUME_FROM_CHECKPOINT: Optional[str] = None

    # Labels and submission.
    MIN_LABEL: int = 1
    MAX_LABEL: int = 19
    SUBMISSION_BASENAME: str = "prediction"
    SUBMISSION_ZIP_NAME: str = "prediction.zip"

    # Smoke-test limits. These do not affect a normal run.
    SMOKE_TRAIN_SAMPLES: int = 32
    SMOKE_EVAL_SAMPLES: int = 16
    # Four micro-batches cover two accumulation windows and let DDP surface
    # reducer-state errors that only appear on the following forward pass.
    SMOKE_MAX_TRAIN_STEPS: int = 4

    def resolve(self, value: str | Path) -> Path:
        """Resolve a configured path relative to ``train.py``."""

        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (SCRIPT_DIR / path).resolve()

    def enable_smoke_mode(self) -> None:
        """Apply small, isolated settings while retaining the real pipeline."""

        self.OUTPUT_DIR = "outputs/smoke"
        self.CHECKPOINT_DIR = "outputs/smoke/checkpoints"
        self.SUBMISSION_DIR = "outputs/smoke"
        self.CACHE_DIR = "cache/smoke"
        self.NUM_EPOCHS = 1
        self.PER_DEVICE_BATCH_SIZE = 2
        self.EVAL_BATCH_SIZE = 2
        self.GRADIENT_ACCUMULATION_STEPS = 1
        self.MAX_LENGTH = 64
        self.NUM_WORKERS = 0
        self.PREPROCESS_NUM_WORKERS = 1
        self.EARLY_STOPPING_PATIENCE = 1
        self.LOG_EVERY_N_STEPS = 1
        self.RESUME_FROM_CHECKPOINT = None


# ---------------------------------------------------------------------------
# 2. Logging, reproducibility, and distributed initialization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistributedContext:
    """Runtime rank, world size, backend, and device information."""

    rank: int
    local_rank: int
    world_size: int
    distributed: bool
    device: torch.device
    backend: Optional[str]

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def configure_logging(rank: int) -> None:
    """Configure concise rank-aware logging."""

    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f"%(asctime)s | rank={rank} | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch CPU, and all visible CUDA devices."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Deterministically seed an individual DataLoader worker."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def parse_args() -> argparse.Namespace:
    """Parse the intentionally small command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run the real pipeline on a tiny subset for a few steps.",
    )
    parser.add_argument(
        "--ddp-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank_argument",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def maybe_self_launch_ddp(args: argparse.Namespace) -> bool:
    """Relaunch this file with two torchrun workers when two GPUs exist."""

    already_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if args.ddp_worker or already_distributed or torch.cuda.device_count() < 2:
        return False

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        "--ddp-worker",
    ]
    if args.smoke_test:
        command.append("--smoke-test")
    print("Detected at least two GPUs; launching PyTorch DDP:")
    print(" ".join(command))
    subprocess.run(
        command,
        check=True,
        cwd=str(SCRIPT_DIR),
        env=os.environ.copy(),
    )
    return True


def initialize_distributed(config: Config) -> DistributedContext:
    """Initialize DDP from torchrun environment variables when present."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} CUDA devices exist."
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    backend: Optional[str] = None
    if distributed:
        backend = (
            "nccl"
            if device.type == "cuda" and dist.is_nccl_available() and os.name != "nt"
            else "gloo"
        )
        init_arguments: dict[str, Any] = {
            "backend": backend,
            "init_method": "env://",
            # Rank 0 may build the first D3Tok cache while its peer waits.
            "timeout": timedelta(minutes=config.DDP_TIMEOUT_MINUTES),
        }
        supports_device_id = False
        if backend == "nccl" and device.type == "cuda":
            try:
                supports_device_id = (
                    "device_id"
                    in inspect.signature(dist.init_process_group).parameters
                )
            except (TypeError, ValueError):
                # Older/wrapped PyTorch callables may not expose a signature.
                supports_device_id = False
        if supports_device_id:
            # Bind NCCL to LOCAL_RANK explicitly instead of asking it to infer a
            # device from the global rank (which also emits a hang warning).
            init_arguments["device_id"] = device
        dist.init_process_group(**init_arguments)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0

    configure_logging(rank)
    LOGGER.warning(
        "Runtime initialized: rank=%d local_rank=%d world_size=%d device=%s backend=%s",
        rank,
        local_rank,
        world_size,
        device,
        backend or "none",
    )
    return DistributedContext(rank, local_rank, world_size, distributed, device, backend)


def distributed_barrier(context: DistributedContext) -> None:
    """Synchronize ranks when DDP is active."""

    if context.distributed:
        if context.backend == "nccl" and context.device.type == "cuda":
            dist.barrier(device_ids=[context.local_rank])
        else:
            dist.barrier()


# ---------------------------------------------------------------------------
# 3. Safe tabular loading and Strict Track validation
# ---------------------------------------------------------------------------


ID_ALIASES = ("ID", "Sentence ID", "Sentence_ID")
TEXT_ALIASES = ("Sentence", "sentence", "text")
LABEL_ALIASES = ("Readability_Level_19", "Prediction", "label")
DOCUMENT_ALIASES = ("Document", "document")


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV, TSV, or Parquet without coercing textual IDs to numbers."""

    if not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=True)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=True)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format {suffix!r}: {path}")


def resolve_column(
    columns: Sequence[str],
    preferred: str,
    aliases: Sequence[str],
    role: str,
    *,
    required: bool,
) -> Optional[str]:
    """Resolve one unambiguous column, rejecting missing or multiple aliases."""

    choices = tuple(dict.fromkeys((preferred, *aliases)))
    matches = [name for name in choices if name in columns]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous {role} columns {matches}. Keep exactly one. "
            f"Available columns: {list(columns)}"
        )
    if not matches:
        if required:
            raise ValueError(
                f"Missing {role} column. Expected one of {list(choices)}. "
                f"Available columns: {list(columns)}"
            )
        return None
    return matches[0]


def load_split(
    path: Path,
    split_name: str,
    config: Config,
    *,
    require_label: bool,
) -> pd.DataFrame:
    """Load and validate a split while preserving row order and string IDs."""

    frame = read_table(path)
    columns = [str(column) for column in frame.columns]
    id_column = resolve_column(
        columns, config.ID_COLUMN, ID_ALIASES, "ID", required=True
    )
    text_column = resolve_column(
        columns, config.TEXT_COLUMN, TEXT_ALIASES, "text", required=True
    )
    label_column = resolve_column(
        columns,
        config.LABEL_COLUMN,
        LABEL_ALIASES,
        "label",
        required=require_label,
    )
    document_column = resolve_column(
        columns, "Document", DOCUMENT_ALIASES, "document", required=False
    )
    assert id_column is not None and text_column is not None

    if frame[id_column].isna().any():
        rows = frame.index[frame[id_column].isna()].tolist()[:10]
        raise ValueError(f"{split_name}: missing IDs at rows {rows}")
    frame["_id"] = frame[id_column].astype(str)
    blank_ids = frame["_id"].str.strip().eq("")
    if blank_ids.any():
        raise ValueError(
            f"{split_name}: blank IDs at rows {frame.index[blank_ids].tolist()[:10]}"
        )
    duplicate_ids = frame.loc[frame["_id"].duplicated(keep=False), "_id"].unique()
    if len(duplicate_ids):
        raise ValueError(
            f"{split_name}: duplicate IDs found, examples: {duplicate_ids[:10].tolist()}"
        )

    if frame[text_column].isna().any():
        rows = frame.index[frame[text_column].isna()].tolist()[:10]
        raise ValueError(f"{split_name}: missing sentences at rows {rows}")
    frame["_text"] = frame[text_column].astype(str)
    blank_text = frame["_text"].str.strip().eq("")
    if blank_text.any():
        raise ValueError(
            f"{split_name}: blank sentences at rows {frame.index[blank_text].tolist()[:10]}"
        )

    frame["_label"] = np.nan
    if label_column is not None:
        label_source = frame[label_column]
        if not require_label and label_source.isna().all():
            label_column = None
        else:
            if label_source.isna().any():
                rows = frame.index[label_source.isna()].tolist()[:10]
                raise ValueError(f"{split_name}: missing labels at rows {rows}")
            numeric = pd.to_numeric(label_source, errors="coerce")
            invalid_numeric = numeric.isna() | ~np.isfinite(numeric.to_numpy(dtype=float))
            non_integer = ~np.isclose(
                numeric.to_numpy(dtype=float), np.rint(numeric.to_numpy(dtype=float))
            )
            invalid = invalid_numeric.to_numpy() | non_integer
            if invalid.any():
                rows = np.flatnonzero(invalid)[:10].tolist()
                raise ValueError(f"{split_name}: non-integer labels at rows {rows}")
            labels = np.rint(numeric.to_numpy(dtype=float)).astype(np.int64)
            outside = (labels < config.MIN_LABEL) | (labels > config.MAX_LABEL)
            if outside.any():
                bad = sorted(set(labels[outside].tolist()))
                raise ValueError(
                    f"{split_name}: labels outside [{config.MIN_LABEL}, "
                    f"{config.MAX_LABEL}]: {bad}"
                )
            frame["_label"] = labels

    if document_column is not None:
        frame["_document"] = frame[document_column].map(
            lambda value: None if pd.isna(value) else str(value)
        )
    else:
        frame["_document"] = None
    frame["_original_index"] = np.arange(len(frame), dtype=np.int64)
    frame.attrs["has_labels"] = label_column is not None
    frame.attrs["source_path"] = str(path)
    return frame


def validate_split_isolation(
    train_frame: pd.DataFrame,
    dev_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> None:
    """Reject ID/document leakage and report official-content overlaps."""

    frames = {"train": train_frame, "dev": dev_frame, "test": test_frame}
    names = list(frames)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left = frames[left_name]
            right = frames[right_name]
            overlap_ids = set(left["_id"]) & set(right["_id"])
            if overlap_ids:
                raise ValueError(
                    f"ID leakage between {left_name} and {right_name}: "
                    f"{sorted(overlap_ids)[:10]}"
                )
            if left["_document"].notna().any() and right["_document"].notna().any():
                left_documents = set(left["_document"].dropna())
                right_documents = set(right["_document"].dropna())
                overlap_documents = left_documents & right_documents
                if overlap_documents:
                    raise ValueError(
                        f"Document leakage between {left_name} and {right_name}: "
                        f"{sorted(overlap_documents)[:10]}"
                    )
            duplicate_text_count = len(set(left["_text"]) & set(right["_text"]))
            if duplicate_text_count:
                LOGGER.warning(
                    "%s/%s share %d exact sentence texts; IDs/documents remain isolated.",
                    left_name,
                    right_name,
                    duplicate_text_count,
                )


def log_split_summary(name: str, frame: pd.DataFrame, config: Config) -> None:
    """Log split size and its 19-level label distribution when available."""

    LOGGER.info("%s: %d rows", name, len(frame))
    if frame.attrs.get("has_labels", False):
        counts = frame["_label"].astype(int).value_counts().sort_index()
        distribution = {
            label: int(counts.get(label, 0))
            for label in range(config.MIN_LABEL, config.MAX_LABEL + 1)
        }
        LOGGER.info("%s label distribution: %s", name, distribution)


def smoke_subset(frame: pd.DataFrame, size: int) -> pd.DataFrame:
    """Take a deterministic ordered smoke subset and reset its evaluation index."""

    subset = frame.iloc[: min(size, len(frame))].copy()
    subset["_original_index"] = np.arange(len(subset), dtype=np.int64)
    subset.attrs.update(frame.attrs)
    return subset


# ---------------------------------------------------------------------------
# 4. Arabic normalization, real CAMeL D3Tok, and preprocessing cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessResult:
    """One preprocessed sentence and an optional fallback diagnostic."""

    text: str
    error: Optional[str]


class ArabicD3TokPreprocessor:
    """Apply Unicode normalization, Tatweel removal, D3Tok, then dediacritize."""

    def __init__(self, resource: str) -> None:
        try:
            from camel_tools.disambig.mle import MLEDisambiguator
            from camel_tools.tokenizers.morphological import MorphologicalTokenizer
            from camel_tools.tokenizers.word import simple_word_tokenize
            from camel_tools.utils.dediac import dediac_ar
            from camel_tools.utils.normalize import normalize_unicode
        except ImportError as error:
            raise RuntimeError(
                "CAMeL Tools is required for real D3Tok. Install requirements.txt first."
            ) from error

        self._simple_word_tokenize = simple_word_tokenize
        self._dediac_ar = dediac_ar
        self._normalize_unicode = normalize_unicode
        disambiguator = MLEDisambiguator.pretrained(resource)
        self._d3tok = MorphologicalTokenizer(
            disambiguator=disambiguator,
            scheme="d3tok",
            split=True,
            diac=True,
        )

    def normalize_and_remove_tatweel(self, text: str) -> str:
        """Perform compatibility Unicode normalization and remove only U+0640."""

        normalized = self._normalize_unicode(text, compatibility=True)
        return normalized.replace(TATWEEL, "")

    def fallback(self, normalized_text: str) -> str:
        """Preserve normalized content when morphological tokenization fails."""

        return self._dediac_ar(normalized_text).replace(TATWEEL, "")

    def process(self, text: str) -> PreprocessResult:
        """Preprocess one sentence, using a content-preserving per-row fallback."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("D3Tok received an empty/non-string sentence")
        try:
            normalized = self.normalize_and_remove_tatweel(text)
        except Exception as error:
            normalized = unicodedata.normalize("NFKC", text).replace(TATWEEL, "")
            fallback = self._dediac_ar(normalized)
            return PreprocessResult(
                fallback,
                f"UnicodeNormalizationError: {type(error).__name__}: {error}",
            )

        try:
            words = self._simple_word_tokenize(normalized)
            d3_tokens = self._d3tok.tokenize(words)
            output_tokens = [self._dediac_ar(str(token)) for token in d3_tokens]
            processed = " ".join(token for token in output_tokens if token != "")
            if normalized.strip() and not processed.strip():
                raise ValueError("D3Tok returned no content")
            return PreprocessResult(processed, None)
        except Exception as error:
            return PreprocessResult(
                self.fallback(normalized),
                f"D3TokError: {type(error).__name__}: {error}",
            )


def camel_data_install_command(package: str) -> list[str]:
    """Return the most reliable CAMeL data installer command available."""

    executable = shutil.which("camel_data")
    if executable:
        return [executable, "-i", package]
    return [sys.executable, "-m", "camel_tools.cli.camel_data", "-i", package]


def create_preprocessor(config: Config, *, allow_download: bool) -> ArabicD3TokPreprocessor:
    """Load D3Tok and optionally install CAMeL's light data bundle once."""

    try:
        return ArabicD3TokPreprocessor(config.D3TOK_RESOURCE)
    except Exception as first_error:
        if allow_download and config.AUTO_DOWNLOAD_CAMEL_DATA:
            command = camel_data_install_command(config.CAMEL_DATA_PACKAGE)
            LOGGER.warning(
                "Could not load CAMeL resource %s; installing data package %s.",
                config.D3TOK_RESOURCE,
                config.CAMEL_DATA_PACKAGE,
            )
            try:
                subprocess.run(command, check=True, cwd=str(SCRIPT_DIR))
                return ArabicD3TokPreprocessor(config.D3TOK_RESOURCE)
            except Exception as install_error:
                raise RuntimeError(
                    "Unable to load the required real D3Tok resource "
                    f"{config.D3TOK_RESOURCE!r}. Run `camel_data -i "
                    f"{config.CAMEL_DATA_PACKAGE}` with internet access. "
                    f"Initial error: {first_error}. Install error: {install_error}"
                ) from install_error
        raise RuntimeError(
            "Unable to load the required real D3Tok resource "
            f"{config.D3TOK_RESOURCE!r}. Run `camel_data -i "
            f"{config.CAMEL_DATA_PACKAGE}`. Error: {first_error}"
        ) from first_error


_PROCESS_PREPROCESSOR: Optional[ArabicD3TokPreprocessor] = None


def _preprocess_worker_init(resource: str) -> None:
    global _PROCESS_PREPROCESSOR
    _PROCESS_PREPROCESSOR = ArabicD3TokPreprocessor(resource)


def _preprocess_worker_item(item: tuple[str, str]) -> tuple[str, str, Optional[str]]:
    if _PROCESS_PREPROCESSOR is None:
        raise RuntimeError("Preprocessing worker was not initialized")
    row_id, text = item
    result = _PROCESS_PREPROCESSOR.process(text)
    return row_id, result.text, result.error


def package_version(package: str) -> str:
    """Return an installed package version for cache invalidation."""

    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def preprocessing_fingerprint(
    frame: pd.DataFrame,
    split_name: str,
    source_path: Path,
    config: Config,
) -> str:
    """Hash source identity, ordered ID/text cells, and preprocessing semantics."""

    digest = hashlib.sha256()
    settings = {
        "version": PREPROCESSING_VERSION,
        "split": split_name,
        "source_path": str(source_path.resolve()),
        "id_column": config.ID_COLUMN,
        "text_column": config.TEXT_COLUMN,
        "resource": config.D3TOK_RESOURCE,
        "unicode": "CAMeL normalize_unicode compatibility=True",
        "tatweel": "remove U+0640 before D3Tok",
        "dediac": "CAMeL dediac_ar after D3Tok",
        "camel_tools": package_version("camel-tools"),
    }
    digest.update(json.dumps(settings, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    for row_id, text in zip(frame["_id"], frame["_text"]):
        encoded_id = str(row_id).encode("utf-8")
        encoded_text = str(text).encode("utf-8")
        digest.update(len(encoded_id).to_bytes(8, "big"))
        digest.update(encoded_id)
        digest.update(len(encoded_text).to_bytes(8, "big"))
        digest.update(encoded_text)
    return digest.hexdigest()


def run_arabic_preprocessing_checks(preprocessor: ArabicD3TokPreprocessor) -> None:
    """Exercise Tatweel removal, true D3Tok, dediacritization, and fallback."""

    sample = "الكتــــابُ مفيدٌ."
    normalized = preprocessor.normalize_and_remove_tatweel(sample)
    if TATWEEL in normalized:
        raise AssertionError("Tatweel removal check failed")
    result = preprocessor.process(sample)
    if not result.text.strip():
        raise AssertionError("D3Tok content-preservation check failed")
    if TATWEEL in result.text or any(char in ARABIC_DIACRITICS for char in result.text):
        raise AssertionError("Post-D3Tok Tatweel/diacritic check failed")
    plus_probe = preprocessor.fallback("ال+كِتَابُ")
    if "+" not in plus_probe or any(char in ARABIC_DIACRITICS for char in plus_probe):
        raise AssertionError("dediac_ar must retain D3Tok's '+' marker")
    fallback_probe = preprocessor.fallback(normalized)
    if not fallback_probe.strip() or TATWEEL in fallback_probe:
        raise AssertionError("Fallback content-preservation check failed")

    class _ForcedD3TokFailure:
        def tokenize(self, words: Sequence[str]) -> list[str]:
            del words
            raise RuntimeError("forced internal fallback check")

    real_d3tok = preprocessor._d3tok
    try:
        preprocessor._d3tok = _ForcedD3TokFailure()
        forced_fallback = preprocessor.process(sample)
    finally:
        preprocessor._d3tok = real_d3tok
    if not forced_fallback.error or not forced_fallback.error.startswith("D3TokError:"):
        raise AssertionError("Forced D3Tok failure was not diagnosed")
    if not forced_fallback.text.strip() or any(
        char in ARABIC_DIACRITICS for char in forced_fallback.text
    ):
        raise AssertionError("Forced D3Tok fallback did not preserve clean content")


def build_preprocessing_cache(
    frame: pd.DataFrame,
    cache_path: Path,
    config: Config,
) -> None:
    """Build a Parquet D3Tok cache atomically on the main process."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    items = list(zip(frame["_id"].astype(str), frame["_text"].astype(str)))
    rows: list[tuple[str, str, Optional[str]]]

    if config.PREPROCESS_NUM_WORKERS > 1:
        probe = create_preprocessor(config, allow_download=True)
        run_arabic_preprocessing_checks(probe)
        del probe
        with ProcessPoolExecutor(
            max_workers=config.PREPROCESS_NUM_WORKERS,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_preprocess_worker_init,
            initargs=(config.D3TOK_RESOURCE,),
        ) as executor:
            rows = list(
                tqdm(
                    executor.map(_preprocess_worker_item, items, chunksize=32),
                    total=len(items),
                    desc=f"D3Tok {cache_path.stem}",
                )
            )
    else:
        preprocessor = create_preprocessor(config, allow_download=True)
        run_arabic_preprocessing_checks(preprocessor)
        rows = []
        for row_id, text in tqdm(items, desc=f"D3Tok {cache_path.stem}"):
            result = preprocessor.process(text)
            rows.append((row_id, result.text, result.error))

    cache_frame = pd.DataFrame(
        {
            "_id": [row[0] for row in rows],
            "_original_index": frame["_original_index"].to_numpy(dtype=np.int64),
            "_processed_text": [row[1] for row in rows],
            "_fallback_error": [row[2] for row in rows],
        }
    )
    if cache_frame["_processed_text"].isna().any() or cache_frame["_processed_text"].eq("").any():
        raise RuntimeError("Preprocessing produced an empty cached sentence")
    temporary_path = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        cache_frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_cached_preprocessing(frame: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    """Attach cached text only after exact ordered ID/index validation."""

    cache_frame = pd.read_parquet(cache_path)
    expected_ids = frame["_id"].astype(str).tolist()
    cached_ids = cache_frame["_id"].astype(str).tolist()
    if cached_ids != expected_ids:
        raise RuntimeError(f"Cache ID/order mismatch: {cache_path}")
    expected_indices = frame["_original_index"].astype(int).tolist()
    cached_indices = cache_frame["_original_index"].astype(int).tolist()
    if cached_indices != expected_indices:
        raise RuntimeError(f"Cache index mismatch: {cache_path}")
    output = frame.copy()
    output["_processed_text"] = cache_frame["_processed_text"].astype(str).to_numpy()
    output["_fallback_error"] = cache_frame["_fallback_error"].to_numpy()
    output.attrs.update(frame.attrs)
    return output


def preprocess_split_cached(
    frame: pd.DataFrame,
    split_name: str,
    source_path: Path,
    config: Config,
    context: DistributedContext,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a fingerprinted cache on rank 0, then load it on every rank."""

    fingerprint = preprocessing_fingerprint(frame, split_name, source_path, config)
    cache_path = config.resolve(config.CACHE_DIR) / f"{split_name}-{fingerprint}.parquet"
    status: list[Optional[str]] = [None]
    if context.is_main:
        try:
            if config.FORCE_REPROCESS or not cache_path.is_file():
                LOGGER.info("Building %s preprocessing cache: %s", split_name, cache_path)
                build_preprocessing_cache(frame, cache_path, config)
            else:
                LOGGER.info("Reusing %s preprocessing cache: %s", split_name, cache_path)
        except Exception as error:
            status[0] = f"{type(error).__name__}: {error}"
    if context.distributed:
        dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(f"Failed to prepare {split_name} cache: {status[0]}")
    distributed_barrier(context)

    processed = load_cached_preprocessing(frame, cache_path)
    failures = processed.loc[processed["_fallback_error"].notna(), ["_id", "_fallback_error"]]
    report = {
        "split": split_name,
        "rows": len(processed),
        "cache": str(cache_path),
        "fingerprint": fingerprint,
        "fallback_count": len(failures),
        # Keep every affected ID/error so preprocessing failures are auditable.
        "fallback_errors": failures.to_dict(orient="records"),
    }
    if context.is_main and len(failures):
        LOGGER.warning(
            "%s used the content-preserving D3Tok fallback for %d/%d rows.",
            split_name,
            len(failures),
            len(processed),
        )
    return processed, report


# ---------------------------------------------------------------------------
# 5. Dataset, collator, and distributed weighted sampling
# ---------------------------------------------------------------------------


class BARECDataset(Dataset[dict[str, Any]]):
    """Tokenize cached D3Tok sentences lazily while retaining IDs and row indices."""

    def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int) -> None:
        self.texts = frame["_processed_text"].astype(str).tolist()
        self.ids = frame["_id"].astype(str).tolist()
        self.indices = frame["_original_index"].astype(int).tolist()
        self.has_labels = bool(frame.attrs.get("has_labels", False))
        self.labels = (
            frame["_label"].astype(float).tolist() if self.has_labels else None
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        item: dict[str, Any] = dict(encoded)
        item["sample_id"] = self.ids[index]
        item["original_index"] = self.indices[index]
        if self.labels is not None:
            item["label"] = self.labels[index]
        return item


class BARECCollator:
    """Dynamically pad model inputs without passing IDs/labels into the encoder."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        model_features: list[dict[str, Any]] = []
        sample_ids: list[str] = []
        indices: list[int] = []
        labels: list[float] = []
        labels_present = "label" in features[0]
        for feature in features:
            model_features.append(
                {
                    key: value
                    for key, value in feature.items()
                    if key not in {"sample_id", "original_index", "label"}
                }
            )
            sample_ids.append(str(feature["sample_id"]))
            indices.append(int(feature["original_index"]))
            if labels_present:
                labels.append(float(feature["label"]))
        batch = dict(self.tokenizer.pad(model_features, padding=True, return_tensors="pt"))
        batch["sample_ids"] = sample_ids
        batch["original_indices"] = torch.tensor(indices, dtype=torch.long)
        if labels_present:
            batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch


class DistributedWeightedSampler(Sampler[int]):
    """Draw one deterministic global weighted sample list and shard it by rank.

    Every rank reconstructs the same list using ``seed + epoch``.  Taking every
    ``num_replicas``-th element gives equal-length, non-overlapping positions in
    that global draw, so DDP ranks always execute the same number of steps.
    """

    def __init__(
        self,
        weights: Sequence[float] | torch.Tensor,
        *,
        num_replicas: int,
        rank: int,
        replacement: bool,
        seed: int,
    ) -> None:
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("Invalid rank/num_replicas for weighted sampler")
        self.weights = torch.as_tensor(weights, dtype=torch.double, device="cpu")
        if self.weights.ndim != 1 or len(self.weights) == 0:
            raise ValueError("Sampler weights must be a non-empty 1D sequence")
        if not torch.isfinite(self.weights).all() or (self.weights <= 0).any():
            raise ValueError("Sampler weights must be finite and strictly positive")
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0
        if replacement:
            self.num_samples = math.ceil(len(self.weights) / num_replicas)
        else:
            self.num_samples = len(self.weights) // num_replicas
        if self.num_samples == 0:
            raise ValueError("Dataset is too small for the requested world size")
        self.total_size = self.num_samples * num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        global_indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=self.replacement,
            generator=generator,
        ).tolist()
        rank_indices = global_indices[self.rank : self.total_size : self.num_replicas]
        if len(rank_indices) != self.num_samples:
            raise AssertionError("Distributed weighted sampler produced uneven shards")
        return iter(rank_indices)

    def __len__(self) -> int:
        return self.num_samples


def sample_weights_from_labels(
    labels: Sequence[int], config: Config
) -> tuple[np.ndarray, dict[int, int], dict[int, float]]:
    """Compute per-sample inverse-frequency weights with configurable smoothing."""

    label_array = np.asarray(labels, dtype=np.int64)
    counts = {
        label: int(np.sum(label_array == label))
        for label in range(config.MIN_LABEL, config.MAX_LABEL + 1)
    }
    class_weights = {
        label: (1.0 / count) ** config.SAMPLER_ALPHA
        for label, count in counts.items()
        if count > 0
    }
    weights = np.asarray([class_weights[int(label)] for label in label_array], dtype=np.float64)
    return weights, counts, class_weights


def run_sampler_checks() -> None:
    """Verify deterministic equal-size two-rank weighted sampling."""

    weights = [1.0, 2.0, 3.0, 4.0, 5.0]
    left = DistributedWeightedSampler(
        weights, num_replicas=2, rank=0, replacement=True, seed=17
    )
    right = DistributedWeightedSampler(
        weights, num_replicas=2, rank=1, replacement=True, seed=17
    )
    left.set_epoch(3)
    right.set_epoch(3)
    left_indices = list(left)
    right_indices = list(right)
    if len(left_indices) != len(right_indices):
        raise AssertionError("Weighted sampler rank lengths differ")
    repeated = list(left)
    if repeated != left_indices:
        raise AssertionError("Weighted sampler is not deterministic")
    interleaved: list[int] = []
    for pair in zip(left_indices, right_indices):
        interleaved.extend(pair)
    generator = torch.Generator().manual_seed(20)
    expected = torch.multinomial(
        torch.tensor(weights, dtype=torch.double),
        left.total_size,
        replacement=True,
        generator=generator,
    ).tolist()
    if interleaved != expected:
        raise AssertionError("Weighted sampler shards do not reconstruct the global draw")


# ---------------------------------------------------------------------------
# 6. Regression model and optimization
# ---------------------------------------------------------------------------


class ArabicReadabilityRegressor(nn.Module):
    """Fine-tune the complete task-specific AraBERT regression model."""

    def __init__(
        self,
        model_name: str,
        dropout: float,
        output_offset: float,
    ) -> None:
        super().__init__()
        pretrained = AutoModelForSequenceClassification.from_pretrained(model_name)
        if int(pretrained.config.num_labels) != 1:
            raise RuntimeError(
                "The configured checkpoint must have one regression output, "
                f"but {model_name!r} declares {pretrained.config.num_labels}."
            )
        classifier = getattr(pretrained, "classifier", None)
        if not isinstance(classifier, nn.Linear) or classifier.out_features != 1:
            raise RuntimeError(
                "The configured checkpoint must expose a one-output linear "
                "classifier so its pretrained regression head can be retained."
            )
        checkpoint_dropout = getattr(pretrained, "dropout", None)
        self.encoder = pretrained.base_model
        self.dropout = (
            checkpoint_dropout
            if isinstance(checkpoint_dropout, nn.Module)
            else nn.Dropout(dropout)
        )
        self.regression_head = classifier
        self.output_offset = float(output_offset)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return one unrounded readability score per input sentence."""

        encoder_arguments: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_arguments["token_type_ids"] = token_type_ids
        outputs = self.encoder(**encoder_arguments)
        if outputs.pooler_output is None:
            raise RuntimeError(
                "The pretrained regression checkpoint did not return pooler_output."
            )
        zero_based_score = self.regression_head(
            self.dropout(outputs.pooler_output)
        ).squeeze(-1)
        return zero_based_score + self.output_offset


def unwrap_model(model: nn.Module) -> ArabicReadabilityRegressor:
    """Return the underlying model whether or not DDP wraps it."""

    unwrapped = model.module if isinstance(model, DDP) else model
    if not isinstance(unwrapped, ArabicReadabilityRegressor):
        raise TypeError(f"Unexpected model type: {type(unwrapped)}")
    return unwrapped


def split_decay_parameters(module: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Separate weight-decayed tensors from biases and normalization weights."""

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if name.endswith("bias") or "layernorm" in lowered or "layer_norm" in lowered:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return decay, no_decay


def create_optimizer(model: nn.Module, config: Config) -> AdamW:
    """Create AdamW groups with independent encoder/head learning rates."""

    base = unwrap_model(model)
    encoder_decay, encoder_no_decay = split_decay_parameters(base.encoder)
    head_decay, head_no_decay = split_decay_parameters(
        nn.ModuleList([base.dropout, base.regression_head])
    )
    groups = [
        {
            "params": encoder_decay,
            "lr": config.ENCODER_LR,
            "weight_decay": config.WEIGHT_DECAY,
            "group_name": "encoder_decay",
        },
        {
            "params": encoder_no_decay,
            "lr": config.ENCODER_LR,
            "weight_decay": 0.0,
            "group_name": "encoder_no_decay",
        },
        {
            "params": head_decay,
            "lr": config.HEAD_LR,
            "weight_decay": config.WEIGHT_DECAY,
            "group_name": "head_decay",
        },
        {
            "params": head_no_decay,
            "lr": config.HEAD_LR,
            "weight_decay": 0.0,
            "group_name": "head_no_decay",
        },
    ]
    return AdamW(groups)


def make_grad_scaler(enabled: bool) -> Any:
    """Construct GradScaler across old and new PyTorch AMP APIs."""

    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool) -> contextlib.AbstractContextManager[Any]:
    """Return CUDA FP16 autocast or a no-op context."""

    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def scaled_optimizer_step(
    scaler: Any,
    optimizer: torch.optim.Optimizer,
) -> tuple[bool, float, float]:
    """Run a scaled optimizer update and report whether AMP skipped it."""

    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    optimizer_stepped = not scaler.is_enabled() or scale_after >= scale_before
    return optimizer_stepped, scale_before, scale_after


def model_forward(model: nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
    """Pass only encoder tensors from a collated batch into the model."""

    predictions = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch.get("token_type_ids"),
    )
    if predictions.ndim != 1 or predictions.shape[0] != batch["input_ids"].shape[0]:
        raise AssertionError(
            f"Regression output must have shape [batch_size], got {tuple(predictions.shape)}"
        )
    return predictions


def run_regression_shape_check(
    model: ArabicReadabilityRegressor,
    tokenizer: Any,
    text: str,
    device: torch.device,
    max_length: int,
) -> None:
    """Verify batch-one output shape and gradient connectivity before DDP."""

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        output = model(**encoded)
        if output.shape != torch.Size([1]):
            raise AssertionError(
                "Batch-size-one regression output must have shape [1], "
                f"got {tuple(output.shape)}"
            )
        output.sum().backward()
        disconnected = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        if disconnected:
            raise AssertionError(
                "Trainable model parameters are disconnected from the regression "
                f"loss: {disconnected}"
            )
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)


# ---------------------------------------------------------------------------
# 7. Metrics and distributed evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvaluationOutput:
    """Ordered predictions, optional labels, and calculated metrics on rank 0."""

    ids: list[str]
    indices: np.ndarray
    raw_predictions: np.ndarray
    final_predictions: np.ndarray
    labels: Optional[np.ndarray]
    metrics: Optional[dict[str, float]]


def round_and_clip(raw_predictions: Sequence[float], config: Config) -> np.ndarray:
    """Convert regression scores to legal integer readability levels."""

    raw = np.asarray(raw_predictions, dtype=np.float64)
    return np.clip(np.rint(raw), config.MIN_LABEL, config.MAX_LABEL).astype(np.int64)


def calculate_metrics(
    labels: Sequence[float], raw_predictions: Sequence[float], config: Config
) -> dict[str, float]:
    """Calculate raw MSE and integer-label BAREC metrics."""

    truth = np.asarray(labels, dtype=np.int64)
    raw = np.asarray(raw_predictions, dtype=np.float64)
    final = round_and_clip(raw, config)
    qwk = float(
        cohen_kappa_score(
            truth,
            final,
            weights="quadratic",
            labels=list(range(config.MIN_LABEL, config.MAX_LABEL + 1)),
        )
    )
    return {
        "mse": float(np.mean(np.square(raw - truth))),
        "mae": float(np.mean(np.abs(final - truth))),
        "qwk": qwk,
        "exact_accuracy": float(np.mean(final == truth)),
        "adjacent_accuracy": float(np.mean(np.abs(final - truth) <= 1)),
    }


def merge_evaluation_payloads(
    payloads: Sequence[Mapping[str, Any]], expected_size: int
) -> tuple[list[str], np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Sort gathered records by original index and remove sampler padding."""

    records: dict[int, tuple[str, float, Optional[float]]] = {}
    any_labels = False
    for payload in payloads:
        indices = payload["indices"]
        ids = payload["ids"]
        predictions = payload["predictions"]
        labels = payload.get("labels")
        if not (len(indices) == len(ids) == len(predictions)):
            raise RuntimeError("Malformed distributed evaluation payload")
        if labels is not None and len(labels) != len(indices):
            raise RuntimeError("Malformed distributed evaluation labels")
        any_labels = any_labels or labels is not None
        for position, index in enumerate(indices):
            integer_index = int(index)
            record_label = None if labels is None else float(labels[position])
            candidate = (str(ids[position]), float(predictions[position]), record_label)
            if integer_index in records:
                previous = records[integer_index]
                if previous[0] != candidate[0]:
                    raise RuntimeError("Padded evaluation index maps to different IDs")
                continue
            records[integer_index] = candidate

    expected_indices = list(range(expected_size))
    if sorted(records) != expected_indices:
        missing = sorted(set(expected_indices) - set(records))[:10]
        extra = sorted(set(records) - set(expected_indices))[:10]
        raise RuntimeError(
            f"Evaluation gather mismatch: expected={expected_size}, "
            f"unique={len(records)}, missing={missing}, extra={extra}"
        )
    ordered = [records[index] for index in expected_indices]
    ordered_ids = [record[0] for record in ordered]
    predictions = np.asarray([record[1] for record in ordered], dtype=np.float64)
    label_array: Optional[np.ndarray] = None
    if any_labels:
        if any(record[2] is None for record in ordered):
            raise RuntimeError("Only part of the evaluation set has labels")
        label_array = np.asarray([record[2] for record in ordered], dtype=np.float64)
    return ordered_ids, np.arange(expected_size), predictions, label_array


def run_gather_order_check() -> None:
    """Verify out-of-order/padded evaluation records are restored exactly once."""

    payloads = [
        {"indices": [2, 0], "ids": ["c", "a"], "predictions": [3.0, 1.0]},
        {"indices": [1, 0], "ids": ["b", "a"], "predictions": [2.0, 1.0]},
    ]
    ids, indices, predictions, labels = merge_evaluation_payloads(payloads, 3)
    if ids != ["a", "b", "c"] or indices.tolist() != [0, 1, 2]:
        raise AssertionError("Evaluation gather ordering check failed")
    if predictions.tolist() != [1.0, 2.0, 3.0] or labels is not None:
        raise AssertionError("Evaluation gather padding check failed")


def move_model_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move only tensors to the selected accelerator."""

    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader[Any],
    expected_size: int,
    config: Config,
    context: DistributedContext,
    *,
    description: str,
) -> Optional[EvaluationOutput]:
    """Run distributed inference and return ordered results on rank 0 only."""

    model.eval()
    local_indices: list[int] = []
    local_ids: list[str] = []
    local_predictions: list[float] = []
    local_labels: Optional[list[float]] = []
    amp_enabled = config.USE_FP16 and context.device.type == "cuda"
    progress = tqdm(loader, desc=description, disable=not context.is_main, leave=False)
    for batch in progress:
        batch = move_model_batch(batch, context.device)
        with autocast_context(amp_enabled):
            predictions = model_forward(model, batch)
        local_indices.extend(batch["original_indices"].cpu().tolist())
        local_ids.extend(batch["sample_ids"])
        local_predictions.extend(predictions.float().cpu().tolist())
        if "labels" in batch:
            assert local_labels is not None
            local_labels.extend(batch["labels"].float().cpu().tolist())
        else:
            local_labels = None

    payload: dict[str, Any] = {
        "indices": local_indices,
        "ids": local_ids,
        "predictions": local_predictions,
        "labels": local_labels,
    }
    if context.distributed:
        gathered: list[Optional[dict[str, Any]]] = [None] * context.world_size
        dist.all_gather_object(gathered, payload)
        payloads = [item for item in gathered if item is not None]
    else:
        payloads = [payload]

    if not context.is_main:
        return None
    ids, indices, raw_predictions, labels = merge_evaluation_payloads(
        payloads, expected_size
    )
    final_predictions = round_and_clip(raw_predictions, config)
    metrics = calculate_metrics(labels, raw_predictions, config) if labels is not None else None
    return EvaluationOutput(
        ids,
        indices,
        raw_predictions,
        final_predictions,
        labels,
        metrics,
    )


def broadcast_object(value: Any, context: DistributedContext) -> Any:
    """Broadcast a small Python decision/metric payload from rank 0."""

    if not context.distributed:
        return value
    objects = [value if context.is_main else None]
    dist.broadcast_object_list(objects, src=0)
    return objects[0]


# ---------------------------------------------------------------------------
# 8. Checkpoint management
# ---------------------------------------------------------------------------


def json_safe(value: Any) -> Any:
    """Convert dataclasses/NumPy/non-finite values into strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json_dump(payload: Any, path: Path) -> None:
    """Write JSON atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(payload: Any, path: Path) -> None:
    """Write a torch checkpoint atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_torch_file(path: Path, device: torch.device) -> Any:
    """Load trusted local checkpoints across PyTorch's weights_only API change."""

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def cpu_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy an unwrapped model state to CPU for portable checkpoints."""

    return {
        name: tensor.detach().cpu()
        for name, tensor in unwrap_model(model).state_dict().items()
    }


def local_rng_state() -> dict[str, Any]:
    """Capture this rank's Python, NumPy, CPU, and CUDA RNG streams."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        # Save only this worker's device.  A rank must not initialize or depend
        # on the RNG streams of every visible GPU.
        "cuda": (
            torch.cuda.get_rng_state(torch.cuda.current_device()).cpu()
            if torch.cuda.is_available()
            else None
        ),
    }


def gather_rng_states(context: DistributedContext) -> list[dict[str, Any]]:
    """Gather resumable RNG states for every DDP rank."""

    state = local_rng_state()
    if not context.distributed:
        return [state]
    gathered: list[Optional[dict[str, Any]]] = [None] * context.world_size
    dist.all_gather_object(gathered, state)
    return [item for item in gathered if item is not None]


def restore_rng_state(states: Sequence[Mapping[str, Any]], rank: int) -> None:
    """Restore this rank's RNG state from a checkpoint when available."""

    if not states:
        return
    state = states[min(rank, len(states) - 1)]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state(
            state["cuda"].cpu(), device=torch.cuda.current_device()
        )


def save_best_model(
    model: nn.Module,
    tokenizer: Any,
    config: Config,
    metrics: Mapping[str, float],
) -> Path:
    """Save the task model, tokenizer, config, and best Dev metrics."""

    directory = config.resolve(config.OUTPUT_DIR) / "best_model"
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model_state.pt"
    atomic_torch_save(cpu_model_state(model), model_path)
    tokenizer.save_pretrained(directory / "tokenizer")
    atomic_json_dump(asdict(config), directory / "training_config.json")
    atomic_json_dump(dict(metrics), directory / "metrics.json")
    return model_path


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    *,
    epoch: int,
    global_step: int,
    best_qwk: float,
    best_mae: float,
    bad_epochs: int,
    config: Config,
    rng_states: Sequence[Mapping[str, Any]],
) -> None:
    """Save all states needed to resume training and model selection."""

    payload = {
        "model_state": cpu_model_state(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_qwk": best_qwk,
        "best_mae": best_mae,
        "bad_epochs": bad_epochs,
        "config": asdict(config),
        "rng_states": list(rng_states),
    }
    atomic_torch_save(payload, path)


def resume_training(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    context: DistributedContext,
) -> tuple[int, int, float, float, int]:
    """Restore a full training checkpoint and return selection state."""

    checkpoint = load_torch_file(checkpoint_path, context.device)
    unwrap_model(model).load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    scaler_state = checkpoint.get("scaler_state")
    if scaler_state:
        scaler.load_state_dict(scaler_state)
    restore_rng_state(checkpoint.get("rng_states", []), context.rank)
    start_epoch = int(checkpoint["epoch"]) + 1
    LOGGER.info("Resumed training from %s at epoch %d", checkpoint_path, start_epoch + 1)
    return (
        start_epoch,
        int(checkpoint.get("global_step", 0)),
        float(checkpoint.get("best_qwk", -math.inf)),
        float(checkpoint.get("best_mae", math.inf)),
        int(checkpoint.get("bad_epochs", 0)),
    )


# ---------------------------------------------------------------------------
# 9. DataLoaders and training loop
# ---------------------------------------------------------------------------


def make_data_loaders(
    train_frame: pd.DataFrame,
    dev_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    tokenizer: Any,
    config: Config,
    context: DistributedContext,
) -> tuple[
    DataLoader[Any],
    DataLoader[Any],
    DataLoader[Any],
    Sampler[int],
]:
    """Construct train/dev/test loaders with Strict Track-safe sampling roles."""

    train_dataset = BARECDataset(train_frame, tokenizer, config.MAX_LENGTH)
    dev_dataset = BARECDataset(dev_frame, tokenizer, config.MAX_LENGTH)
    test_dataset = BARECDataset(test_frame, tokenizer, config.MAX_LENGTH)
    collator = BARECCollator(tokenizer)

    if config.USE_WEIGHTED_SAMPLER:
        weights, class_counts, class_weights = sample_weights_from_labels(
            train_frame["_label"].astype(int).tolist(), config
        )
        if context.is_main:
            LOGGER.info("Train class counts: %s", class_counts)
            LOGGER.info("Train class weights (alpha=%s): %s", config.SAMPLER_ALPHA, class_weights)
        train_sampler: Sampler[int] = DistributedWeightedSampler(
            weights,
            num_replicas=context.world_size,
            rank=context.rank,
            replacement=config.SAMPLER_REPLACEMENT,
            seed=config.SEED,
        )
    else:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=config.SEED,
            drop_last=True,
        )

    if context.distributed:
        dev_sampler: Sampler[int] = DistributedSampler(
            dev_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=False,
            drop_last=False,
        )
        test_sampler: Sampler[int] = DistributedSampler(
            test_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        dev_sampler = SequentialSampler(dev_dataset)
        test_sampler = SequentialSampler(test_dataset)

    generator = torch.Generator().manual_seed(config.SEED + context.rank)
    common = {
        "collate_fn": collator,
        "num_workers": config.NUM_WORKERS,
        "pin_memory": config.PIN_MEMORY and context.device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": config.NUM_WORKERS > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.PER_DEVICE_BATCH_SIZE,
        sampler=train_sampler,
        drop_last=True,
        **common,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        sampler=dev_sampler,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        sampler=test_sampler,
        drop_last=False,
        **common,
    )
    if len(train_loader) == 0:
        raise ValueError(
            "Train DataLoader has zero batches. Reduce PER_DEVICE_BATCH_SIZE or world size."
        )
    return train_loader, dev_loader, test_loader, train_sampler


def learning_rates(optimizer: torch.optim.Optimizer) -> tuple[float, float]:
    """Read current encoder and head learning rates from named groups."""

    encoder_lr = next(
        float(group["lr"])
        for group in optimizer.param_groups
        if str(group.get("group_name", "")).startswith("encoder")
    )
    head_lr = next(
        float(group["lr"])
        for group in optimizer.param_groups
        if str(group.get("group_name", "")).startswith("head")
    )
    return encoder_lr, head_lr


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    config: Config,
    context: DistributedContext,
    epoch: int,
    global_step: int,
    *,
    max_steps: Optional[int],
) -> tuple[float, int]:
    """Train exactly one epoch on Train; Dev/Test never call this function."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_function = nn.MSELoss()
    amp_enabled = config.USE_FP16 and context.device.type == "cuda"
    total_loss = 0.0
    total_examples = 0
    start = time.perf_counter()
    effective_steps = len(loader) if max_steps is None else min(len(loader), max_steps)
    progress = tqdm(
        total=effective_steps,
        desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS}",
        disable=not context.is_main,
    )

    for step, batch in enumerate(loader):
        if step >= effective_steps:
            break
        batch = move_model_batch(batch, context.device)
        is_update_step = (
            (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0
            or step + 1 == effective_steps
        )
        synchronization = (
            contextlib.nullcontext()
            if is_update_step or not isinstance(model, DDP)
            else model.no_sync()
        )
        with synchronization:
            with autocast_context(amp_enabled):
                predictions = model_forward(model, batch)
                raw_loss = loss_function(predictions, batch["labels"])
                window_start = (
                    step // config.GRADIENT_ACCUMULATION_STEPS
                ) * config.GRADIENT_ACCUMULATION_STEPS
                accumulation_window = min(
                    config.GRADIENT_ACCUMULATION_STEPS,
                    effective_steps - window_start,
                )
                scaled_loss = raw_loss / accumulation_window
            scaler.scale(scaled_loss).backward()

        batch_size = int(batch["labels"].shape[0])
        total_loss += float(raw_loss.detach().item()) * batch_size
        total_examples += batch_size

        if is_update_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer_stepped, scale_before, scale_after = scaled_optimizer_step(
                scaler, optimizer
            )
            if optimizer_stepped:
                scheduler.step()
                global_step += 1
            elif context.is_main:
                LOGGER.warning(
                    "AMP overflow: optimizer and scheduler update skipped; "
                    "GradScaler %.0f -> %.0f.",
                    scale_before,
                    scale_after,
                )
            optimizer.zero_grad(set_to_none=True)

        if context.is_main:
            encoder_lr, head_lr = learning_rates(optimizer)
            elapsed = time.perf_counter() - start
            memory = "cpu"
            if context.device.type == "cuda":
                allocated = torch.cuda.memory_allocated(context.device) / (1024**3)
                reserved = torch.cuda.memory_reserved(context.device) / (1024**3)
                memory = f"{allocated:.2f}/{reserved:.2f}GiB"
            progress.update(1)
            if (
                (step + 1) % config.LOG_EVERY_N_STEPS == 0
                or step + 1 == effective_steps
            ):
                progress.set_postfix(
                    loss=f"{raw_loss.item():.4f}",
                    avg=f"{total_loss / max(total_examples, 1):.4f}",
                    enc_lr=f"{encoder_lr:.2e}",
                    head_lr=f"{head_lr:.2e}",
                    accum=f"{(step % config.GRADIENT_ACCUMULATION_STEPS) + 1}/"
                    f"{config.GRADIENT_ACCUMULATION_STEPS}",
                    elapsed=f"{elapsed:.0f}s",
                    mem=memory,
                )
    progress.close()

    totals = torch.tensor(
        [total_loss, float(total_examples)], dtype=torch.float64, device=context.device
    )
    if context.distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    mean_loss = float(totals[0].item() / max(totals[1].item(), 1.0))
    return mean_loss, global_step


def write_training_history(history: Sequence[Mapping[str, Any]], config: Config) -> None:
    """Persist one row of real metrics per completed epoch."""

    path = config.resolve(config.OUTPUT_DIR) / "logs" / "training_history.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


# ---------------------------------------------------------------------------
# 10. Submission creation and exact validation
# ---------------------------------------------------------------------------


INTEGER_PATTERN = re.compile(r"^[0-9]+$")


def validate_submission(
    prediction_path: Path,
    zip_path: Path,
    expected_ids: Sequence[str],
    config: Config,
) -> None:
    """Validate exact filename, CSV schema/order/range, and one-entry ZIP layout."""

    if prediction_path.name != config.SUBMISSION_BASENAME:
        raise ValueError(f"Submission file must be named {config.SUBMISSION_BASENAME!r}")
    if zip_path.name != config.SUBMISSION_ZIP_NAME:
        raise ValueError(f"ZIP must be named {config.SUBMISSION_ZIP_NAME!r}")
    with prediction_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] != ["Sentence ID", "Prediction"]:
        raise ValueError("Submission header must be exactly: Sentence ID,Prediction")
    data_rows = rows[1:]
    if len(data_rows) != len(expected_ids):
        raise ValueError(
            f"Submission has {len(data_rows)} rows; expected {len(expected_ids)}"
        )
    actual_ids: list[str] = []
    for row_number, row in enumerate(data_rows, start=2):
        if len(row) != 2:
            raise ValueError(f"Submission row {row_number} has {len(row)} columns")
        row_id, prediction = row
        actual_ids.append(row_id)
        if not INTEGER_PATTERN.fullmatch(prediction):
            raise ValueError(f"Prediction at row {row_number} is not an integer: {prediction!r}")
        value = int(prediction)
        if not config.MIN_LABEL <= value <= config.MAX_LABEL:
            raise ValueError(f"Prediction at row {row_number} is outside label range: {value}")
    expected = [str(row_id) for row_id in expected_ids]
    if actual_ids != expected:
        mismatch = next(
            (
                index
                for index, (actual, wanted) in enumerate(zip(actual_ids, expected), start=2)
                if actual != wanted
            ),
            None,
        )
        raise ValueError(f"Submission ID/order mismatch at CSV row {mismatch}")
    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError("Submission contains duplicate IDs")

    with zipfile.ZipFile(zip_path, "r") as archive:
        names = archive.namelist()
        if names != [config.SUBMISSION_BASENAME]:
            raise ValueError(
                f"ZIP must contain only {config.SUBMISSION_BASENAME!r} at root; got {names}"
            )
        zipped_bytes = archive.read(config.SUBMISSION_BASENAME)
    if zipped_bytes != prediction_path.read_bytes():
        raise ValueError("ZIP entry differs from the validated prediction file")


def run_submission_validator_checks(config: Config) -> None:
    """Confirm that malformed CSV and nested ZIP layouts are rejected."""

    with tempfile.TemporaryDirectory(prefix="barec-submission-check-") as directory:
        root = Path(directory)
        prediction_path = root / config.SUBMISSION_BASENAME
        zip_path = root / config.SUBMISSION_ZIP_NAME
        prediction_path.write_text("Wrong,Prediction\nprobe,1\n", encoding="utf-8")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(prediction_path, arcname=config.SUBMISSION_BASENAME)
        try:
            validate_submission(prediction_path, zip_path, ["probe"], config)
        except ValueError as error:
            if "header" not in str(error).lower():
                raise AssertionError("Malformed-header check failed for the wrong reason") from error
        else:
            raise AssertionError("Submission validator accepted a malformed header")

        prediction_path.write_text(
            "Sentence ID,Prediction\nprobe,1\n", encoding="utf-8", newline="\n"
        )
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(
                prediction_path,
                arcname=f"nested/{config.SUBMISSION_BASENAME}",
            )
        try:
            validate_submission(prediction_path, zip_path, ["probe"], config)
        except ValueError as error:
            if "zip" not in str(error).lower():
                raise AssertionError("Nested-ZIP check failed for the wrong reason") from error
        else:
            raise AssertionError("Submission validator accepted a nested ZIP entry")


def create_submission(
    ids: Sequence[str],
    predictions: Sequence[int],
    config: Config,
) -> tuple[Path, Path]:
    """Write extensionless UTF-8 CSV, ZIP it at root, then validate both."""

    directory = config.resolve(config.SUBMISSION_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    prediction_path = directory / config.SUBMISSION_BASENAME
    zip_path = directory / config.SUBMISSION_ZIP_NAME
    if len(ids) != len(predictions):
        raise ValueError("ID/prediction length mismatch")
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["Sentence ID", "Prediction"])
        writer.writerows(
            (str(row_id), str(int(prediction)))
            for row_id, prediction in zip(ids, predictions)
        )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(prediction_path, arcname=config.SUBMISSION_BASENAME)
    validate_submission(prediction_path, zip_path, ids, config)
    print(f"Submission created successfully:\n{zip_path.resolve()}")
    return prediction_path, zip_path


def write_diagnostics(output: EvaluationOutput, config: Config) -> Path:
    """Write raw/final Test scores separately from the submission."""

    data: dict[str, Any] = {
        "Sentence ID": output.ids,
        "raw_prediction": output.raw_predictions,
        "Prediction": output.final_predictions,
    }
    if output.labels is not None:
        data["gold_label"] = output.labels.astype(np.int64)
    path = (
        config.resolve(config.OUTPUT_DIR)
        / "diagnostics"
        / "test_predictions_with_raw_scores.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path


# ---------------------------------------------------------------------------
# 11. End-to-end workflow
# ---------------------------------------------------------------------------


def prepare_output_directories(config: Config, context: DistributedContext) -> None:
    """Create output directories on rank 0 before shared use."""

    if context.is_main:
        for path in (
            config.resolve(config.OUTPUT_DIR),
            config.resolve(config.CHECKPOINT_DIR),
            config.resolve(config.CACHE_DIR),
            config.resolve(config.SUBMISSION_DIR),
        ):
            path.mkdir(parents=True, exist_ok=True)
    distributed_barrier(context)


def write_preprocessing_report(
    reports: Sequence[Mapping[str, Any]], config: Config
) -> None:
    """Persist cache fingerprints and all counted fallback diagnostics."""

    path = config.resolve(config.OUTPUT_DIR) / "logs" / "preprocessing_report.json"
    atomic_json_dump(
        {
            "preprocessing_version": PREPROCESSING_VERSION,
            "reports": list(reports),
            "total_fallback_count": sum(int(report["fallback_count"]) for report in reports),
        },
        path,
    )


def train_select_and_predict(
    train_frame: pd.DataFrame,
    dev_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    config: Config,
    context: DistributedContext,
    *,
    smoke_test: bool,
) -> None:
    """Fine-tune on Train, select only on Dev, then infer Test once."""

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=True)
    train_loader, dev_loader, test_loader, train_sampler = make_data_loaders(
        train_frame, dev_frame, test_frame, tokenizer, config, context
    )

    base_model = ArabicReadabilityRegressor(
        config.MODEL_NAME,
        config.DROPOUT,
        config.MODEL_OUTPUT_OFFSET,
    )
    base_model.to(context.device)
    run_regression_shape_check(
        base_model,
        tokenizer,
        str(train_frame.iloc[0]["_processed_text"]),
        context.device,
        config.MAX_LENGTH,
    )
    model: nn.Module = base_model
    if context.distributed:
        model = DDP(
            model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            output_device=context.local_rank if context.device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    optimizer = create_optimizer(model, config)
    epoch_loader_steps = len(train_loader)
    if smoke_test:
        epoch_loader_steps = min(epoch_loader_steps, config.SMOKE_MAX_TRAIN_STEPS)
    updates_per_epoch = math.ceil(
        epoch_loader_steps / config.GRADIENT_ACCUMULATION_STEPS
    )
    total_updates = max(1, updates_per_epoch * config.NUM_EPOCHS)
    warmup_steps = int(total_updates * config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    amp_enabled = config.USE_FP16 and context.device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)

    if context.is_main:
        effective_batch = (
            config.PER_DEVICE_BATCH_SIZE
            * context.world_size
            * config.GRADIENT_ACCUMULATION_STEPS
        )
        LOGGER.info("Per-device batch size: %d", config.PER_DEVICE_BATCH_SIZE)
        LOGGER.info("DDP world size / GPUs used: %d", context.world_size)
        LOGGER.info("Gradient accumulation steps: %d", config.GRADIENT_ACCUMULATION_STEPS)
        LOGGER.info("Effective global batch size: %d", effective_batch)
        LOGGER.info("FP16 enabled: %s", amp_enabled)

    start_epoch = 0
    global_step = 0
    best_qwk = -math.inf
    best_mae = math.inf
    bad_epochs = 0
    best_model_path = config.resolve(config.OUTPUT_DIR) / "best_model" / "model_state.pt"
    checkpoint_path = config.resolve(config.CHECKPOINT_DIR) / "last.pt"
    history_path = config.resolve(config.OUTPUT_DIR) / "logs" / "training_history.csv"
    if config.RESUME_FROM_CHECKPOINT:
        resume_path = config.resolve(config.RESUME_FROM_CHECKPOINT)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        if not best_model_path.is_file():
            raise FileNotFoundError(
                "Resume requires the matching best-model state at "
                f"{best_model_path}. Preserve the complete outputs directory, "
                "not only checkpoints/last.pt."
            )
        start_epoch, global_step, best_qwk, best_mae, bad_epochs = resume_training(
            resume_path, model, optimizer, scheduler, scaler, context
        )

    history: list[dict[str, Any]] = []
    if context.is_main and config.RESUME_FROM_CHECKPOINT and history_path.is_file():
        history = pd.read_csv(history_path).to_dict(orient="records")
    has_selected_model = bool(
        config.RESUME_FROM_CHECKPOINT and best_model_path.is_file()
    )
    epoch_range: Iterable[int] = range(start_epoch, config.NUM_EPOCHS)
    if config.RESUME_FROM_CHECKPOINT and bad_epochs >= config.EARLY_STOPPING_PATIENCE:
        if context.is_main:
            LOGGER.info(
                "Resume checkpoint had already reached early stopping; "
                "skipping further training and using its best model."
            )
        epoch_range = ()

    for epoch in epoch_range:
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)  # type: ignore[attr-defined]
        epoch_start = time.perf_counter()
        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            config,
            context,
            epoch,
            global_step,
            max_steps=config.SMOKE_MAX_TRAIN_STEPS if smoke_test else None,
        )
        dev_output = evaluate_model(
            model,
            dev_loader,
            len(dev_frame),
            config,
            context,
            description="Dev",
        )

        decision: Optional[dict[str, Any]] = None
        if context.is_main:
            if dev_output is None or dev_output.metrics is None:
                raise RuntimeError("Dev labels/metrics are required for model selection")
            metrics = dev_output.metrics
            qwk = metrics["qwk"]
            mae = metrics["mae"]
            selection_qwk = qwk if math.isfinite(qwk) else -math.inf
            qwk_tied = (
                abs(selection_qwk - best_qwk) <= 1e-12
                if math.isfinite(selection_qwk) and math.isfinite(best_qwk)
                else selection_qwk == best_qwk
            )
            improved = (
                not has_selected_model
                or selection_qwk > best_qwk + 1e-12
                or (qwk_tied and mae < best_mae)
            )
            if improved:
                best_qwk = selection_qwk
                best_mae = mae
                bad_epochs = 0
                save_best_model(model, tokenizer, config, metrics)
                has_selected_model = True
            else:
                bad_epochs += 1
            elapsed = time.perf_counter() - epoch_start
            history_row = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": train_loss,
                "dev_mse": metrics["mse"],
                "dev_mae": metrics["mae"],
                "dev_qwk": metrics["qwk"],
                "dev_exact_accuracy": metrics["exact_accuracy"],
                "dev_adjacent_accuracy": metrics["adjacent_accuracy"],
                "epoch_seconds": elapsed,
                "is_best": improved,
            }
            history.append(history_row)
            write_training_history(history, config)
            LOGGER.info(
                "Epoch %d | train_loss=%.6f dev_mse=%.6f dev_mae=%.6f "
                "dev_qwk=%s exact=%.4f adjacent=%.4f time=%.1fs best=%s",
                epoch + 1,
                train_loss,
                metrics["mse"],
                metrics["mae"],
                f"{metrics['qwk']:.6f}" if math.isfinite(metrics["qwk"]) else "nan",
                metrics["exact_accuracy"],
                metrics["adjacent_accuracy"],
                elapsed,
                improved,
            )
            decision = {
                "best_qwk": best_qwk,
                "best_mae": best_mae,
                "bad_epochs": bad_epochs,
                "has_selected_model": has_selected_model,
                "stop": bad_epochs >= config.EARLY_STOPPING_PATIENCE,
            }

        decision = broadcast_object(decision, context)
        if not isinstance(decision, dict):
            raise RuntimeError("Failed to broadcast model-selection decision")
        best_qwk = float(decision["best_qwk"])
        best_mae = float(decision["best_mae"])
        bad_epochs = int(decision["bad_epochs"])
        has_selected_model = bool(decision["has_selected_model"])
        rng_states = gather_rng_states(context)
        if context.is_main:
            save_training_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=epoch,
                global_step=global_step,
                best_qwk=best_qwk,
                best_mae=best_mae,
                bad_epochs=bad_epochs,
                config=config,
                rng_states=rng_states,
            )
        distributed_barrier(context)
        if bool(decision["stop"]):
            if context.is_main:
                LOGGER.info("Early stopping after %d non-improving epoch(s).", bad_epochs)
            break

    distributed_barrier(context)
    if not best_model_path.is_file():
        raise RuntimeError(
            "No best model exists. Ensure NUM_EPOCHS permits at least one completed epoch."
        )
    best_state = load_torch_file(best_model_path, context.device)
    unwrap_model(model).load_state_dict(best_state, strict=True)
    distributed_barrier(context)

    test_output = evaluate_model(
        model,
        test_loader,
        len(test_frame),
        config,
        context,
        description="Test inference",
    )
    if context.is_main:
        if test_output is None:
            raise RuntimeError("Rank 0 did not receive Test predictions")
        expected_ids = test_frame.sort_values("_original_index")["_id"].astype(str).tolist()
        if test_output.ids != expected_ids:
            raise RuntimeError("Test predictions are not in the original ID order")
        diagnostics_path = write_diagnostics(test_output, config)
        LOGGER.info("Test diagnostics: %s", diagnostics_path)
        if test_output.metrics is not None:
            LOGGER.info(
                "Open-Test metrics (diagnostic only; never used for selection): %s",
                test_output.metrics,
            )
        create_submission(test_output.ids, test_output.final_predictions, config)
    distributed_barrier(context)


def run_pipeline(config: Config, context: DistributedContext, *, smoke_test: bool) -> None:
    """Validate data, preprocess all splits, and execute the strict workflow."""

    prepare_output_directories(config, context)
    if context.is_main:
        run_submission_validator_checks(config)
    distributed_barrier(context)
    train_path = config.resolve(config.TRAIN_PATH)
    dev_path = config.resolve(config.DEV_PATH)
    test_path = config.resolve(config.TEST_PATH)
    train_frame = load_split(train_path, "train", config, require_label=True)
    dev_frame = load_split(dev_path, "dev", config, require_label=True)
    test_frame = load_split(test_path, "test", config, require_label=False)
    validate_split_isolation(train_frame, dev_frame, test_frame)

    if smoke_test:
        train_frame = smoke_subset(train_frame, config.SMOKE_TRAIN_SAMPLES)
        dev_frame = smoke_subset(dev_frame, config.SMOKE_EVAL_SAMPLES)
        test_frame = smoke_subset(test_frame, config.SMOKE_EVAL_SAMPLES)

    if context.is_main:
        log_split_summary("Train", train_frame, config)
        log_split_summary("Dev", dev_frame, config)
        log_split_summary("Test", test_frame, config)

    train_processed, train_report = preprocess_split_cached(
        train_frame, "train", train_path, config, context
    )
    dev_processed, dev_report = preprocess_split_cached(
        dev_frame, "dev", dev_path, config, context
    )
    test_processed, test_report = preprocess_split_cached(
        test_frame, "test", test_path, config, context
    )
    if context.is_main:
        write_preprocessing_report(
            [train_report, dev_report, test_report], config
        )

    run_sampler_checks()
    run_gather_order_check()
    train_select_and_predict(
        train_processed,
        dev_processed,
        test_processed,
        config,
        context,
        smoke_test=smoke_test,
    )


def validate_config(config: Config) -> None:
    """Fail fast on unsafe or nonsensical centralized settings."""

    if config.MIN_LABEL >= config.MAX_LABEL:
        raise ValueError("MIN_LABEL must be less than MAX_LABEL")
    positive_integer_fields = {
        "MAX_LENGTH": config.MAX_LENGTH,
        "NUM_EPOCHS": config.NUM_EPOCHS,
        "PER_DEVICE_BATCH_SIZE": config.PER_DEVICE_BATCH_SIZE,
        "EVAL_BATCH_SIZE": config.EVAL_BATCH_SIZE,
        "GRADIENT_ACCUMULATION_STEPS": config.GRADIENT_ACCUMULATION_STEPS,
        "EARLY_STOPPING_PATIENCE": config.EARLY_STOPPING_PATIENCE,
        "DDP_TIMEOUT_MINUTES": config.DDP_TIMEOUT_MINUTES,
        "LOG_EVERY_N_STEPS": config.LOG_EVERY_N_STEPS,
        "SMOKE_TRAIN_SAMPLES": config.SMOKE_TRAIN_SAMPLES,
        "SMOKE_EVAL_SAMPLES": config.SMOKE_EVAL_SAMPLES,
        "SMOKE_MAX_TRAIN_STEPS": config.SMOKE_MAX_TRAIN_STEPS,
    }
    invalid = {name: value for name, value in positive_integer_fields.items() if value <= 0}
    if invalid:
        raise ValueError(f"Configuration values must be positive: {invalid}")
    if config.NUM_WORKERS < 0 or config.PREPROCESS_NUM_WORKERS < 1:
        raise ValueError("Worker counts cannot be negative/zero")
    if not 0.0 <= config.WARMUP_RATIO < 1.0:
        raise ValueError("WARMUP_RATIO must be in [0, 1)")
    if not 0.0 <= config.DROPOUT < 1.0:
        raise ValueError("DROPOUT must be in [0, 1)")
    if not math.isfinite(config.MODEL_OUTPUT_OFFSET):
        raise ValueError("MODEL_OUTPUT_OFFSET must be finite")
    if config.ENCODER_LR <= 0.0 or config.HEAD_LR <= 0.0:
        raise ValueError("ENCODER_LR and HEAD_LR must be positive")
    if config.WEIGHT_DECAY < 0.0 or config.MAX_GRAD_NORM <= 0.0:
        raise ValueError("WEIGHT_DECAY must be non-negative and MAX_GRAD_NORM positive")
    if config.SAMPLER_ALPHA < 0.0:
        raise ValueError("SAMPLER_ALPHA cannot be negative")


def main() -> None:
    """Entrypoint for automatic DDP launch and the complete baseline."""

    args = parse_args()
    if args.local_rank_argument is not None:
        os.environ.setdefault("LOCAL_RANK", str(args.local_rank_argument))
    if maybe_self_launch_ddp(args):
        return
    config = Config()
    if args.smoke_test:
        config.enable_smoke_mode()
    validate_config(config)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    context = initialize_distributed(config)
    seed_everything(config.SEED + context.rank)
    try:
        run_pipeline(config, context, smoke_test=args.smoke_test)
    finally:
        if context.distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
