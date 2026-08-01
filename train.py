#!/usr/bin/env python
"""BAREC 2026 sentence-level Strict Track cascaded multi-task ensemble.

The complete workflow lives in this file: validation, Arabic D3Tok
preprocessing, distributed fine-tuning, per-seed model selection, raw-score
averaging, inference, and submission validation. Edit :class:`Config` and run
``python train.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
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
from dataclasses import asdict, dataclass, replace
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
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


# ---------------------------------------------------------------------------
# 1. Centralized configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger("barec")
PREPROCESSING_VERSION = "barec-structured-pair-v4"
CHECKPOINT_SELECTION_POLICY = "dev_floor_qwk_then_floor_mae_v1"
FINAL_DISCRETIZATION_POLICY = "floor_after_uniform_raw_mean_v1"
TATWEEL = "\u0640"
ARABIC_DIACRITICS = frozenset(
    chr(codepoint)
    for start, end in (
        (0x0610, 0x061A),
        (0x064B, 0x065F),
        (0x0670, 0x0670),
        (0x06D6, 0x06ED),
        (0x08D3, 0x08FF),
    )
    for codepoint in range(start, end + 1)
    if unicodedata.category(chr(codepoint)).startswith("M")
)
FIELD_TOKENS = (
    "[WC]",
    "[DC]",
    "[WLA]",
    "[WLS]",
    "[DOM_AH]",
    "[DOM_SS]",
    "[DOM_STEM]",
    "[DOM_UNKNOWN]",
    "[TC_FOUNDATIONAL]",
    "[TC_ADVANCED]",
    "[TC_SPECIALIZED]",
    "[TC_UNKNOWN]",
)


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
    SEED_RUNS_DIR: str = "outputs/seeds"

    # Columns.
    ID_COLUMN: str = "ID"
    TEXT_COLUMN: str = "Sentence"
    LABEL_COLUMN: str = "Readability_Level_19"
    LABEL_3_COLUMN: str = "Readability_Level_3"
    LABEL_5_COLUMN: str = "Readability_Level_5"
    LABEL_7_COLUMN: str = "Readability_Level_7"

    # Model and Arabic preprocessing.
    MODEL_NAME: str = "CAMeL-Lab/readability-arabertv2-d3tok-CE"
    MODEL_MODE: str = "cascaded_hmtl"
    MAX_LENGTH: int = 512
    DROPOUT: float = 0.1
    AUX_CE3_WEIGHT: float = 0.5
    AUX_CE5_WEIGHT: float = 0.5
    AUX_CE7_WEIGHT: float = 0.5
    CASCADE_TEMPERATURE: float = 1.0
    D3TOK_RESOURCE: str = "msa"
    AUTO_DOWNLOAD_CAMEL_DATA: bool = True
    CAMEL_DATA_PACKAGE: str = "disambig-bert-unfactored-msa"
    FORCE_REPROCESS: bool = False
    PREPROCESS_NUM_WORKERS: int = 1
    D3TOK_BATCH_SIZE: int = 256

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
    ENSEMBLE_SEEDS: tuple[int, ...] = (42, 52, 62, 72, 82)
    # Active seed for one ensemble member. The top-level pipeline overwrites
    # this field with each value from ENSEMBLE_SEEDS.
    SEED: int = 42
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True
    USE_FP16: bool = True
    USE_WEIGHTED_SAMPLER: bool = True
    SAMPLER_ALPHA: float = 0.5
    SAMPLER_REPLACEMENT: bool = True
    DDP_TIMEOUT_MINUTES: int = 180
    LOG_EVERY_N_STEPS: int = 50
    RESUME_FROM_CHECKPOINT: Optional[str] = None

    # Labels and submission.
    MIN_LABEL: int = 1
    MAX_LABEL: int = 19
    # Every submitted ZIP must contain this exact extensionless root member.
    SUBMISSION_BASENAME: str = "prediction"
    SUBMISSION_DOWN_BASENAME: str = "prediction_down"
    SUBMISSION_DOWN_ZIP_NAME: str = "prediction_down.zip"

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
        self.SEED_RUNS_DIR = "outputs/smoke/seeds"
        self.CACHE_DIR = "cache/smoke"
        # Two members are sufficient to exercise aggregation while keeping the
        # real-model smoke test reasonably short.
        self.ENSEMBLE_SEEDS = (42, 52)
        self.SEED = self.ENSEMBLE_SEEDS[0]
        self.NUM_EPOCHS = 1
        self.PER_DEVICE_BATCH_SIZE = 2
        self.EVAL_BATCH_SIZE = 2
        self.GRADIENT_ACCUMULATION_STEPS = 1
        # Keep the production sequence budget so smoke mode exercises the same
        # no-D3Tok-truncation invariant.
        self.MAX_LENGTH = 512
        self.NUM_WORKERS = 0
        self.PREPROCESS_NUM_WORKERS = 1
        self.EARLY_STOPPING_PATIENCE = 1
        self.LOG_EVERY_N_STEPS = 1
        self.RESUME_FROM_CHECKPOINT = None


def config_for_seed(config: Config, seed: int) -> Config:
    """Create isolated artifact paths for one deterministic ensemble member."""

    member = replace(config)
    member.SEED = int(seed)
    member_root = config.resolve(config.SEED_RUNS_DIR) / f"seed_{seed}"
    member.OUTPUT_DIR = str(member_root)
    member.CHECKPOINT_DIR = str(member_root / "checkpoints")
    member.SUBMISSION_DIR = str(member_root)
    if config.RESUME_FROM_CHECKPOINT:
        resume_template = str(config.RESUME_FROM_CHECKPOINT)
        member.RESUME_FROM_CHECKPOINT = (
            resume_template.format(seed=seed)
            if "{seed}" in resume_template
            else resume_template
        )
    return member


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
    child_environment = os.environ.copy()
    # Rank 0 can legitimately spend several minutes building the first D3Tok
    # cache while rank 1 waits at a collective. PyTorch's default NCCL monitor
    # may otherwise abort the healthy worker after roughly eight minutes.
    child_environment.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600")
    subprocess.run(
        command,
        check=True,
        cwd=str(SCRIPT_DIR),
        env=child_environment,
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
LABEL_3_ALIASES = ("Readability_Level_3", "label3", "label_3")
LABEL_5_ALIASES = ("Readability_Level_5", "label5", "label_5")
LABEL_7_ALIASES = ("Readability_Level_7", "label7", "label_7")
DOCUMENT_ALIASES = ("Document", "document")
DOMAIN_ALIASES = ("Domain", "domain")
TEXT_CLASS_ALIASES = ("Text_Class", "Text Class", "text_class", "text class")

# Index 0 is deliberately unused so a legal 19-level label can index these
# tables directly.  These are the official BAREC nested label mappings.
LABEL3_FROM_19 = np.asarray(
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3],
    dtype=np.int64,
)
LABEL5_FROM_19 = np.asarray(
    [0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5],
    dtype=np.int64,
)
LABEL7_FROM_19 = np.asarray(
    [0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7],
    dtype=np.int64,
)


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


def validated_integer_labels(
    source: pd.Series,
    split_name: str,
    role: str,
    minimum: int,
    maximum: int,
) -> np.ndarray:
    """Parse one complete bounded integer-label column with useful errors."""

    if source.isna().any():
        rows = source.index[source.isna()].tolist()[:10]
        raise ValueError(f"{split_name}: missing {role} labels at rows {rows}")
    numeric = pd.to_numeric(source, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    # Labels are categorical IDs, so even a small fractional component is an
    # error rather than a value to tolerate as approximate floating point.
    invalid = ~np.isfinite(values) | (values != np.rint(values))
    if invalid.any():
        rows = np.flatnonzero(invalid)[:10].tolist()
        raise ValueError(f"{split_name}: non-integer {role} labels at rows {rows}")
    labels = np.rint(values).astype(np.int64)
    outside = (labels < minimum) | (labels > maximum)
    if outside.any():
        bad = sorted(set(labels[outside].tolist()))
        raise ValueError(
            f"{split_name}: {role} labels outside [{minimum}, {maximum}]: {bad}"
        )
    return labels


def auxiliary_labels_from_19(labels: Sequence[int]) -> dict[int, np.ndarray]:
    """Return official 3/5/7-level labels for legal 19-level labels."""

    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or ((values < 1) | (values > 19)).any():
        raise ValueError("19-level labels must be a 1D array within [1, 19]")
    return {
        3: LABEL3_FROM_19[values],
        5: LABEL5_FROM_19[values],
        7: LABEL7_FROM_19[values],
    }


def run_auxiliary_mapping_check() -> None:
    """Verify all 19 official mappings and zero-based CE ranges."""

    labels19 = np.arange(1, 20, dtype=np.int64)
    mapped = auxiliary_labels_from_19(labels19)
    expected = {
        3: [1] * 11 + [2] * 2 + [3] * 6,
        5: [1] * 7 + [2] * 4 + [3] * 2 + [4] * 2 + [5] * 4,
        7: [1] * 4 + [2] * 3 + [3] * 2 + [4] * 2 + [5] * 2 + [6] * 2 + [7] * 4,
    }
    for level_count, wanted in expected.items():
        if mapped[level_count].tolist() != wanted:
            raise AssertionError(f"Official {level_count}-level mapping check failed")
        zero_based = mapped[level_count] - 1
        if int(zero_based.min()) != 0 or int(zero_based.max()) != level_count - 1:
            raise AssertionError(
                f"Zero-based {level_count}-level CE range check failed"
            )


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
    require_auxiliary = require_label and config.MODEL_MODE == "cascaded_hmtl"
    label_3_column = resolve_column(
        columns,
        config.LABEL_3_COLUMN,
        LABEL_3_ALIASES,
        "3-level label",
        required=require_auxiliary,
    )
    label_5_column = resolve_column(
        columns,
        config.LABEL_5_COLUMN,
        LABEL_5_ALIASES,
        "5-level label",
        required=require_auxiliary,
    )
    label_7_column = resolve_column(
        columns,
        config.LABEL_7_COLUMN,
        LABEL_7_ALIASES,
        "7-level label",
        required=require_auxiliary,
    )
    document_column = resolve_column(
        columns, "Document", DOCUMENT_ALIASES, "document", required=False
    )
    domain_column = resolve_column(
        columns, "Domain", DOMAIN_ALIASES, "domain", required=False
    )
    text_class_column = resolve_column(
        columns, "Text_Class", TEXT_CLASS_ALIASES, "text class", required=False
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
    frame["_label3"] = -1
    frame["_label5"] = -1
    frame["_label7"] = -1
    if label_column is not None:
        label_source = frame[label_column]
        if not require_label and label_source.isna().all():
            label_column = None
        else:
            labels = validated_integer_labels(
                label_source,
                split_name,
                "19-level",
                config.MIN_LABEL,
                config.MAX_LABEL,
            )
            frame["_label"] = labels
            expected_auxiliary = auxiliary_labels_from_19(labels)
            auxiliary_columns = {
                3: label_3_column,
                5: label_5_column,
                7: label_7_column,
            }
            for level_count, auxiliary_column in auxiliary_columns.items():
                actual = expected_auxiliary[level_count]
                if auxiliary_column is not None:
                    actual = validated_integer_labels(
                        frame[auxiliary_column],
                        split_name,
                        f"{level_count}-level",
                        1,
                        level_count,
                    )
                    mismatch = actual != expected_auxiliary[level_count]
                    if mismatch.any():
                        rows = np.flatnonzero(mismatch)[:10].tolist()
                        raise ValueError(
                            f"{split_name}: {level_count}-level labels do not "
                            f"match the official 19→{level_count} mapping at rows {rows}"
                        )
                # CrossEntropyLoss expects zero-based class indices.
                frame[f"_label{level_count}"] = actual - 1

    if document_column is not None:
        frame["_document"] = frame[document_column].map(
            lambda value: None if pd.isna(value) else str(value)
        )
    else:
        frame["_document"] = None
    frame["_domain"] = (
        frame[domain_column].map(lambda value: None if pd.isna(value) else str(value))
        if domain_column is not None
        else None
    )
    frame["_text_class"] = (
        frame[text_class_column].map(
            lambda value: None if pd.isna(value) else str(value)
        )
        if text_class_column is not None
        else None
    )
    frame["_original_index"] = np.arange(len(frame), dtype=np.int64)
    frame.attrs["has_labels"] = label_column is not None
    frame.attrs["has_auxiliary_labels"] = label_column is not None
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
        if config.MODEL_MODE == "cascaded_hmtl":
            for level_count in (3, 5, 7):
                auxiliary_counts = (
                    frame[f"_label{level_count}"].astype(int) + 1
                ).value_counts().sort_index()
                LOGGER.info(
                    "%s %d-level distribution: %s",
                    name,
                    level_count,
                    {
                        label: int(auxiliary_counts.get(label, 0))
                        for label in range(1, level_count + 1)
                    },
                )


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
    """Parallel text views, sentence statistics, and an optional diagnostic."""

    d3tok_text: str
    surface_text: str
    diacritic_coverage: float
    word_count: int
    word_length_mean: float
    word_length_std: float
    error: Optional[str]


class ArabicD3TokPreprocessor:
    """Keep diacritics for BERT D3Tok, then build normalized model views."""

    def __init__(self, resource: str, *, use_gpu: Optional[bool] = None) -> None:
        try:
            from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
            from camel_tools.tokenizers.word import simple_word_tokenize
            from camel_tools.utils.dediac import dediac_ar
            from camel_tools.utils.normalize import normalize_unicode
        except ImportError as error:
            raise RuntimeError(
                "CAMeL Tools with the BERT unfactored disambiguator is required "
                "for official D3Tok. Install requirements.txt first."
            ) from error

        self._simple_word_tokenize = simple_word_tokenize
        self._dediac_ar = dediac_ar
        self._normalize_unicode = normalize_unicode
        use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu
        self._disambiguator = BERTUnfactoredDisambiguator.pretrained(
            model_name=resource,
            pretrained_cache=False,
            top=1,
            use_gpu=use_gpu,
        )

    def normalize_keep_diacritics(self, text: str) -> str:
        """Normalize Unicode and remove Kashida without deleting diacritics."""

        normalized = self._normalize_unicode(text, compatibility=True)
        normalized = normalized.replace(TATWEEL, "")
        return re.sub(r"(?<=\B)ى(?=\B)", "ي", normalized)

    def surface_view(self, normalized_text: str) -> str:
        """Remove Arabic diacritics while preserving words and punctuation."""

        return self._dediac_ar(normalized_text).replace(TATWEEL, "")

    def sentence_statistics(
        self,
        normalized_text: str,
        surface_text: str,
    ) -> tuple[float, int, float, float]:
        """Compute DC before dediacritization and length stats on Surface view."""

        diacritic_count = sum(char in ARABIC_DIACRITICS for char in normalized_text)
        diacritic_coverage = (
            0.0
            if not normalized_text
            else float(diacritic_count / len(normalized_text))
        )
        surface_tokens = self._simple_word_tokenize(
            surface_text,
            split_digits=True,
        )
        lengths = np.asarray(
            [len(token) for token in surface_tokens],
            dtype=np.float32,
        )
        if not len(lengths):
            return diacritic_coverage, 0, 0.0, 0.0
        return (
            diacritic_coverage,
            int(len(lengths)),
            float(lengths.mean()),
            float(lengths.std()),
        )

    @staticmethod
    def _is_diacritic_only(token: str) -> bool:
        """Return whether a tokenizer item contains Arabic marks and no base."""

        return bool(token) and all(char in ARABIC_DIACRITICS for char in token)

    @staticmethod
    def _contains_arabic_letter(token: str) -> bool:
        """Return whether a token contains an Arabic-script letter."""

        return any(
            (
                "\u0600" <= char <= "\u06ff"
                or "\u0750" <= char <= "\u077f"
                or "\u08a0" <= char <= "\u08ff"
            )
            and unicodedata.category(char).startswith("L")
            for char in token
        )

    def tokenize_for_d3tok(self, normalized_text: str) -> list[str]:
        """Repair detached marks and remove mark-only items before BERT."""

        raw_tokens = self._simple_word_tokenize(
            normalized_text,
            split_digits=True,
        )
        repaired_tokens: list[str] = []
        for token in raw_tokens:
            if self._is_diacritic_only(token):
                if (
                    repaired_tokens
                    and self._contains_arabic_letter(repaired_tokens[-1])
                ):
                    repaired_tokens[-1] += token
                # A leading mark or a mark after punctuation has no lexical
                # base for morphological analysis. DC was already computed on
                # the untouched normalized sentence, so dropping it here does
                # not alter the orthographic statistic.
                continue
            repaired_tokens.append(token)
        if not repaired_tokens:
            raise ValueError("D3Tok tokenization produced no lexical content")
        if any(self._is_diacritic_only(token) for token in repaired_tokens):
            raise AssertionError("A detached Arabic diacritic reached BERT D3Tok")
        return repaired_tokens

    def _render_d3tok_sentence(
        self,
        sentence_analysis: Sequence[Any],
        input_tokens: Sequence[str],
    ) -> tuple[str, Optional[str]]:
        """Render D3Tok, falling back only for individual unanalyzable tokens."""

        rendered_words: list[str] = []
        fallback_positions: list[int] = []
        if len(sentence_analysis) != len(input_tokens):
            raise RuntimeError(
                "BERT D3Tok token/analysis length mismatch: "
                f"tokens={len(input_tokens)}, analyses={len(sentence_analysis)}"
            )
        for position, (item, input_token) in enumerate(
            zip(sentence_analysis, input_tokens)
        ):
            analysis: Mapping[str, Any] = {}
            if item.analyses:
                scored_analysis = item.analyses[0]
                analysis = (
                    scored_analysis.analysis
                    if hasattr(scored_analysis, "analysis")
                    else scored_analysis[1]
                )
            d3tok = analysis.get("d3tok")
            if d3tok is None:
                fallback_token = self._dediac_ar(input_token).replace(TATWEEL, "")
                if not fallback_token.strip():
                    raise ValueError(
                        "BERT D3Tok token fallback produced no content at "
                        f"position {position}"
                    )
                rendered_words.append(fallback_token)
                fallback_positions.append(position)
                continue
            rendered_words.append(
                self._dediac_ar(str(d3tok))
                .replace("_+", " +")
                .replace("+_", "+ ")
            )
        processed = " ".join(rendered_words)
        if not processed.strip():
            raise ValueError("BERT D3Tok returned no content")
        diagnostic = (
            None
            if not fallback_positions
            else "TokenD3TokFallback: positions="
            + ",".join(str(position) for position in fallback_positions)
        )
        return processed, diagnostic

    def process_many(self, texts: Sequence[str]) -> list[PreprocessResult]:
        """Process a batch through BERT, isolating any per-row failures."""

        results: list[Optional[PreprocessResult]] = [None] * len(texts)
        valid_indices: list[int] = []
        surface_texts: list[str] = [""] * len(texts)
        statistics: list[tuple[float, int, float, float]] = [
            (0.0, 0, 0.0, 0.0)
        ] * len(texts)
        tokenized_sentences: list[list[str]] = []

        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError("D3Tok received an empty/non-string sentence")
            try:
                normalized = self.normalize_keep_diacritics(text)
            except Exception as error:
                normalized = unicodedata.normalize("NFKC", text).replace(TATWEEL, "")
                surface = self.surface_view(normalized)
                dc, wc, wla, wls = self.sentence_statistics(normalized, surface)
                results[index] = PreprocessResult(
                    surface,
                    surface,
                    dc,
                    wc,
                    wla,
                    wls,
                    f"UnicodeNormalizationError: {type(error).__name__}: {error}",
                )
                continue
            surface = self.surface_view(normalized)
            if not surface.strip():
                raise ValueError("Surface preprocessing produced no content")
            surface_texts[index] = surface
            statistics[index] = self.sentence_statistics(normalized, surface)
            valid_indices.append(index)
            tokenized_sentences.append(self.tokenize_for_d3tok(normalized))

        if tokenized_sentences:
            try:
                analyzed_sentences = self._disambiguator.disambiguate_sentences(
                    tokenized_sentences
                )
                if len(analyzed_sentences) != len(tokenized_sentences):
                    raise RuntimeError(
                        "BERT D3Tok returned a different number of sentences"
                    )
                for index, sentence_analysis, words in zip(
                    valid_indices,
                    analyzed_sentences,
                    tokenized_sentences,
                ):
                    dc, wc, wla, wls = statistics[index]
                    d3tok_text, diagnostic = self._render_d3tok_sentence(
                        sentence_analysis,
                        words,
                    )
                    results[index] = PreprocessResult(
                        d3tok_text,
                        surface_texts[index],
                        dc,
                        wc,
                        wla,
                        wls,
                        diagnostic,
                    )
            except Exception as batch_error:
                # A malformed row should not force every row in a BERT batch to
                # fall back. Retry individually so diagnostics remain exact.
                for index, words in zip(valid_indices, tokenized_sentences):
                    try:
                        sentence_analysis = self._disambiguator.disambiguate_sentences(
                            [words]
                        )[0]
                        dc, wc, wla, wls = statistics[index]
                        d3tok_text, diagnostic = self._render_d3tok_sentence(
                            sentence_analysis,
                            words,
                        )
                        results[index] = PreprocessResult(
                            d3tok_text,
                            surface_texts[index],
                            dc,
                            wc,
                            wla,
                            wls,
                            diagnostic,
                        )
                    except Exception as row_error:
                        dc, wc, wla, wls = statistics[index]
                        results[index] = PreprocessResult(
                            surface_texts[index],
                            surface_texts[index],
                            dc,
                            wc,
                            wla,
                            wls,
                            "D3TokError: "
                            f"{type(row_error).__name__}: {row_error}; "
                            "batch_error="
                            f"{type(batch_error).__name__}: {batch_error}",
                        )

        if any(result is None for result in results):
            raise RuntimeError("BERT D3Tok preprocessing left an unresolved row")
        return [result for result in results if result is not None]

    def process(self, text: str) -> PreprocessResult:
        """Preprocess one sentence using the same path as batched inference."""

        return self.process_many([text])[0]


def camel_data_install_command(package: str) -> list[str]:
    """Return the most reliable CAMeL data installer command available."""

    executable = shutil.which("camel_data")
    if executable:
        return [executable, "-i", package]
    return [sys.executable, "-m", "camel_tools.cli.camel_data", "-i", package]


def create_preprocessor(config: Config, *, allow_download: bool) -> ArabicD3TokPreprocessor:
    """Load BERT D3Tok and optionally install its CAMeL data package once."""

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
PreprocessedRow = tuple[
    str,
    str,
    str,
    float,
    int,
    float,
    float,
    Optional[str],
]


def _preprocess_worker_init(resource: str) -> None:
    global _PROCESS_PREPROCESSOR
    # Multiple spawned workers must not each allocate a full BERT tagger on
    # the same GPU. The default single-process path still uses CUDA when present.
    _PROCESS_PREPROCESSOR = ArabicD3TokPreprocessor(resource, use_gpu=False)


def _preprocess_worker_batch(
    items: Sequence[tuple[str, str]],
) -> list[PreprocessedRow]:
    if _PROCESS_PREPROCESSOR is None:
        raise RuntimeError("Preprocessing worker was not initialized")
    results = _PROCESS_PREPROCESSOR.process_many([text for _, text in items])
    return [
        (
            row_id,
            result.d3tok_text,
            result.surface_text,
            result.diacritic_coverage,
            result.word_count,
            result.word_length_mean,
            result.word_length_std,
            result.error,
        )
        for (row_id, _), result in zip(items, results)
    ]


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
        "normalization": "CAMeL normalize_unicode compatibility=True",
        "kashida": "remove U+0640 while retaining Arabic diacritics",
        "disambiguator": "BERTUnfactoredDisambiguator top=1 sees diacritics",
        "detached_diacritics": (
            "reattach mark-only tokens to the preceding Arabic word and drop "
            "residual mark-only tokens before BERT"
        ),
        "segmentation": "analysis['d3tok'] with _+ and +_ boundary conversion",
        "token_fallback": "missing d3tok falls back per-token, not per-sentence",
        "dediac": "CAMeL dediac_ar after BERT D3Tok and on Surface view",
        "statistics": "DC before dediac; WC/WLA/WLS on Surface tokens",
        "punctuation": "preserved in D3Tok and Surface views",
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
    """Exercise the fixed diacritics-aware preprocessing order and fallback."""

    sample = "الكتــــابُ مفيدٌ."
    normalized = preprocessor.normalize_keep_diacritics(sample)
    if TATWEEL in normalized:
        raise AssertionError("Kashida removal check failed")
    if not any(char in ARABIC_DIACRITICS for char in normalized):
        raise AssertionError("Diacritics must remain available to BERT D3Tok")
    detached_probe = preprocessor.tokenize_for_d3tok("الله ُ، ِ عَلَى")
    if not any(
        token.startswith("الله") and "\u064f" in token
        for token in detached_probe
    ):
        raise AssertionError("Detached diacritic was not reattached to its word")
    detached_mark_probes = (
        detached_probe,
        preprocessor.tokenize_for_d3tok("ِ عَلَى قُوَّةِ المُؤْمِنِ."),
        preprocessor.tokenize_for_d3tok("(حَدِيثَة)ً تَحْتَ الصّورَةِ"),
        preprocessor.tokenize_for_d3tok("كلمة ٔ أخرى"),
        preprocessor.tokenize_for_d3tok("لفظ ۤ آخر"),
    )
    if any(
        preprocessor._is_diacritic_only(token)
        for tokens in detached_mark_probes
        for token in tokens
    ):
        raise AssertionError("A mark-only token survived D3Tok preparation")
    result = preprocessor.process(sample)
    if not result.d3tok_text.strip() or not result.surface_text.strip():
        raise AssertionError("D3Tok content-preservation check failed")
    if TATWEEL in result.d3tok_text or any(
        char in ARABIC_DIACRITICS for char in result.d3tok_text
    ):
        raise AssertionError("Post-D3Tok Tatweel/diacritic check failed")
    if any(char in ARABIC_DIACRITICS for char in result.surface_text):
        raise AssertionError("Surface view must be dediacritized")
    if "." not in result.d3tok_text or "." not in result.surface_text:
        raise AssertionError("Punctuation must be preserved in both text views")
    plus_probe = preprocessor.surface_view("ال+كِتَابُ")
    if "+" not in plus_probe or any(char in ARABIC_DIACRITICS for char in plus_probe):
        raise AssertionError("dediac_ar must retain D3Tok's '+' marker")
    fallback_probe = preprocessor.surface_view(normalized)
    if not fallback_probe.strip() or TATWEEL in fallback_probe:
        raise AssertionError("Fallback content-preservation check failed")
    if result.word_count <= 0 or result.word_length_mean <= 0.0:
        raise AssertionError("Surface statistics must be positive for the probe")
    expected_dc = sum(char in ARABIC_DIACRITICS for char in normalized) / len(
        normalized
    )
    if not math.isclose(result.diacritic_coverage, expected_dc):
        raise AssertionError("Diacritic coverage was not computed before dediac")

    class _MissingD3TokItem:
        analyses = [
            type("_ScoredAnalysis", (), {"analysis": {"diac": "كِتاب"}})()
        ]

    token_fallback_text, token_fallback_error = preprocessor._render_d3tok_sentence(
        [_MissingD3TokItem()],
        ["كِتاب"],
    )
    if token_fallback_text != "كتاب" or not (
        token_fallback_error
        and token_fallback_error.startswith("TokenD3TokFallback:")
    ):
        raise AssertionError("Per-token missing-d3tok fallback check failed")

    class _ForcedBERTD3TokFailure:
        def disambiguate_sentences(
            self, sentences: Sequence[Sequence[str]]
        ) -> list[Any]:
            del sentences
            raise RuntimeError("forced internal fallback check")

    real_disambiguator = preprocessor._disambiguator
    try:
        preprocessor._disambiguator = _ForcedBERTD3TokFailure()
        forced_fallback = preprocessor.process(sample)
    finally:
        preprocessor._disambiguator = real_disambiguator
    if not forced_fallback.error or not forced_fallback.error.startswith("D3TokError:"):
        raise AssertionError("Forced D3Tok failure was not diagnosed")
    if not forced_fallback.d3tok_text.strip() or any(
        char in ARABIC_DIACRITICS for char in forced_fallback.d3tok_text
    ):
        raise AssertionError("Forced D3Tok fallback did not preserve clean content")


def d3tok_progress_options(cache_path: Path, total_batches: int) -> dict[str, Any]:
    """Return a compact progress label that leaves room for Kaggle counters."""

    if total_batches <= 0:
        raise ValueError("D3Tok progress requires at least one batch")
    split_name = cache_path.stem.split("-", 1)[0].strip() or "split"
    return {
        "total": total_batches,
        "desc": f"BERT D3Tok {split_name}",
        "unit": "batch",
        "dynamic_ncols": True,
    }


def build_preprocessing_cache(
    frame: pd.DataFrame,
    cache_path: Path,
    config: Config,
) -> None:
    """Build a Parquet D3Tok cache atomically on the main process."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    items = list(zip(frame["_id"].astype(str), frame["_text"].astype(str)))
    item_batches = [
        items[start : start + config.D3TOK_BATCH_SIZE]
        for start in range(0, len(items), config.D3TOK_BATCH_SIZE)
    ]
    progress_options = d3tok_progress_options(cache_path, len(item_batches))
    LOGGER.info(
        "%s preprocessing: %d sentences in %d batches (up to %d sentences/batch).",
        progress_options["desc"],
        len(items),
        len(item_batches),
        config.D3TOK_BATCH_SIZE,
    )
    rows: list[PreprocessedRow]

    if config.PREPROCESS_NUM_WORKERS > 1:
        probe = create_preprocessor(config, allow_download=True)
        run_arabic_preprocessing_checks(probe)
        del probe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with ProcessPoolExecutor(
            max_workers=config.PREPROCESS_NUM_WORKERS,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_preprocess_worker_init,
            initargs=(config.D3TOK_RESOURCE,),
        ) as executor:
            rows = [
                row
                for batch_rows in tqdm(
                    executor.map(_preprocess_worker_batch, item_batches, chunksize=1),
                    **progress_options,
                )
                for row in batch_rows
            ]
    else:
        preprocessor = create_preprocessor(config, allow_download=True)
        run_arabic_preprocessing_checks(preprocessor)
        rows = []
        for item_batch in tqdm(
            item_batches,
            **progress_options,
        ):
            results = preprocessor.process_many([text for _, text in item_batch])
            rows.extend(
                (
                    row_id,
                    result.d3tok_text,
                    result.surface_text,
                    result.diacritic_coverage,
                    result.word_count,
                    result.word_length_mean,
                    result.word_length_std,
                    result.error,
                )
                for (row_id, _), result in zip(item_batch, results)
            )
        del preprocessor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cache_frame = pd.DataFrame(
        {
            "_id": [row[0] for row in rows],
            "_original_index": frame["_original_index"].to_numpy(dtype=np.int64),
            "_d3tok_text": [row[1] for row in rows],
            "_surface_text": [row[2] for row in rows],
            "_diacritic_coverage": [row[3] for row in rows],
            "_word_count": [row[4] for row in rows],
            "_word_length_mean": [row[5] for row in rows],
            "_word_length_std": [row[6] for row in rows],
            "_fallback_error": [row[7] for row in rows],
        }
    )
    for text_column in ("_d3tok_text", "_surface_text"):
        if (
            cache_frame[text_column].isna().any()
            or cache_frame[text_column].astype(str).str.strip().eq("").any()
        ):
            raise RuntimeError(
                f"Preprocessing produced an empty cached field: {text_column}"
            )
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
    required_cache_columns = (
        "_d3tok_text",
        "_surface_text",
        "_diacritic_coverage",
        "_word_count",
        "_word_length_mean",
        "_word_length_std",
        "_fallback_error",
    )
    missing_cache_columns = [
        column for column in required_cache_columns if column not in cache_frame
    ]
    if missing_cache_columns:
        raise RuntimeError(
            f"Cache schema is missing {missing_cache_columns}: {cache_path}"
        )
    output = frame.copy()
    output["_d3tok_text"] = cache_frame["_d3tok_text"].astype(str).to_numpy()
    output["_surface_text"] = cache_frame["_surface_text"].astype(str).to_numpy()
    output["_diacritic_coverage"] = pd.to_numeric(
        cache_frame["_diacritic_coverage"],
        errors="raise",
    ).to_numpy(dtype=np.float64)
    output["_word_count"] = pd.to_numeric(
        cache_frame["_word_count"],
        errors="raise",
    ).to_numpy(dtype=np.int64)
    output["_word_length_mean"] = pd.to_numeric(
        cache_frame["_word_length_mean"],
        errors="raise",
    ).to_numpy(dtype=np.float64)
    output["_word_length_std"] = pd.to_numeric(
        cache_frame["_word_length_std"],
        errors="raise",
    ).to_numpy(dtype=np.float64)
    numeric_feature_columns = (
        "_diacritic_coverage",
        "_word_length_mean",
        "_word_length_std",
    )
    for column in numeric_feature_columns:
        values = output[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Cache contains non-finite values in {column}")
    if (output["_word_count"].to_numpy(dtype=np.int64) <= 0).any():
        raise RuntimeError("Cache contains non-positive Surface word counts")
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


def normalized_metadata_key(value: Any) -> str:
    """Normalize a small categorical metadata value without leaking row IDs."""

    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().casefold()).replace("&", "and")


def domain_token(value: Any) -> str:
    """Map the three stable BAREC domains to compact learned tokens."""

    key = normalized_metadata_key(value)
    mapping = {
        "arts and humanities": "[DOM_AH]",
        "social sciences": "[DOM_SS]",
        "stem": "[DOM_STEM]",
    }
    return mapping.get(key, "[DOM_UNKNOWN]")


def text_class_token(value: Any) -> str:
    """Map BAREC readership groups to compact learned tokens."""

    key = normalized_metadata_key(value)
    mapping = {
        "foundational": "[TC_FOUNDATIONAL]",
        "advanced": "[TC_ADVANCED]",
        "specialized": "[TC_SPECIALIZED]",
    }
    return mapping.get(key, "[TC_UNKNOWN]")


def structured_feature_groups(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Serialize features as atomic groups ordered from highest to lowest priority."""

    return (
        f"[WC] {int(row['_word_count'])}",
        f"[DC] {float(row['_diacritic_coverage']):.3f}",
        f"[WLA] {float(row['_word_length_mean']):.3f}",
        f"[WLS] {float(row['_word_length_std']):.3f}",
        domain_token(row.get("_domain")),
        text_class_token(row.get("_text_class")),
    )


def structured_feature_text(row: Mapping[str, Any]) -> str:
    """Return the complete feature block for logging and diagnostics."""

    return " ".join(structured_feature_groups(row))


def validate_structured_tokenizer(tokenizer: Any) -> None:
    """Ensure every field marker is atomic and the pair separator is available."""

    if tokenizer.sep_token is None or tokenizer.sep_token_id is None:
        raise RuntimeError("The tokenizer must define a [SEP] token for paired input")
    for token in FIELD_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if (
            token_id is None
            or token_id == tokenizer.unk_token_id
            or encoded != [token_id]
        ):
            raise RuntimeError(
                f"Structured field token is not encoded atomically: {token}"
            )


def encode_structured_pair(
    tokenizer: Any,
    d3tok_text: str,
    surface_text: str,
    feature_groups: Sequence[str],
    max_length: int,
) -> dict[str, Any]:
    """Keep D3Tok/Surface first, then retain the largest priority feature prefix."""

    d3tok_ids = tokenizer.encode(d3tok_text, add_special_tokens=False)
    surface_ids = tokenizer.encode(surface_text, add_special_tokens=False)
    feature_group_ids = [
        tokenizer.encode(group, add_special_tokens=False)
        for group in feature_groups
    ]
    if (
        not d3tok_ids
        or not surface_ids
        or not feature_group_ids
        or any(not group_ids for group_ids in feature_group_ids)
    ):
        raise ValueError("Structured pair contains an empty tokenized component")

    pair_special_tokens = int(tokenizer.num_special_tokens_to_add(pair=True))
    d3tok_only_length = pair_special_tokens + len(d3tok_ids)
    if d3tok_only_length > max_length:
        raise ValueError(
            "D3Tok alone exceeds MAX_LENGTH. D3Tok was not truncated; this "
            "sample requires a chunking strategy. "
            f"d3tok_tokens={len(d3tok_ids)}, pair_special_tokens="
            f"{pair_special_tokens}, max_length={max_length}"
        )

    # First try the complete Surface and every atomic feature group. If that
    # overflows, remove whole groups from the lowest-priority end. Only after
    # every feature is gone may Surface be truncated.
    selected_group_ids = list(feature_group_ids)

    def encoded_length() -> int:
        feature_length = sum(len(group_ids) for group_ids in selected_group_ids)
        feature_separator = 1 if selected_group_ids else 0
        return (
            pair_special_tokens
            + len(d3tok_ids)
            + len(surface_ids)
            + feature_separator
            + feature_length
        )

    while selected_group_ids and encoded_length() > max_length:
        selected_group_ids.pop()

    if encoded_length() > max_length:
        surface_budget = max_length - pair_special_tokens - len(d3tok_ids)
        surface_ids = surface_ids[: max(0, surface_budget)]

    selected_feature_ids = [
        token_id
        for group_ids in selected_group_ids
        for token_id in group_ids
    ]
    paired_ids = list(surface_ids)
    if selected_feature_ids:
        paired_ids.append(int(tokenizer.sep_token_id))
        paired_ids.extend(selected_feature_ids)

    encoded = dict(
        tokenizer.prepare_for_model(
            d3tok_ids,
            pair_ids=paired_ids,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
    )
    if len(encoded["input_ids"]) > max_length:
        raise AssertionError("Reserved structured input exceeded MAX_LENGTH")
    encoded_d3tok_ids = encoded["input_ids"][1 : 1 + len(d3tok_ids)]
    if encoded_d3tok_ids != d3tok_ids:
        raise AssertionError("D3Tok was altered or truncated during pair encoding")
    if selected_feature_ids:
        feature_start = len(encoded["input_ids"]) - 1 - len(selected_feature_ids)
        if encoded["input_ids"][feature_start:-1] != selected_feature_ids:
            raise AssertionError(
                "A selected structured feature group was altered during encoding"
            )
    return encoded


class BARECDataset(Dataset[dict[str, Any]]):
    """Encode D3Tok as segment A and Surface/features as segment B."""

    def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int) -> None:
        self.d3tok_texts = frame["_d3tok_text"].astype(str).tolist()
        rows = frame.to_dict(orient="records")
        self.surface_texts = [str(row["_surface_text"]) for row in rows]
        self.feature_groups = [structured_feature_groups(row) for row in rows]
        self.ids = frame["_id"].astype(str).tolist()
        self.indices = frame["_original_index"].astype(int).tolist()
        self.has_labels = bool(frame.attrs.get("has_labels", False))
        self.labels = (
            frame["_label"].astype(float).tolist() if self.has_labels else None
        )
        self.auxiliary_labels = (
            {
                3: frame["_label3"].astype(int).tolist(),
                5: frame["_label5"].astype(int).tolist(),
                7: frame["_label7"].astype(int).tolist(),
            }
            if self.has_labels
            else None
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.d3tok_texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        encoded = encode_structured_pair(
            self.tokenizer,
            self.d3tok_texts[index],
            self.surface_texts[index],
            self.feature_groups[index],
            self.max_length,
        )
        item: dict[str, Any] = dict(encoded)
        item["sample_id"] = self.ids[index]
        item["original_index"] = self.indices[index]
        if self.labels is not None:
            item["label"] = self.labels[index]
            assert self.auxiliary_labels is not None
            item["label3"] = self.auxiliary_labels[3][index]
            item["label5"] = self.auxiliary_labels[5][index]
            item["label7"] = self.auxiliary_labels[7][index]
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
        auxiliary_labels: dict[int, list[int]] = {3: [], 5: [], 7: []}
        labels_present = "label" in features[0]
        non_model_keys = {
            "sample_id",
            "original_index",
            "label",
            "label3",
            "label5",
            "label7",
        }
        for feature in features:
            model_features.append(
                {
                    key: value
                    for key, value in feature.items()
                    if key not in non_model_keys
                }
            )
            sample_ids.append(str(feature["sample_id"]))
            indices.append(int(feature["original_index"]))
            if labels_present:
                labels.append(float(feature["label"]))
                for level_count in auxiliary_labels:
                    auxiliary_labels[level_count].append(
                        int(feature[f"label{level_count}"])
                    )
        batch = dict(self.tokenizer.pad(model_features, padding=True, return_tensors="pt"))
        batch["sample_ids"] = sample_ids
        batch["original_indices"] = torch.tensor(indices, dtype=torch.long)
        if labels_present:
            batch["labels"] = torch.tensor(labels, dtype=torch.float32)
            for level_count, values in auxiliary_labels.items():
                batch[f"labels{level_count}"] = torch.tensor(
                    values, dtype=torch.long
                )
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
    """AraBERT encoder with a dropout and scalar sentence-regression head."""

    def __init__(self, model_name: str, dropout: float) -> None:
        super().__init__()
        # The regression head consumes raw CLS from last_hidden_state.  A
        # BertModel pooler would therefore be trainable but disconnected from
        # the loss, which makes DDP fail on the following iteration when
        # find_unused_parameters=False.  Do not instantiate those two unused
        # pooler tensors in the first place.
        self.encoder = AutoModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
        )
        trainable_pooler_parameters = [
            name
            for name, parameter in self.encoder.named_parameters()
            if parameter.requires_grad and "pooler" in name.lower()
        ]
        if trainable_pooler_parameters:
            raise RuntimeError(
                "The CLS regressor must not retain trainable encoder-pooler "
                f"parameters: {trainable_pooler_parameters}"
            )
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.regression_head = nn.Linear(hidden_size, 1)

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
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.regression_head(self.dropout(cls_embedding)).squeeze(-1)


class CascadedHierarchicalReadabilityRegressor(nn.Module):
    """Joint 3→5→7→19 model with differentiable coarse predictions."""

    def __init__(self, model_name: str, dropout: float, temperature: float) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
        )
        trainable_pooler_parameters = [
            name
            for name, parameter in self.encoder.named_parameters()
            if parameter.requires_grad and "pooler" in name.lower()
        ]
        if trainable_pooler_parameters:
            raise RuntimeError(
                "The cascaded CLS model must not retain trainable encoder-pooler "
                f"parameters: {trainable_pooler_parameters}"
            )
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.temperature = float(temperature)
        self.head3 = nn.Linear(hidden_size, 3)
        self.head5 = nn.Linear(hidden_size + 3, 5)
        self.head7 = nn.Linear(hidden_size + 5, 7)
        self.head19 = nn.Linear(hidden_size + 7, 1)

    @staticmethod
    def _soft_condition(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Build stable differentiable probabilities without detaching logits."""

        probabilities = torch.softmax(logits.float() / temperature, dim=-1)
        return probabilities.to(dtype=logits.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        encoder_arguments: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_arguments["token_type_ids"] = token_type_ids
        outputs = self.encoder(**encoder_arguments)
        hidden = outputs.last_hidden_state[:, 0, :]

        logits3 = self.head3(self.dropout(hidden))
        probabilities3 = self._soft_condition(logits3, self.temperature)
        logits5 = self.head5(self.dropout(torch.cat([hidden, probabilities3], dim=-1)))
        probabilities5 = self._soft_condition(logits5, self.temperature)
        logits7 = self.head7(self.dropout(torch.cat([hidden, probabilities5], dim=-1)))
        probabilities7 = self._soft_condition(logits7, self.temperature)
        score19 = self.head19(
            self.dropout(torch.cat([hidden, probabilities7], dim=-1))
        ).squeeze(-1)
        return {
            "score19": score19,
            "logits3": logits3,
            "logits5": logits5,
            "logits7": logits7,
        }


ReadabilityModel = (
    ArabicReadabilityRegressor | CascadedHierarchicalReadabilityRegressor
)


def create_model(config: Config) -> ReadabilityModel:
    """Instantiate the selected architecture without changing its checkpoint mode."""

    if config.MODEL_MODE == "baseline_mse":
        return ArabicReadabilityRegressor(config.MODEL_NAME, config.DROPOUT)
    if config.MODEL_MODE == "cascaded_hmtl":
        return CascadedHierarchicalReadabilityRegressor(
            config.MODEL_NAME,
            config.DROPOUT,
            config.CASCADE_TEMPERATURE,
        )
    raise ValueError(f"Unsupported MODEL_MODE: {config.MODEL_MODE!r}")


def unwrap_model(model: nn.Module) -> ReadabilityModel:
    """Return the underlying model whether or not DDP wraps it."""

    unwrapped = model.module if isinstance(model, DDP) else model
    if not isinstance(
        unwrapped,
        (ArabicReadabilityRegressor, CascadedHierarchicalReadabilityRegressor),
    ):
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
    if isinstance(base, ArabicReadabilityRegressor):
        head_modules = nn.ModuleList([base.dropout, base.regression_head])
    else:
        head_modules = nn.ModuleList(
            [base.dropout, base.head3, base.head5, base.head7, base.head19]
        )
    head_decay, head_no_decay = split_decay_parameters(head_modules)
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


def model_forward(
    model: nn.Module, batch: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    """Pass only encoder tensors from a collated batch into the model."""

    raw_output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch.get("token_type_ids"),
    )
    outputs = (
        {"score19": raw_output}
        if torch.is_tensor(raw_output)
        else dict(raw_output)
    )
    score19 = outputs.get("score19")
    if score19 is None or score19.ndim != 1 or score19.shape[0] != batch["input_ids"].shape[0]:
        raise AssertionError(
            "Regression output must have shape [batch_size], got "
            f"{None if score19 is None else tuple(score19.shape)}"
        )
    expected_logits = {"logits3": 3, "logits5": 5, "logits7": 7}
    present_logits = {key for key in expected_logits if key in outputs}
    if present_logits and present_logits != set(expected_logits):
        raise AssertionError("Cascaded output must provide all 3/5/7-level logits")
    for key, class_count in expected_logits.items():
        if key in outputs and outputs[key].shape != torch.Size(
            [batch["input_ids"].shape[0], class_count]
        ):
            raise AssertionError(
                f"{key} must have shape [batch_size, {class_count}], "
                f"got {tuple(outputs[key].shape)}"
            )
    return outputs


def compute_training_losses(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    config: Config,
) -> dict[str, torch.Tensor]:
    """Calculate the complete FP32 baseline or cascaded joint objective."""

    mse19 = nn.functional.mse_loss(
        outputs["score19"].float(), batch["labels"].float()
    )
    losses = {
        "mse19": mse19,
        "ce3": mse19.new_zeros(()),
        "ce5": mse19.new_zeros(()),
        "ce7": mse19.new_zeros(()),
    }
    if config.MODEL_MODE == "cascaded_hmtl":
        for level_count in (3, 5, 7):
            logits_key = f"logits{level_count}"
            labels_key = f"labels{level_count}"
            if logits_key not in outputs or labels_key not in batch:
                raise RuntimeError(
                    f"Cascaded training requires {logits_key} and {labels_key}"
                )
            losses[f"ce{level_count}"] = nn.functional.cross_entropy(
                outputs[logits_key].float(), batch[labels_key].long()
            )
    losses["total"] = (
        losses["mse19"]
        + config.AUX_CE3_WEIGHT * losses["ce3"]
        + config.AUX_CE5_WEIGHT * losses["ce5"]
        + config.AUX_CE7_WEIGHT * losses["ce7"]
    )
    return losses


def run_regression_shape_check(
    model: ReadabilityModel,
    tokenizer: Any,
    d3tok_text: str,
    surface_text: str,
    feature_groups: Sequence[str],
    device: torch.device,
    max_length: int,
) -> None:
    """Verify batch-one output shape and gradient connectivity before DDP."""

    encoded = encode_structured_pair(
        tokenizer,
        d3tok_text,
        surface_text,
        feature_groups,
        max_length,
    )
    token_type_ids = encoded.get("token_type_ids")
    if token_type_ids is None or not {0, 1}.issubset(set(token_type_ids)):
        raise AssertionError("Paired input must contain both BERT token-type segments")
    encoded = dict(
        tokenizer.pad([encoded], padding=True, return_tensors="pt")
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        output = model_forward(model, encoded)
        output["score19"].sum().backward()
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
    round_predictions: np.ndarray
    up_predictions: np.ndarray
    down_predictions: np.ndarray
    labels: Optional[np.ndarray]
    metrics: Optional[dict[str, float]]


def round_and_clip(raw_predictions: Sequence[float], config: Config) -> np.ndarray:
    """Convert regression scores to legal integer readability levels."""

    raw = np.asarray(raw_predictions, dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("Raw predictions contain NaN or infinity")
    return np.clip(np.rint(raw), config.MIN_LABEL, config.MAX_LABEL).astype(np.int64)


def ceil_and_clip(raw_predictions: Sequence[float], config: Config) -> np.ndarray:
    """Round every non-integer score upward, then enforce the legal range."""

    raw = np.asarray(raw_predictions, dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("Raw predictions contain NaN or infinity")
    return np.clip(np.ceil(raw), config.MIN_LABEL, config.MAX_LABEL).astype(np.int64)


def floor_and_clip(raw_predictions: Sequence[float], config: Config) -> np.ndarray:
    """Round every score downward, then enforce the legal label range."""

    raw = np.asarray(raw_predictions, dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("Raw predictions contain NaN or infinity")
    return np.clip(np.floor(raw), config.MIN_LABEL, config.MAX_LABEL).astype(np.int64)


def run_discretization_checks(config: Config) -> None:
    """Lock round/ceil/floor semantics, clipping, and finite-input rejection."""

    raw = [0.2, 1.0, 1.5, 2.5, 3.5, 7.0, 7.0001, 18.5, 19.2]
    if round_and_clip(raw, config).tolist() != [1, 1, 2, 2, 4, 7, 7, 18, 19]:
        raise AssertionError("np.rint discretization check failed")
    if ceil_and_clip(raw, config).tolist() != [1, 1, 2, 3, 4, 7, 8, 19, 19]:
        raise AssertionError("np.ceil discretization check failed")
    if floor_and_clip(raw, config).tolist() != [1, 1, 1, 2, 3, 7, 7, 18, 19]:
        raise AssertionError("np.floor discretization check failed")
    for discretizer in (round_and_clip, ceil_and_clip, floor_and_clip):
        try:
            discretizer([float("nan")], config)
        except ValueError:
            pass
        else:
            raise AssertionError("Non-finite prediction was not rejected")


def calculate_metrics(
    labels: Sequence[float], raw_predictions: Sequence[float], config: Config
) -> dict[str, float]:
    """Calculate raw MSE and integer-label BAREC metrics."""

    truth = np.asarray(labels, dtype=np.int64)
    raw = np.asarray(raw_predictions, dtype=np.float64)
    rounded = round_and_clip(raw, config)
    up = ceil_and_clip(raw, config)
    down = floor_and_clip(raw, config)
    round_qwk = float(
        cohen_kappa_score(
            truth,
            rounded,
            weights="quadratic",
            labels=list(range(config.MIN_LABEL, config.MAX_LABEL + 1)),
        )
    )
    up_qwk = float(
        cohen_kappa_score(
            truth,
            up,
            weights="quadratic",
            labels=list(range(config.MIN_LABEL, config.MAX_LABEL + 1)),
        )
    )
    down_qwk = float(
        cohen_kappa_score(
            truth,
            down,
            weights="quadratic",
            labels=list(range(config.MIN_LABEL, config.MAX_LABEL + 1)),
        )
    )
    return {
        "mse": float(np.mean(np.square(raw - truth))),
        "round_mae": float(np.mean(np.abs(rounded - truth))),
        "round_qwk": round_qwk,
        "round_exact_accuracy": float(np.mean(rounded == truth)),
        "round_adjacent_accuracy": float(np.mean(np.abs(rounded - truth) <= 1)),
        "up_mae": float(np.mean(np.abs(up - truth))),
        "up_qwk": up_qwk,
        "up_exact_accuracy": float(np.mean(up == truth)),
        "up_adjacent_accuracy": float(np.mean(np.abs(up - truth) <= 1)),
        "down_mae": float(np.mean(np.abs(down - truth))),
        "down_qwk": down_qwk,
        "down_exact_accuracy": float(np.mean(down == truth)),
        "down_adjacent_accuracy": float(np.mean(np.abs(down - truth) <= 1)),
    }


def down_checkpoint_decision(
    metrics: Mapping[str, float],
    *,
    best_down_qwk: float,
    best_down_mae: float,
    has_selected_model: bool,
) -> tuple[float, float, bool]:
    """Apply the fixed Dev floor-QWK/floor-MAE checkpoint ordering."""

    down_qwk = float(metrics["down_qwk"])
    down_mae = float(metrics["down_mae"])
    if not math.isfinite(down_mae):
        raise ValueError("Dev down_mae must be finite for checkpoint selection")
    selection_qwk = down_qwk if math.isfinite(down_qwk) else -math.inf
    qwk_tied = (
        abs(selection_qwk - best_down_qwk) <= 1e-12
        if math.isfinite(selection_qwk) and math.isfinite(best_down_qwk)
        else selection_qwk == best_down_qwk
    )
    improved = (
        not has_selected_model
        or selection_qwk > best_down_qwk + 1e-12
        or (qwk_tied and down_mae < best_down_mae)
    )
    return selection_qwk, down_mae, improved


def merge_evaluation_payloads(
    payloads: Sequence[Mapping[str, Any]], expected_size: int
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    dict[int, float],
]:
    """Sort gathered records by original index and remove sampler padding."""

    records: dict[
        int,
        tuple[
            str,
            float,
            Optional[float],
            Optional[dict[int, tuple[int, int]]],
        ],
    ] = {}
    any_labels = False
    for payload in payloads:
        indices = payload["indices"]
        ids = payload["ids"]
        predictions = payload["predictions"]
        labels = payload.get("labels")
        auxiliary_predictions = payload.get("auxiliary_predictions")
        auxiliary_labels = payload.get("auxiliary_labels")
        if not (len(indices) == len(ids) == len(predictions)):
            raise RuntimeError("Malformed distributed evaluation payload")
        if labels is not None and len(labels) != len(indices):
            raise RuntimeError("Malformed distributed evaluation labels")
        if (auxiliary_predictions is None) != (auxiliary_labels is None):
            raise RuntimeError("Only one auxiliary evaluation payload was provided")
        if auxiliary_predictions is not None:
            for level_count in (3, 5, 7):
                if not (
                    len(auxiliary_predictions[level_count])
                    == len(auxiliary_labels[level_count])
                    == len(indices)
                ):
                    raise RuntimeError("Malformed auxiliary evaluation payload")
        any_labels = any_labels or labels is not None
        for position, index in enumerate(indices):
            integer_index = int(index)
            record_label = None if labels is None else float(labels[position])
            auxiliary_record = None
            if auxiliary_predictions is not None and auxiliary_labels is not None:
                auxiliary_record = {
                    level_count: (
                        int(auxiliary_predictions[level_count][position]),
                        int(auxiliary_labels[level_count][position]),
                    )
                    for level_count in (3, 5, 7)
                }
            candidate = (
                str(ids[position]),
                float(predictions[position]),
                record_label,
                auxiliary_record,
            )
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
    auxiliary_accuracies: dict[int, float] = {}
    if any(record[3] is not None for record in ordered):
        if any(record[3] is None for record in ordered):
            raise RuntimeError("Only part of the evaluation set has auxiliary outputs")
        for level_count in (3, 5, 7):
            correct = sum(
                int(record[3][level_count][0] == record[3][level_count][1])
                for record in ordered
                if record[3] is not None
            )
            auxiliary_accuracies[level_count] = float(correct / expected_size)
    return (
        ordered_ids,
        np.arange(expected_size),
        predictions,
        label_array,
        auxiliary_accuracies,
    )


def run_gather_order_check() -> None:
    """Verify out-of-order/padded evaluation records are restored exactly once."""

    payloads = [
        {"indices": [2, 0], "ids": ["c", "a"], "predictions": [3.0, 1.0]},
        {"indices": [1, 0], "ids": ["b", "a"], "predictions": [2.0, 1.0]},
    ]
    ids, indices, predictions, labels, auxiliary = merge_evaluation_payloads(
        payloads, 3
    )
    if ids != ["a", "b", "c"] or indices.tolist() != [0, 1, 2]:
        raise AssertionError("Evaluation gather ordering check failed")
    if (
        predictions.tolist() != [1.0, 2.0, 3.0]
        or labels is not None
        or auxiliary
    ):
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
    local_auxiliary_predictions: dict[int, list[int]] = {3: [], 5: [], 7: []}
    local_auxiliary_labels: dict[int, list[int]] = {3: [], 5: [], 7: []}
    amp_enabled = config.USE_FP16 and context.device.type == "cuda"
    progress = tqdm(loader, desc=description, disable=not context.is_main, leave=False)
    for batch in progress:
        batch = move_model_batch(batch, context.device)
        with autocast_context(amp_enabled):
            outputs = model_forward(model, batch)
            predictions = outputs["score19"]
        local_indices.extend(batch["original_indices"].cpu().tolist())
        local_ids.extend(batch["sample_ids"])
        local_predictions.extend(predictions.float().cpu().tolist())
        if "labels" in batch:
            assert local_labels is not None
            local_labels.extend(batch["labels"].float().cpu().tolist())
            if all(f"logits{level}" in outputs for level in (3, 5, 7)):
                for level_count in (3, 5, 7):
                    auxiliary_predictions = outputs[f"logits{level_count}"].argmax(
                        dim=-1
                    )
                    local_auxiliary_predictions[level_count].extend(
                        auxiliary_predictions.cpu().tolist()
                    )
                    local_auxiliary_labels[level_count].extend(
                        batch[f"labels{level_count}"].cpu().tolist()
                    )
        else:
            local_labels = None

    payload: dict[str, Any] = {
        "indices": local_indices,
        "ids": local_ids,
        "predictions": local_predictions,
        "labels": local_labels,
        "auxiliary_predictions": (
            local_auxiliary_predictions
            if local_auxiliary_predictions[3]
            else None
        ),
        "auxiliary_labels": (
            local_auxiliary_labels if local_auxiliary_labels[3] else None
        ),
    }
    if context.distributed:
        gathered: list[Optional[dict[str, Any]]] = [None] * context.world_size
        dist.all_gather_object(gathered, payload)
        payloads = [item for item in gathered if item is not None]
    else:
        payloads = [payload]

    if not context.is_main:
        return None
    (
        ids,
        indices,
        raw_predictions,
        labels,
        auxiliary_accuracies,
    ) = merge_evaluation_payloads(payloads, expected_size)
    round_predictions = round_and_clip(raw_predictions, config)
    up_predictions = ceil_and_clip(raw_predictions, config)
    down_predictions = floor_and_clip(raw_predictions, config)
    metrics = calculate_metrics(labels, raw_predictions, config) if labels is not None else None
    if metrics is not None:
        for level_count, accuracy in auxiliary_accuracies.items():
            metrics[f"aux_accuracy_{level_count}"] = accuracy
    return EvaluationOutput(
        ids,
        indices,
        raw_predictions,
        round_predictions,
        up_predictions,
        down_predictions,
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
    best_down_qwk: float,
    best_down_mae: float,
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
        "best_down_qwk": best_down_qwk,
        "best_down_mae": best_down_mae,
        "bad_epochs": bad_epochs,
        "config": asdict(config),
        "resume_signature": resume_compatibility_signature(config),
        "rng_states": list(rng_states),
    }
    atomic_torch_save(payload, path)


def resume_compatibility_signature(config: Config) -> dict[str, Any]:
    """Describe model/training semantics that a resume must not change."""

    signature: dict[str, Any] = {
        "MODEL_NAME": config.MODEL_NAME,
        "MODEL_MODE": config.MODEL_MODE,
        "MAX_LENGTH": config.MAX_LENGTH,
        "MIN_LABEL": config.MIN_LABEL,
        "MAX_LABEL": config.MAX_LABEL,
        "DROPOUT": config.DROPOUT,
        "STRUCTURED_SPECIAL_TOKENS": tuple(FIELD_TOKENS),
        "PREPROCESSING_VERSION": PREPROCESSING_VERSION,
        "CHECKPOINT_SELECTION_POLICY": CHECKPOINT_SELECTION_POLICY,
        "FINAL_DISCRETIZATION_POLICY": FINAL_DISCRETIZATION_POLICY,
    }
    if config.MODEL_MODE == "cascaded_hmtl":
        signature.update(
            {
                "CASCADE_TEMPERATURE": config.CASCADE_TEMPERATURE,
                "AUX_CE3_WEIGHT": config.AUX_CE3_WEIGHT,
                "AUX_CE5_WEIGHT": config.AUX_CE5_WEIGHT,
                "AUX_CE7_WEIGHT": config.AUX_CE7_WEIGHT,
            }
        )
    return signature


def validate_resume_compatibility(
    checkpoint: Mapping[str, Any], config: Config
) -> None:
    """Reject checkpoint/config combinations that silently change semantics."""

    expected = resume_compatibility_signature(config)
    saved_signature = checkpoint.get("resume_signature")
    checkpoint_config = checkpoint.get("config", {})
    if saved_signature is None:
        # Older scalar-baseline checkpoints predate explicit signatures. Read
        # their compatible fields so the error is informative, but the floor
        # policy fields below deliberately make unsigned checkpoints invalid.
        # Reusing them could silently preserve a round-selected best model.
        saved_mode = str(checkpoint_config.get("MODEL_MODE", "baseline_mse"))
        if config.MODEL_MODE != "baseline_mse" or saved_mode != "baseline_mse":
            raise RuntimeError(
                "Cascaded resume checkpoint lacks a compatibility signature"
            )
        saved_signature = {
            key: checkpoint_config[key]
            for key in ("MODEL_NAME", "MODEL_MODE", "MAX_LENGTH", "DROPOUT")
            if key in checkpoint_config
        }

    mismatches: dict[str, tuple[Any, Any]] = {}
    for key, saved_value in dict(saved_signature).items():
        if key not in expected:
            continue
        expected_value = expected[key]
        if key == "STRUCTURED_SPECIAL_TOKENS":
            saved_value = tuple(saved_value)
        if saved_value != expected_value:
            mismatches[key] = (saved_value, expected_value)
    missing = sorted(set(expected).difference(saved_signature))
    policy_fields = {
        "CHECKPOINT_SELECTION_POLICY",
        "FINAL_DISCRETIZATION_POLICY",
    }
    if missing and (
        config.MODEL_MODE == "cascaded_hmtl" or policy_fields.intersection(missing)
    ):
        raise RuntimeError(
            "Resume checkpoint is missing compatibility fields required by the "
            "floor-selection pipeline: "
            f"{missing}"
        )
    if mismatches:
        details = ", ".join(
            f"{key}: saved={saved!r}, requested={requested!r}"
            for key, (saved, requested) in sorted(mismatches.items())
        )
        raise RuntimeError(f"Checkpoint/config compatibility mismatch: {details}")


def resume_training(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    config: Config,
    context: DistributedContext,
) -> tuple[int, int, float, float, int]:
    """Restore a full training checkpoint and return selection state."""

    checkpoint = load_torch_file(checkpoint_path, context.device)
    validate_resume_compatibility(checkpoint, config)
    missing_selection_state = sorted(
        {"best_down_qwk", "best_down_mae"}.difference(checkpoint)
    )
    if missing_selection_state:
        raise RuntimeError(
            "Resume checkpoint lacks floor-selection state: "
            f"{missing_selection_state}. Restart this seed from epoch 1."
        )
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
        float(checkpoint["best_down_qwk"]),
        float(checkpoint["best_down_mae"]),
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
        if context.is_main:
            LOGGER.info(
                "Train weighted sampler: OFF; using the natural Train "
                "distribution with deterministic distributed shuffling."
            )
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
) -> tuple[dict[str, float], int]:
    """Train exactly one epoch on Train; Dev/Test never call this function."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = config.USE_FP16 and context.device.type == "cuda"
    loss_totals = {"total": 0.0, "mse19": 0.0, "ce3": 0.0, "ce5": 0.0, "ce7": 0.0}
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
                outputs = model_forward(model, batch)
            component_losses = compute_training_losses(outputs, batch, config)
            raw_loss = component_losses["total"]
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
        loss_totals["total"] += float(raw_loss.detach().item()) * batch_size
        for name, loss in component_losses.items():
            if name == "total":
                continue
            loss_totals[name] += float(loss.detach().item()) * batch_size
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
                    mse19=f"{component_losses['mse19'].item():.4f}",
                    ce3=f"{component_losses['ce3'].item():.3f}",
                    ce5=f"{component_losses['ce5'].item():.3f}",
                    ce7=f"{component_losses['ce7'].item():.3f}",
                    avg=f"{loss_totals['total'] / max(total_examples, 1):.4f}",
                    enc_lr=f"{encoder_lr:.2e}",
                    head_lr=f"{head_lr:.2e}",
                    accum=f"{(step % config.GRADIENT_ACCUMULATION_STEPS) + 1}/"
                    f"{config.GRADIENT_ACCUMULATION_STEPS}",
                    elapsed=f"{elapsed:.0f}s",
                    mem=memory,
                )
    progress.close()

    loss_names = ("total", "mse19", "ce3", "ce5", "ce7")
    totals = torch.tensor(
        [*(loss_totals[name] for name in loss_names), float(total_examples)],
        dtype=torch.float64,
        device=context.device,
    )
    if context.distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    denominator = max(totals[-1].item(), 1.0)
    mean_losses = {
        name: float(totals[index].item() / denominator)
        for index, name in enumerate(loss_names)
    }
    return mean_losses, global_step


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
    *,
    expected_prediction_name: str,
    expected_zip_name: str,
) -> None:
    """Validate exact filename, CSV schema/order/range, and one-entry ZIP layout."""

    if prediction_path.name != expected_prediction_name:
        raise ValueError(
            f"Submission source file must be named {expected_prediction_name!r}"
        )
    if zip_path.name != expected_zip_name:
        raise ValueError(f"ZIP must be named {expected_zip_name!r}")
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
        prediction_path = root / config.SUBMISSION_DOWN_BASENAME
        zip_path = root / config.SUBMISSION_DOWN_ZIP_NAME
        prediction_path.write_text("Wrong,Prediction\nprobe,1\n", encoding="utf-8")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(prediction_path, arcname=config.SUBMISSION_BASENAME)
        try:
            validate_submission(
                prediction_path,
                zip_path,
                ["probe"],
                config,
                expected_prediction_name=config.SUBMISSION_DOWN_BASENAME,
                expected_zip_name=config.SUBMISSION_DOWN_ZIP_NAME,
            )
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
            validate_submission(
                prediction_path,
                zip_path,
                ["probe"],
                config,
                expected_prediction_name=config.SUBMISSION_DOWN_BASENAME,
                expected_zip_name=config.SUBMISSION_DOWN_ZIP_NAME,
            )
        except ValueError as error:
            if "zip" not in str(error).lower():
                raise AssertionError("Nested-ZIP check failed for the wrong reason") from error
        else:
            raise AssertionError("Submission validator accepted a nested ZIP entry")


def create_submission_variant(
    ids: Sequence[str],
    predictions: Sequence[int],
    config: Config,
    *,
    prediction_name: str,
    zip_name: str,
) -> tuple[Path, Path]:
    """Write one named source CSV while keeping the official ZIP member name."""

    directory = config.resolve(config.SUBMISSION_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    prediction_path = directory / prediction_name
    zip_path = directory / zip_name
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
    validate_submission(
        prediction_path,
        zip_path,
        ids,
        config,
        expected_prediction_name=prediction_name,
        expected_zip_name=zip_name,
    )
    return prediction_path, zip_path


def create_submissions(
    ids: Sequence[str],
    down_predictions: Sequence[int],
    config: Config,
) -> dict[str, tuple[Path, Path]]:
    """Create the one floor-discretized submission from the raw-score ensemble."""

    directory = config.resolve(config.SUBMISSION_DIR)
    # Remove stale artifacts from the previous generic/round/up output policy
    # so a rerun cannot leave a misleading ZIP beside the floor-selected one.
    for stale_name in (
        config.SUBMISSION_BASENAME,
        f"{config.SUBMISSION_BASENAME}.zip",
        "prediction_round",
        "prediction_round.zip",
        "prediction_up",
        "prediction_up.zip",
    ):
        (directory / stale_name).unlink(missing_ok=True)
    down_paths = create_submission_variant(
        ids,
        down_predictions,
        config,
        prediction_name=config.SUBMISSION_DOWN_BASENAME,
        zip_name=config.SUBMISSION_DOWN_ZIP_NAME,
    )
    print(
        "Floor-discretized submission created successfully:\n"
        f"down: {down_paths[1].resolve()}"
    )
    return {"down": down_paths}


def write_diagnostics(
    output: EvaluationOutput,
    config: Config,
    *,
    split_name: str,
) -> Path:
    """Write raw/final scores for one split separately from the submission."""

    data: dict[str, Any] = {
        "Sentence ID": output.ids,
        "raw_prediction": output.raw_predictions,
        "Prediction_down": output.down_predictions,
        "Prediction_round": output.round_predictions,
        "Prediction_up": output.up_predictions,
    }
    if output.labels is not None:
        data["gold_label"] = output.labels.astype(np.int64)
    path = (
        config.resolve(config.OUTPUT_DIR)
        / "diagnostics"
        / f"{split_name}_predictions_with_raw_scores.csv"
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
            config.resolve(config.CACHE_DIR),
            config.resolve(config.SUBMISSION_DIR),
            config.resolve(config.SEED_RUNS_DIR),
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


@dataclass
class SeedRunOutput:
    """Best-checkpoint Dev/Test predictions produced by one random seed."""

    seed: int
    dev: EvaluationOutput
    test: EvaluationOutput


def average_evaluation_outputs(
    outputs: Sequence[EvaluationOutput],
    config: Config,
) -> EvaluationOutput:
    """Uniformly average raw scores while preserving exact row alignment."""

    if not outputs:
        raise ValueError("At least one seed output is required for ensembling")
    reference = outputs[0]
    for output in outputs[1:]:
        if output.ids != reference.ids:
            raise RuntimeError("Seed outputs contain different ID orders")
        if not np.array_equal(output.indices, reference.indices):
            raise RuntimeError("Seed outputs contain different original indices")
        if (output.labels is None) != (reference.labels is None):
            raise RuntimeError("Only part of the seed outputs contains labels")
        if (
            reference.labels is not None
            and output.labels is not None
            and not np.array_equal(output.labels, reference.labels)
        ):
            raise RuntimeError("Seed outputs contain different gold labels")

    raw_predictions = np.mean(
        np.stack(
            [output.raw_predictions.astype(np.float64) for output in outputs],
            axis=0,
        ),
        axis=0,
    )
    round_predictions = round_and_clip(raw_predictions, config)
    up_predictions = ceil_and_clip(raw_predictions, config)
    down_predictions = floor_and_clip(raw_predictions, config)
    metrics = (
        calculate_metrics(reference.labels, raw_predictions, config)
        if reference.labels is not None
        else None
    )
    return EvaluationOutput(
        ids=list(reference.ids),
        indices=reference.indices.copy(),
        raw_predictions=raw_predictions,
        round_predictions=round_predictions,
        up_predictions=up_predictions,
        down_predictions=down_predictions,
        labels=None if reference.labels is None else reference.labels.copy(),
        metrics=metrics,
    )


def run_ensemble_averaging_check(config: Config) -> None:
    """Verify that raw scores are averaged before rounding and clipping."""

    left = EvaluationOutput(
        ids=["a", "b"],
        indices=np.asarray([0, 1], dtype=np.int64),
        raw_predictions=np.asarray([1.2, 3.4], dtype=np.float64),
        round_predictions=np.asarray([1, 3], dtype=np.int64),
        up_predictions=np.asarray([2, 4], dtype=np.int64),
        down_predictions=np.asarray([1, 3], dtype=np.int64),
        labels=np.asarray([1.0, 4.0], dtype=np.float64),
        metrics=None,
    )
    right = EvaluationOutput(
        ids=["a", "b"],
        indices=np.asarray([0, 1], dtype=np.int64),
        raw_predictions=np.asarray([2.0, 4.6], dtype=np.float64),
        round_predictions=np.asarray([2, 5], dtype=np.int64),
        up_predictions=np.asarray([2, 5], dtype=np.int64),
        down_predictions=np.asarray([2, 4], dtype=np.int64),
        labels=np.asarray([1.0, 4.0], dtype=np.float64),
        metrics=None,
    )
    averaged = average_evaluation_outputs([left, right], config)
    if not np.allclose(averaged.raw_predictions, [1.6, 4.0]):
        raise AssertionError("Multi-seed raw-score averaging check failed")
    if averaged.round_predictions.tolist() != [2, 4]:
        raise AssertionError("Multi-seed round-after-average check failed")
    if averaged.up_predictions.tolist() != [2, 4]:
        raise AssertionError("Multi-seed ceil-after-average check failed")
    if averaged.down_predictions.tolist() != [1, 4]:
        raise AssertionError("Multi-seed floor-after-average check failed")


def write_ensemble_report(
    seed_runs: Sequence[SeedRunOutput],
    ensemble_dev: EvaluationOutput,
    ensemble_test: EvaluationOutput,
    config: Config,
) -> Path:
    """Record member metrics and the fixed uniform-averaging policy."""

    dev_down_qwks = np.asarray(
        [
            run.dev.metrics["down_qwk"]
            for run in seed_runs
            if run.dev.metrics is not None
        ],
        dtype=np.float64,
    )
    payload = {
        "seeds": [run.seed for run in seed_runs],
        "model_mode": config.MODEL_MODE,
        "joint_loss": {
            "mse19_weight": 1.0,
            "ce3_weight": config.AUX_CE3_WEIGHT,
            "ce5_weight": config.AUX_CE5_WEIGHT,
            "ce7_weight": config.AUX_CE7_WEIGHT,
        },
        "aggregation": "uniform_mean_of_raw_regression_scores",
        "final_discretization_policy": FINAL_DISCRETIZATION_POLICY,
        "discretization": {
            "down_primary": "numpy_floor_then_clip",
            "round_diagnostic_only": "numpy_rint_then_clip",
            "up_diagnostic_only": "numpy_ceil_then_clip",
        },
        "checkpoint_selection": CHECKPOINT_SELECTION_POLICY,
        "threshold_optimization": False,
        "learned_ensemble_weights": False,
        "members": [
            {
                "seed": run.seed,
                "dev_metrics": run.dev.metrics,
                # Open-Test metrics are diagnostic only and never participate
                # in checkpoint selection or ensemble construction.
                "open_test_metrics_diagnostic_only": run.test.metrics,
            }
            for run in seed_runs
        ],
        "member_dev_down_qwk_mean": (
            float(dev_down_qwks.mean()) if len(dev_down_qwks) else None
        ),
        "member_dev_down_qwk_std": (
            float(dev_down_qwks.std()) if len(dev_down_qwks) else None
        ),
        "ensemble_dev_metrics": ensemble_dev.metrics,
        "ensemble_open_test_metrics_diagnostic_only": ensemble_test.metrics,
    }
    path = config.resolve(config.OUTPUT_DIR) / "logs" / "ensemble_report.json"
    atomic_json_dump(payload, path)
    return path


def train_select_and_predict(
    train_frame: pd.DataFrame,
    dev_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    config: Config,
    context: DistributedContext,
    *,
    smoke_test: bool,
) -> Optional[SeedRunOutput]:
    """Train one seed, select only on Dev, and return best-model predictions."""

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=True)
    added_tokens = tokenizer.add_tokens(list(FIELD_TOKENS), special_tokens=False)
    validate_structured_tokenizer(tokenizer)
    if context.is_main:
        LOGGER.info(
            "Structured field tokens added to tokenizer: %d/%d",
            added_tokens,
            len(FIELD_TOKENS),
        )
    train_loader, dev_loader, test_loader, train_sampler = make_data_loaders(
        train_frame, dev_frame, test_frame, tokenizer, config, context
    )

    base_model = create_model(config)
    embedding_count = int(
        base_model.encoder.get_input_embeddings().num_embeddings
    )
    if embedding_count != len(tokenizer):
        base_model.encoder.resize_token_embeddings(len(tokenizer))
    if int(base_model.encoder.get_input_embeddings().num_embeddings) != len(tokenizer):
        raise RuntimeError("Tokenizer/model vocabulary sizes remain inconsistent")
    base_model.to(context.device)
    probe_row = train_frame.iloc[0].to_dict()
    run_regression_shape_check(
        base_model,
        tokenizer,
        str(probe_row["_d3tok_text"]),
        str(probe_row["_surface_text"]),
        structured_feature_groups(probe_row),
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
        LOGGER.info("Model mode: %s", config.MODEL_MODE)
        LOGGER.info(
            "Checkpoint selection: Dev floor QWK, tie-break Dev floor MAE; "
            "final ensemble discretization: floor after raw-score averaging."
        )
        if config.MODEL_MODE == "cascaded_hmtl":
            LOGGER.info(
                "Joint loss: MSE19 + %.3f*CE3 + %.3f*CE5 + %.3f*CE7; "
                "cascade temperature=%.3f",
                config.AUX_CE3_WEIGHT,
                config.AUX_CE5_WEIGHT,
                config.AUX_CE7_WEIGHT,
                config.CASCADE_TEMPERATURE,
            )

    start_epoch = 0
    global_step = 0
    best_down_qwk = -math.inf
    best_down_mae = math.inf
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
        (
            start_epoch,
            global_step,
            best_down_qwk,
            best_down_mae,
            bad_epochs,
        ) = resume_training(
            resume_path, model, optimizer, scheduler, scaler, config, context
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
        train_losses, global_step = train_one_epoch(
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
            selection_qwk, down_mae, improved = down_checkpoint_decision(
                metrics,
                best_down_qwk=best_down_qwk,
                best_down_mae=best_down_mae,
                has_selected_model=has_selected_model,
            )
            if improved:
                best_down_qwk = selection_qwk
                best_down_mae = down_mae
                bad_epochs = 0
                save_best_model(model, tokenizer, config, metrics)
                has_selected_model = True
            else:
                bad_epochs += 1
            elapsed = time.perf_counter() - epoch_start
            history_row = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": train_losses["total"],
                "train_mse19": train_losses["mse19"],
                "train_ce3": train_losses["ce3"],
                "train_ce5": train_losses["ce5"],
                "train_ce7": train_losses["ce7"],
                "dev_mse": metrics["mse"],
                "dev_down_mae": metrics["down_mae"],
                "dev_down_qwk": metrics["down_qwk"],
                "dev_down_exact_accuracy": metrics["down_exact_accuracy"],
                "dev_down_adjacent_accuracy": metrics["down_adjacent_accuracy"],
                "dev_round_mae": metrics["round_mae"],
                "dev_round_qwk": metrics["round_qwk"],
                "dev_round_exact_accuracy": metrics["round_exact_accuracy"],
                "dev_round_adjacent_accuracy": metrics[
                    "round_adjacent_accuracy"
                ],
                "dev_up_mae": metrics["up_mae"],
                "dev_up_qwk": metrics["up_qwk"],
                "dev_up_exact_accuracy": metrics["up_exact_accuracy"],
                "dev_up_adjacent_accuracy": metrics["up_adjacent_accuracy"],
                "dev_aux_accuracy_3": metrics.get("aux_accuracy_3"),
                "dev_aux_accuracy_5": metrics.get("aux_accuracy_5"),
                "dev_aux_accuracy_7": metrics.get("aux_accuracy_7"),
                "epoch_seconds": elapsed,
                "is_best": improved,
            }
            history.append(history_row)
            write_training_history(history, config)
            LOGGER.info(
                "Epoch %d | train_total=%.6f mse19=%.6f ce3=%.4f ce5=%.4f "
                "ce7=%.4f | Dev down_qwk=%s down_mae=%.4f "
                "round_qwk=%s round_mae=%.4f up_qwk=%s up_mae=%.4f "
                "aux_acc=(%s,%s,%s) time=%.1fs down_best=%s",
                epoch + 1,
                train_losses["total"],
                train_losses["mse19"],
                train_losses["ce3"],
                train_losses["ce5"],
                train_losses["ce7"],
                f"{metrics['down_qwk']:.6f}"
                if math.isfinite(metrics["down_qwk"])
                else "nan",
                metrics["down_mae"],
                f"{metrics['round_qwk']:.6f}"
                if math.isfinite(metrics["round_qwk"])
                else "nan",
                metrics["round_mae"],
                f"{metrics['up_qwk']:.6f}"
                if math.isfinite(metrics["up_qwk"])
                else "nan",
                metrics["up_mae"],
                f"{metrics['aux_accuracy_3']:.4f}"
                if "aux_accuracy_3" in metrics
                else "n/a",
                f"{metrics['aux_accuracy_5']:.4f}"
                if "aux_accuracy_5" in metrics
                else "n/a",
                f"{metrics['aux_accuracy_7']:.4f}"
                if "aux_accuracy_7" in metrics
                else "n/a",
                elapsed,
                improved,
            )
            decision = {
                "best_down_qwk": best_down_qwk,
                "best_down_mae": best_down_mae,
                "bad_epochs": bad_epochs,
                "has_selected_model": has_selected_model,
                "stop": bad_epochs >= config.EARLY_STOPPING_PATIENCE,
            }

        decision = broadcast_object(decision, context)
        if not isinstance(decision, dict):
            raise RuntimeError("Failed to broadcast model-selection decision")
        best_down_qwk = float(decision["best_down_qwk"])
        best_down_mae = float(decision["best_down_mae"])
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
                best_down_qwk=best_down_qwk,
                best_down_mae=best_down_mae,
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

    best_dev_output = evaluate_model(
        model,
        dev_loader,
        len(dev_frame),
        config,
        context,
        description=f"Seed {config.SEED} best Dev",
    )
    test_output = evaluate_model(
        model,
        test_loader,
        len(test_frame),
        config,
        context,
        description=f"Seed {config.SEED} Test",
    )
    seed_output: Optional[SeedRunOutput] = None
    if context.is_main:
        if best_dev_output is None or best_dev_output.metrics is None:
            raise RuntimeError("Rank 0 did not receive best-checkpoint Dev metrics")
        if test_output is None:
            raise RuntimeError("Rank 0 did not receive Test predictions")
        expected_dev_ids = (
            dev_frame.sort_values("_original_index")["_id"].astype(str).tolist()
        )
        expected_ids = test_frame.sort_values("_original_index")["_id"].astype(str).tolist()
        if best_dev_output.ids != expected_dev_ids:
            raise RuntimeError("Dev predictions are not in the original ID order")
        if test_output.ids != expected_ids:
            raise RuntimeError("Test predictions are not in the original ID order")
        dev_diagnostics = write_diagnostics(
            best_dev_output,
            config,
            split_name="dev",
        )
        test_diagnostics = write_diagnostics(
            test_output,
            config,
            split_name="test",
        )
        LOGGER.info(
            "Seed %d diagnostics: Dev=%s Test=%s",
            config.SEED,
            dev_diagnostics,
            test_diagnostics,
        )
        LOGGER.info(
            "Seed %d best-checkpoint Dev metrics: %s",
            config.SEED,
            best_dev_output.metrics,
        )
        if test_output.metrics is not None:
            LOGGER.info(
                "Seed %d Open-Test metrics (diagnostic only; never used for "
                "selection or weighting): %s",
                config.SEED,
                test_output.metrics,
            )
        seed_output = SeedRunOutput(config.SEED, best_dev_output, test_output)
    distributed_barrier(context)
    return seed_output


def train_multiseed_ensemble(
    train_frame: pd.DataFrame,
    dev_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    config: Config,
    context: DistributedContext,
    *,
    smoke_test: bool,
) -> None:
    """Train every configured seed and uniformly average its raw predictions."""

    seed_runs: list[SeedRunOutput] = []
    for member_index, seed in enumerate(config.ENSEMBLE_SEEDS, start=1):
        member_config = config_for_seed(config, seed)
        prepare_output_directories(member_config, context)
        seed_everything(seed + context.rank)
        if context.is_main:
            LOGGER.info(
                "Starting ensemble member %d/%d with seed=%d; artifacts=%s",
                member_index,
                len(config.ENSEMBLE_SEEDS),
                seed,
                member_config.resolve(member_config.OUTPUT_DIR),
            )
        member_output = train_select_and_predict(
            train_frame,
            dev_frame,
            test_frame,
            member_config,
            context,
            smoke_test=smoke_test,
        )
        if context.is_main:
            if member_output is None:
                raise RuntimeError(f"Seed {seed} produced no rank-0 output")
            seed_runs.append(member_output)
        distributed_barrier(context)
        gc.collect()
        if context.device.type == "cuda":
            torch.cuda.empty_cache()

    if context.is_main:
        if len(seed_runs) != len(config.ENSEMBLE_SEEDS):
            raise RuntimeError("The multi-seed ensemble is missing member outputs")
        ensemble_dev = average_evaluation_outputs(
            [run.dev for run in seed_runs],
            config,
        )
        ensemble_test = average_evaluation_outputs(
            [run.test for run in seed_runs],
            config,
        )
        dev_diagnostics = write_diagnostics(
            ensemble_dev,
            config,
            split_name="ensemble_dev",
        )
        test_diagnostics = write_diagnostics(
            ensemble_test,
            config,
            split_name="test",
        )
        report_path = write_ensemble_report(
            seed_runs,
            ensemble_dev,
            ensemble_test,
            config,
        )
        LOGGER.info("Uniform multi-seed ensemble Dev metrics: %s", ensemble_dev.metrics)
        LOGGER.info("Ensemble Dev diagnostics: %s", dev_diagnostics)
        LOGGER.info("Ensemble Test diagnostics: %s", test_diagnostics)
        LOGGER.info("Ensemble report: %s", report_path)
        if ensemble_test.metrics is not None:
            LOGGER.info(
                "Ensemble Open-Test metrics (diagnostic only; never used for "
                "selection or weighting): %s",
                ensemble_test.metrics,
            )
        create_submissions(
            ensemble_test.ids,
            ensemble_test.down_predictions,
            config,
        )
    distributed_barrier(context)


def run_pipeline(config: Config, context: DistributedContext, *, smoke_test: bool) -> None:
    """Validate data, preprocess all splits, and execute the strict workflow."""

    prepare_output_directories(config, context)
    if context.is_main:
        run_auxiliary_mapping_check()
        run_discretization_checks(config)
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
    # The BERT D3Tok disambiguator is large and no longer needed after all
    # cached views have been materialized. Reclaim cyclic Python objects before
    # the AraBERT encoder and Adam states are allocated.
    gc.collect()
    if context.device.type == "cuda":
        torch.cuda.empty_cache()

    run_sampler_checks()
    run_gather_order_check()
    run_ensemble_averaging_check(config)
    train_multiseed_ensemble(
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
    if config.MODEL_MODE not in {"baseline_mse", "cascaded_hmtl"}:
        raise ValueError(
            "MODEL_MODE must be 'baseline_mse' or 'cascaded_hmtl'"
        )
    positive_integer_fields = {
        "MAX_LENGTH": config.MAX_LENGTH,
        "NUM_EPOCHS": config.NUM_EPOCHS,
        "PER_DEVICE_BATCH_SIZE": config.PER_DEVICE_BATCH_SIZE,
        "EVAL_BATCH_SIZE": config.EVAL_BATCH_SIZE,
        "GRADIENT_ACCUMULATION_STEPS": config.GRADIENT_ACCUMULATION_STEPS,
        "EARLY_STOPPING_PATIENCE": config.EARLY_STOPPING_PATIENCE,
        "DDP_TIMEOUT_MINUTES": config.DDP_TIMEOUT_MINUTES,
        "LOG_EVERY_N_STEPS": config.LOG_EVERY_N_STEPS,
        "D3TOK_BATCH_SIZE": config.D3TOK_BATCH_SIZE,
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
    auxiliary_weights = {
        "AUX_CE3_WEIGHT": config.AUX_CE3_WEIGHT,
        "AUX_CE5_WEIGHT": config.AUX_CE5_WEIGHT,
        "AUX_CE7_WEIGHT": config.AUX_CE7_WEIGHT,
    }
    if any(
        not math.isfinite(weight) or weight < 0.0
        for weight in auxiliary_weights.values()
    ):
        raise ValueError(
            f"Auxiliary CE weights must be finite and non-negative: {auxiliary_weights}"
        )
    if not math.isfinite(config.CASCADE_TEMPERATURE) or config.CASCADE_TEMPERATURE <= 0.0:
        raise ValueError("CASCADE_TEMPERATURE must be finite and positive")
    if config.ENCODER_LR <= 0.0 or config.HEAD_LR <= 0.0:
        raise ValueError("ENCODER_LR and HEAD_LR must be positive")
    if config.WEIGHT_DECAY < 0.0 or config.MAX_GRAD_NORM <= 0.0:
        raise ValueError("WEIGHT_DECAY must be non-negative and MAX_GRAD_NORM positive")
    if config.SAMPLER_ALPHA < 0.0:
        raise ValueError("SAMPLER_ALPHA cannot be negative")
    if config.SUBMISSION_DOWN_BASENAME == config.SUBMISSION_DOWN_ZIP_NAME:
        raise ValueError("Down submission source and ZIP names must be distinct")
    if config.SUBMISSION_BASENAME != "prediction":
        raise ValueError("The official ZIP member must remain named 'prediction'")
    if not config.ENSEMBLE_SEEDS:
        raise ValueError("ENSEMBLE_SEEDS must contain at least one seed")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        for seed in config.ENSEMBLE_SEEDS
    ):
        raise ValueError("ENSEMBLE_SEEDS must contain non-negative integers")
    if len(set(config.ENSEMBLE_SEEDS)) != len(config.ENSEMBLE_SEEDS):
        raise ValueError("ENSEMBLE_SEEDS must not contain duplicates")
    if (
        config.RESUME_FROM_CHECKPOINT
        and len(config.ENSEMBLE_SEEDS) > 1
        and "{seed}" not in str(config.RESUME_FROM_CHECKPOINT)
    ):
        raise ValueError(
            "Multi-seed resume requires a {seed} placeholder in "
            "RESUME_FROM_CHECKPOINT"
        )


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
    seed_everything(config.ENSEMBLE_SEEDS[0] + context.rank)
    try:
        run_pipeline(config, context, smoke_test=args.smoke_test)
    finally:
        if context.distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
