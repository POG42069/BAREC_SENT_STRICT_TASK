#!/usr/bin/env python
"""Train the BAREC sentence baseline and predict the private 2026 Blind Test.

The Hugging Face access token is read from an environment variable, a Kaggle
Secret, an existing Hugging Face login, or an interactive hidden prompt.  It is
never accepted as a command-line value, stored in Config/checkpoints, or logged.

Typical Kaggle usage:

    # Add a Kaggle Secret named HF_TOKEN, enable it for the notebook, then run:
    python train_blind.py --download-only
    python train_blind.py --smoke-test
    python train_blind.py

This module intentionally reuses train.py for validation, D3Tok preprocessing,
training, model selection, inference, and submission validation so the Blind
pipeline cannot silently diverge from the Open-Test pipeline.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional, Sequence

def _private_runtime_root() -> Path:
    """Choose scratch storage that Kaggle does not publish as notebook output."""

    kaggle_scratch = Path("/kaggle/temp")
    project_root = Path(__file__).resolve().parent
    kaggle_output = Path("/kaggle/working").resolve()
    candidates = [kaggle_scratch, Path(tempfile.gettempdir()), Path("/tmp")]
    for scratch_parent in dict.fromkeys(candidates):
        if not scratch_parent.is_dir():
            continue
        runtime_root = (
            scratch_parent / "barec_2026_blind_sent_private"
        ).resolve()
        inside_project = (
            runtime_root == project_root or project_root in runtime_root.parents
        )
        inside_kaggle_output = kaggle_output.is_dir() and (
            runtime_root == kaggle_output or kaggle_output in runtime_root.parents
        )
        if not inside_project and not inside_kaggle_output:
            return runtime_root
    raise RuntimeError(
        "No private temporary directory exists outside the repository and "
        "Kaggle output tree. Configure an operating-system temp directory first."
    )


PRIVATE_DOWNLOAD_DIR = _private_runtime_root()

# Set these before importing train/Transformers/Datasets so even private Hub
# README/metadata files go to scratch storage instead of a persistent cache.
os.environ["HF_HUB_CACHE"] = str(PRIVATE_DOWNLOAD_DIR / "huggingface_hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
os.environ["HF_XET_CACHE"] = str(PRIVATE_DOWNLOAD_DIR / "huggingface_xet")
os.environ["HF_DATASETS_CACHE"] = str(PRIVATE_DOWNLOAD_DIR / "datasets_cache")
os.environ["HF_ASSETS_CACHE"] = str(PRIVATE_DOWNLOAD_DIR / "huggingface_assets")
# The Blind download passes its credential explicitly. Public model downloads
# after that point must not attach a cached Hugging Face token implicitly.
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import train as baseline  # noqa: E402


BLIND_DATASET_ID = "CAMeL-Lab/BAREC-Shared-Task-2026-BlindTest-sent"
LOCAL_BLIND_PATH_ENV = "BAREC_2026_BLIND_SENT_LOCAL_PATH"
KAGGLE_SECRET_BROKER_ENV = "KAGGLE_USER_SECRETS_TOKEN"
DEFAULT_TOKEN_ENV = "HF_TOKEN"
TOKEN_ENV_ALIASES = (
    "HF_TOKEN",
    "BAREC_HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


def private_blind_path(preferred_split: str) -> Path:
    """Key the private scratch table by dataset and requested split."""

    key = hashlib.sha256(
        f"{BLIND_DATASET_ID}\0{preferred_split}".encode("utf-8")
    ).hexdigest()[:16]
    return PRIVATE_DOWNLOAD_DIR / f"blind_test_{key}.parquet"


def parse_token_env_name(value: str) -> str:
    """Accept only known credential-variable names, never credential values."""

    if value not in TOKEN_ENV_ALIASES:
        raise argparse.ArgumentTypeError(
            "Use a supported Secret/environment-variable name; do not pass a token value."
        )
    return value


def parse_args() -> argparse.Namespace:
    """Parse the public CLI without ever accepting a token value."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run the real Blind pipeline on tiny Train/Dev/Blind subsets.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Validate private-dataset access and stop before training.",
    )
    parser.add_argument(
        "--refresh-blind",
        action="store_true",
        help="Redownload the private Blind split instead of reusing temporary scratch data.",
    )
    parser.add_argument(
        "--blind-split",
        default="test",
        help="Preferred Hugging Face split name; a sole differently named split is accepted.",
    )
    parser.add_argument(
        "--hf-token-env",
        default=DEFAULT_TOKEN_ENV,
        type=parse_token_env_name,
        help=(
            "Name of the environment variable/Kaggle Secret containing the token "
            "(the value itself must never be passed on the command line)."
        ),
    )
    parser.add_argument("--ddp-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank_argument",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _first_nonempty(values: Sequence[Optional[str]]) -> Optional[str]:
    for value in values:
        if value is not None and value.strip():
            return value.strip()
    return None


def resolve_hf_token(preferred_name: str) -> str:
    """Resolve a token without printing or persisting it."""

    if preferred_name not in TOKEN_ENV_ALIASES:
        raise ValueError(
            "Unsupported Secret/environment-variable name; token values are not accepted."
        )
    env_names = tuple(dict.fromkeys((preferred_name, *TOKEN_ENV_ALIASES)))
    token = _first_nonempty([os.environ.get(name) for name in env_names])
    if token:
        return token

    # Kaggle Secrets are preferred over putting credentials in notebook cells.
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

        client = UserSecretsClient()
        for name in env_names:
            try:
                token = _first_nonempty([client.get_secret(name)])
            except Exception:
                continue
            if token:
                return token
    except Exception:
        # This is expected outside Kaggle or when no matching Secret exists.
        pass

    # Respect an existing `huggingface-cli login` without exposing its value.
    try:
        from huggingface_hub import get_token

        token = _first_nonempty([get_token()])
        if token:
            return token
    except Exception:
        pass

    if sys.stdin.isatty():
        token = _first_nonempty(
            [getpass.getpass("Hugging Face access token (hidden input): ")]
        )
        if token:
            return token

    raise RuntimeError(
        "No Hugging Face access token is available. On Kaggle, create a Secret "
        f"named {preferred_name!r}, enable it for the notebook, and rerun. Never "
        "paste the token into this script, a notebook cell, a CLI argument, or Git."
    )


def scrub_token_environment(preferred_name: str) -> None:
    """Remove credential variables once the authenticated download is complete."""

    for name in dict.fromkeys((preferred_name, *TOKEN_ENV_ALIASES)):
        os.environ.pop(name, None)
    # Kaggle uses this broker credential to retrieve every enabled notebook
    # Secret. DDP/DataLoader children do not need that capability.
    os.environ.pop(KAGGLE_SECRET_BROKER_ENV, None)


def select_blind_split(dataset: Any, preferred_split: str) -> tuple[Any, str]:
    """Select the requested split, accepting a differently named sole split."""

    if not hasattr(dataset, "keys"):
        if not hasattr(dataset, "to_pandas"):
            raise TypeError("load_dataset returned an unsupported object")
        return dataset, preferred_split

    split_names = [str(name) for name in dataset.keys()]
    if preferred_split in split_names:
        return dataset[preferred_split], preferred_split
    if len(split_names) == 1:
        selected = split_names[0]
        baseline.LOGGER.warning(
            "Requested Blind split %r is absent; using the only available split %r.",
            preferred_split,
            selected,
        )
        return dataset[selected], selected
    raise ValueError(
        f"Blind dataset has splits {split_names}, but {preferred_split!r} was requested. "
        "Choose one with --blind-split."
    )


def strip_blind_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove any label-like columns so Blind labels can never enter the pipeline."""

    label_names = {name.casefold() for name in baseline.LABEL_ALIASES}
    label_names.update({"labels", "readability_level"})
    drop_columns = [
        column
        for column in frame.columns
        if str(column).casefold() in label_names
        or str(column).casefold().startswith("readability_level_")
    ]
    if drop_columns:
        baseline.LOGGER.warning(
            "Dropping label-like Blind columns before local materialization: %s",
            [str(column) for column in drop_columns],
        )
        frame = frame.drop(columns=drop_columns)
    return frame


def validate_local_blind(path: Path) -> tuple[int, list[str]]:
    """Validate the temporary private table without printing examples or IDs."""

    config = baseline.Config()
    frame = baseline.load_split(path, "blind", config, require_label=False)
    if bool(frame.attrs.get("has_labels")):
        raise RuntimeError("Blind cache unexpectedly contains a usable label column")
    public_columns = [
        str(column)
        for column in frame.columns
        if not str(column).startswith("_")
    ]
    return len(frame), public_columns


def materialize_blind_dataset(
    *,
    preferred_split: str,
    token_env_name: str,
    refresh: bool,
) -> Path:
    """Download once and atomically save outside the repository/output tree."""

    path = private_blind_path(preferred_split).resolve()
    if path.is_file() and not refresh:
        row_count, columns = validate_local_blind(path)
        baseline.LOGGER.info(
            "Reusing private Blind scratch data: %s (%d rows, columns=%s)",
            path,
            row_count,
            columns,
        )
        return path

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Run: python -m pip install -r requirements.txt"
        ) from exc

    token = resolve_hf_token(token_env_name)
    PRIVATE_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    hf_cache_dir = PRIVATE_DOWNLOAD_DIR / "huggingface_cache"
    baseline.LOGGER.info(
        "Downloading private BAREC 2026 Sentence Blind Test from Hugging Face."
    )
    try:
        loaded = load_dataset(
            BLIND_DATASET_ID,
            token=token,
            cache_dir=str(hf_cache_dir),
            download_mode="force_redownload" if refresh else "reuse_dataset_if_exists",
        )
    except Exception as exc:
        raise RuntimeError(
            "Blind Test download failed. Verify that the Kaggle Secret/environment "
            "token is current and has access to the private dataset. The token was "
            f"not logged. Underlying error type: {type(exc).__name__}."
        ) from None
    finally:
        # Drop this script's reference as soon as the authenticated call finishes.
        token = ""
        scrub_token_environment(token_env_name)

    split, selected_name = select_blind_split(loaded, preferred_split)
    if not hasattr(split, "to_pandas"):
        raise TypeError(f"Blind split {selected_name!r} cannot be converted to a table")
    frame = split.to_pandas()
    frame.columns = [str(column) for column in frame.columns]
    frame = strip_blind_labels(frame)

    config = baseline.Config()
    id_column = baseline.resolve_column(
        list(frame.columns),
        config.ID_COLUMN,
        baseline.ID_ALIASES,
        "ID",
        required=True,
    )
    text_column = baseline.resolve_column(
        list(frame.columns),
        config.TEXT_COLUMN,
        baseline.TEXT_ALIASES,
        "text",
        required=True,
    )
    document_column = baseline.resolve_column(
        list(frame.columns),
        "Document",
        baseline.DOCUMENT_ALIASES,
        "document",
        required=False,
    )
    assert id_column is not None and text_column is not None

    # Materialize only what inference/isolation validation needs. This both
    # canonicalizes aliases and prevents labels or unrelated private metadata
    # from reaching Config, checkpoints, diagnostics, or the model pipeline.
    canonical_data: dict[str, Any] = {
        config.ID_COLUMN: frame[id_column].astype("string"),
        config.TEXT_COLUMN: frame[text_column],
    }
    if document_column is not None:
        canonical_data["Document"] = frame[document_column]
    frame = pd.DataFrame(canonical_data)

    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.parquet")
    try:
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    row_count, columns = validate_local_blind(path)
    baseline.LOGGER.info(
        "Private Blind split %r materialized in non-output scratch storage: "
        "%s (%d rows, columns=%s)",
        selected_name,
        path,
        row_count,
        columns,
    )
    return path


def make_blind_config(blind_path: Path, *, smoke_test: bool) -> baseline.Config:
    """Create a token-free Config with isolated Blind artifacts."""

    config = baseline.Config()
    if smoke_test:
        config.enable_smoke_mode()
        config.OUTPUT_DIR = "outputs/blind/smoke"
        config.CHECKPOINT_DIR = "outputs/blind/smoke/checkpoints"
        config.CACHE_DIR = str(PRIVATE_DOWNLOAD_DIR / "preprocessed_smoke")
        config.SUBMISSION_DIR = "outputs/blind/smoke"
        config.SEED_RUNS_DIR = "outputs/blind/smoke/seeds"
    else:
        config.OUTPUT_DIR = "outputs/blind"
        config.CHECKPOINT_DIR = "outputs/blind/checkpoints"
        config.CACHE_DIR = str(PRIVATE_DOWNLOAD_DIR / "preprocessed")
        config.SUBMISSION_DIR = "outputs/blind"
        config.SEED_RUNS_DIR = "outputs/blind/seeds"
    config.TEST_PATH = str(blind_path.resolve())
    return config


def maybe_self_launch_ddp(args: argparse.Namespace, blind_path: Path) -> bool:
    """Relaunch this Blind script, not train.py, on two visible GPUs."""

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

    child_environment = os.environ.copy()
    child_environment[LOCAL_BLIND_PATH_ENV] = str(blind_path.resolve())
    # Rank 0 owns the potentially long D3Tok cache build while rank 1 waits.
    # Keep the NCCL monitor from treating that expected wait as a deadlock.
    child_environment.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600")
    # Workers only need the already materialized Parquet file. Do not propagate
    # the private credential unnecessarily.
    for name in dict.fromkeys((args.hf_token_env, *TOKEN_ENV_ALIASES)):
        child_environment.pop(name, None)
    child_environment.pop(KAGGLE_SECRET_BROKER_ENV, None)

    print("Detected at least two GPUs; launching the private Blind pipeline with DDP:")
    print(" ".join(command))
    subprocess.run(
        command,
        check=True,
        cwd=str(baseline.SCRIPT_DIR),
        env=child_environment,
    )
    return True


def main() -> None:
    """Download once, then run the unchanged strict sentence pipeline on Blind."""

    args = parse_args()
    if args.local_rank_argument is not None:
        os.environ.setdefault("LOCAL_RANK", str(args.local_rank_argument))

    worker_path = os.environ.get(LOCAL_BLIND_PATH_ENV)
    if args.ddp_worker or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        if not worker_path:
            raise RuntimeError(
                "Blind DDP workers require a path prepared by `python train_blind.py`; "
                "do not invoke --ddp-worker or torchrun manually."
            )
        blind_path = Path(worker_path).resolve()
    else:
        try:
            blind_path = materialize_blind_dataset(
                preferred_split=args.blind_split,
                token_env_name=args.hf_token_env,
                refresh=args.refresh_blind or args.download_only,
            )
        finally:
            # Public model/CAMeL downloads do not need the private credential.
            scrub_token_environment(args.hf_token_env)
        os.environ[LOCAL_BLIND_PATH_ENV] = str(blind_path)

    if args.download_only:
        row_count, columns = validate_local_blind(blind_path)
        print(
            "Blind Test access and schema validation succeeded: "
            f"{row_count} rows, columns={columns}. Private data remains only in "
            "non-output temporary storage."
        )
        return

    if maybe_self_launch_ddp(args, blind_path):
        return

    config = make_blind_config(blind_path, smoke_test=args.smoke_test)
    baseline.validate_config(config)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    context = baseline.initialize_distributed(config)
    baseline.seed_everything(config.SEED + context.rank)
    try:
        baseline.run_pipeline(config, context, smoke_test=args.smoke_test)
    finally:
        if context.distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
