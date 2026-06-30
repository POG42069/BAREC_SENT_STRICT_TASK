"""
BAREC SENT STRICT baseline inspired by the !MSA ensemble approach.

Run:
    python train.py

The file intentionally contains the full pipeline: data loading, Arabic
preprocessing, model/loss definitions, training, prediction, ensembling, and
submission packaging.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

TASK_LEVEL = "sentence"
TRACK = "strict"

TRAIN_PATH = "data/barec-corpus-v1/train.csv"
DEV_PATH = "data/barec-corpus-v1/dev.csv"
TEST_PATH = "data/barec-corpus-v1/test.csv"

import os
if os.path.exists("/content/drive/MyDrive"):
    OUTPUT_DIR = "/content/drive/MyDrive/BAREC_Outputs"
elif os.path.exists("/kaggle/working"):
    OUTPUT_DIR = "/kaggle/working/BAREC_Outputs"
else:
    OUTPUT_DIR = "./outputs"
SUBMISSION_DIR = "./submission"

ID_COLUMN = "Sentence ID"
TEXT_COLUMN = "Sentence"
LABEL_COLUMN = "Readability_Level_19"

NUM_LABELS = 19
MIN_LABEL = 1
MAX_LABEL = 19

EPOCHS = 5
LEARNING_RATE = 2e-5
PER_GPU_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
MAX_LENGTH = 256
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 2
MAX_GRAD_NORM = 1.0
USE_FP16 = True

USE_D3TOK = True
USE_CLASS_WEIGHTS = True
USE_CONFIDENCE_WEIGHTED_ENSEMBLE = True
USE_POST_PROCESSING = True
RETRAIN_IF_EXISTS = False
USE_DDP_AUTO = True
DDP_MASTER_PORT = "12355"

DOC_ID_PREFIX_LENGTH = 7

# Optional debug limits. Leave as None for real training.
MAX_TRAIN_ROWS: Optional[int] = None
MAX_DEV_ROWS: Optional[int] = None
MAX_TEST_ROWS: Optional[int] = None

# Checkpoints were verified on Hugging Face model hub on 2026-06-24.
MODEL_REGISTRY = {
    "arabertv2": "aubmindlab/bert-base-arabertv02",
    "araelectra": "aubmindlab/araelectra-base-discriminator",
    "marbert": "UBC-NLP/MARBERT",
    "camelbert": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
}

VALID_LOSSES = {"ce", "mse", "cor"}

ENSEMBLE_COMBOS = [
    ("arabertv2", "ce"),
    ("arabertv2", "cor"),
    ("araelectra", "mse"),
    ("camelbert", "ce"),
    ("camelbert", "mse"),
    ("marbert", "cor"),
]


# ============================================================
# UTILITIES
# ============================================================


def is_rank_zero(rank: int) -> bool:
    return rank == 0


def log(message: str, rank: int = 0) -> None:
    if is_rank_zero(rank):
        print(message, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed(rank: int, world_size: int) -> None:
    """Initialize torch.distributed for single-node multi-GPU training."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", DDP_MASTER_PORT)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    """Shut down the distributed process group if it was initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(world_size: int) -> None:
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()


def broadcast_bool(value: bool, rank: int, world_size: int) -> bool:
    if world_size <= 1:
        return value
    tensor = torch.tensor([1 if value else 0], dtype=torch.long)
    if torch.cuda.is_available():
        tensor = tensor.cuda(rank)
    dist.broadcast(tensor, src=0)
    return bool(tensor.item())


def get_device(rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}" if torch.cuda.device_count() > 1 else "cuda")
    warnings.warn("No GPU detected. The pipeline will run on CPU and will be much slower.")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clamp_labels(values: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=float(MIN_LABEL))
    arr = np.rint(arr).astype(int)
    return np.clip(arr, MIN_LABEL, MAX_LABEL)


# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================


def clean_arabic_text(text: str) -> str:
    """
    Apply conservative Arabic text normalization.

    The function removes formatting noise but avoids aggressive normalization
    that could erase readability clues. It keeps Arabic letters intact, removes
    tatweel, normalizes a few common Arabic variants, collapses repeated
    punctuation, and trims extra whitespace.
    """
    text = "" if text is None else str(text)
    text = text.replace("\u0640", "")  # Tatweel/kashida elongation mark.
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)  # Arabic diacritics.
    replacements = {
        "إ": "ا",
        "أ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
        
    # 5 Advanced Arabert NLP techniques
    text = text.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')) # 1. Number normalization
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)           # 2. URLs
    text = re.sub(r'@[A-Za-z0-9_]+', '[USER]', text)                 # 3. Mentions
    text = re.sub(r'([ا-ي])\1{2,}', r'\1\1', text)                    # 4. Character repetitions
    text = re.sub(r'(\S)(/)(\S)', r'\1 \2 \3', text)                 # 5. Slash spacing
    
    text = re.sub(r"([!?؟،,.؛:])\1{1,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_D3TOK_TOKENIZER = None
_D3TOK_AVAILABLE = None


def apply_d3tok(text: str) -> str:
    """
    Optionally apply CAMeL Tools D3 tokenization.

    !MSA used D3TOK-style morphological tokenization. Installing camel-tools can
    be heavy on Kaggle, so this baseline falls back to the cleaned text with a
    clear warning when CAMeL Tools is unavailable.
    """
    global _D3TOK_TOKENIZER, _D3TOK_AVAILABLE
    if not USE_D3TOK:
        return text
    if _D3TOK_AVAILABLE is False:
        return text
    if _D3TOK_TOKENIZER is None:
        try:
            from camel_tools.disambig.mle import MLEDisambiguator
            from camel_tools.tokenizers.morphological import MorphologicalTokenizer

            disambiguator = MLEDisambiguator.pretrained()
            _D3TOK_TOKENIZER = MorphologicalTokenizer(disambiguator, scheme="d3tok")
            _D3TOK_AVAILABLE = True
        except Exception as exc:  # pragma: no cover - depends on optional package.
            warnings.warn(
                "USE_D3TOK=True but CAMeL Tools D3TOK could not be initialized. "
                f"Falling back to cleaned text. Reason: {exc}"
            )
            _D3TOK_AVAILABLE = False
            return text
    try:
        return " ".join(_D3TOK_TOKENIZER.tokenize(text.split()))
    except Exception as exc:  # pragma: no cover - tokenizer runtime fallback.
        warnings.warn(f"D3TOK failed for one sample; using cleaned text. Reason: {exc}")
        return text


def preprocess_text(text: str) -> str:
    return apply_d3tok(clean_arabic_text(text))


def load_table(path: str) -> pd.DataFrame:
    """
    Load CSV, TSV, or Parquet data with an explicit file existence check.
    """
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}. Update TRAIN_PATH/DEV_PATH/TEST_PATH "
            "in the CONFIGURATION section at the top of train.py."
        )
    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(data_path)
    if suffix == ".tsv":
        return pd.read_csv(data_path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(data_path)
    raise ValueError(f"Unsupported file type for {data_path}. Use .csv, .tsv, or .parquet.")


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    exact = {column: column for column in df.columns}
    lower = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def infer_columns(df: pd.DataFrame, split_name: str) -> Dict[str, Optional[str]]:
    """
    Infer ID, text, and optional label columns for a split.

    If inference fails, edit ID_COLUMN/TEXT_COLUMN/LABEL_COLUMN in the config.
    Test splits may omit labels.
    """
    id_candidates = [ID_COLUMN, "Sentence ID", "sentence_id", "id", "ID"]
    text_candidates = [TEXT_COLUMN, "Sentence", "sentence", "text", "Text", "content"]
    label_candidates = [
        LABEL_COLUMN,
        "Readability_Level_19",
        "Prediction",
        "label",
        "Label",
        "readability",
        "Readability",
        "readability_level",
        "Readability_Level",
    ]
    id_column = first_existing_column(df, id_candidates)
    text_column = first_existing_column(df, text_candidates)
    label_column = first_existing_column(df, label_candidates)
    if id_column is None:
        raise ValueError(f"Could not infer ID column for {split_name}. Columns: {list(df.columns)}")
    if text_column is None:
        raise ValueError(f"Could not infer text column for {split_name}. Columns: {list(df.columns)}")
    if split_name in {"train", "dev"} and label_column is None:
        raise ValueError(
            f"Could not infer label column for {split_name}. Set LABEL_COLUMN near the top of train.py."
        )
    return {"id": id_column, "text": text_column, "label": label_column}


def parse_label(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("Encountered missing label.")
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if not match:
            raise ValueError(f"Could not parse label from value: {value!r}")
        value = match.group(0)
    label = int(float(value))
    if label < MIN_LABEL or label > MAX_LABEL:
        raise ValueError(f"Label {label} is outside [{MIN_LABEL}, {MAX_LABEL}].")
    return label


def compute_class_weights(labels: list[int], num_labels: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights used by !MSA.

    Formula: w_j = n_samples / (n_classes * n_samples_in_class_j).
    Missing classes receive weight 1.0 to avoid division by zero. We clip very
    large weights for stability on tiny/debug subsets.
    """
    counts = np.bincount([label - MIN_LABEL for label in labels], minlength=num_labels)
    total = len(labels)
    weights = np.ones(num_labels, dtype=np.float32)
    for index, count in enumerate(counts):
        if count > 0:
            weights[index] = total / float(num_labels * count)
    weights = np.clip(weights, 0.05, 10.0)
    return torch.tensor(weights, dtype=torch.float32)


def prepare_split(
    df: pd.DataFrame,
    columns: Dict[str, Optional[str]],
    split_name: str,
    max_rows: Optional[int],
    rank: int,
) -> pd.DataFrame:
    if max_rows is not None:
        df = df.head(max_rows).copy()
    else:
        df = df.copy()
    log(f"Preprocessing {split_name}: {len(df)} rows", rank)
    df["_id"] = df[columns["id"]].astype(str)
    df["_text"] = df[columns["text"]].map(preprocess_text)
    if columns["label"] is not None:
        df["_label"] = df[columns["label"]].map(parse_label)
    return df


def validate_config_and_data(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_cols: Dict[str, Optional[str]],
    dev_cols: Dict[str, Optional[str]],
    test_cols: Dict[str, Optional[str]],
) -> None:
    if PER_GPU_BATCH_SIZE <= 0:
        raise ValueError("PER_GPU_BATCH_SIZE must be positive.")
    for model_key, loss_type in ENSEMBLE_COMBOS:
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key in ENSEMBLE_COMBOS: {model_key}")
        if loss_type not in VALID_LOSSES:
            raise ValueError(f"Unknown loss type in ENSEMBLE_COMBOS: {loss_type}")
    for split_name, df, cols in [
        ("train", train_df, train_cols),
        ("dev", dev_df, dev_cols),
        ("test", test_df, test_cols),
    ]:
        if cols["id"] is None or cols["text"] is None:
            raise ValueError(f"{split_name} must have ID and text columns.")
        if split_name in {"train", "dev"} and cols["label"] is None:
            raise ValueError(f"{split_name} must have labels.")
    for split_name, df in [("train", train_df), ("dev", dev_df)]:
        labels = df["_label"].tolist()
        if not all(MIN_LABEL <= label <= MAX_LABEL for label in labels):
            raise ValueError(f"{split_name} labels must be in [{MIN_LABEL}, {MAX_LABEL}].")


# ============================================================
# DATASET AND MODEL
# ============================================================


class ReadabilityDataset(Dataset):
    """
    Torch dataset that tokenizes Arabic text for readability prediction.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        has_labels: bool,
        class_weights: Optional[torch.Tensor],
    ) -> None:
        self.ids = df["_id"].astype(str).tolist()
        self.texts = df["_text"].astype(str).tolist()
        self.has_labels = has_labels and "_label" in df.columns
        self.labels = df["_label"].astype(int).tolist() if self.has_labels else [MIN_LABEL] * len(df)
        self.row_indices = list(range(len(df)))
        self.tokenizer = tokenizer
        self.class_weights = class_weights.cpu() if class_weights is not None else torch.ones(NUM_LABELS)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        encoded = self.tokenizer(
            self.texts[index],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        label = int(self.labels[index])
        label_index = label - MIN_LABEL
        sample_weight = float(self.class_weights[label_index].item())
        item = {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "label_index": torch.tensor(label_index, dtype=torch.long),
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float),
            "row_index": torch.tensor(self.row_indices[index], dtype=torch.long),
            "sample_id": self.ids[index],
        }
        if "token_type_ids" in encoded:
            item["token_type_ids"] = torch.tensor(encoded["token_type_ids"], dtype=torch.long)
        return item


class ReadabilityModel(nn.Module):
    """
    Transformer wrapper supporting CE, MSE, and COR/CORAL losses.
    """

    def __init__(self, checkpoint_name: str, loss_type: str, class_weights: Optional[torch.Tensor]) -> None:
        super().__init__()
        self.checkpoint_name = checkpoint_name
        self.loss_type = loss_type
        self.encoder = AutoModel.from_pretrained(checkpoint_name)
        hidden_size = self.encoder.config.hidden_size
        output_size = NUM_LABELS if loss_type == "ce" else 1
        self.dropout = nn.Dropout(getattr(self.encoder.config, "hidden_dropout_prob", 0.1))
        self.head = nn.Linear(hidden_size, output_size, bias=(loss_type != "cor"))
        if loss_type == "cor":
            self.coral_bias = nn.Parameter(torch.zeros(NUM_LABELS - 1))
        self.register_buffer(
            "class_weights",
            class_weights.clone().detach().float() if class_weights is not None else torch.ones(NUM_LABELS),
        )

    def pool(self, outputs: Any) -> torch.Tensor:
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_indices: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**model_inputs)
        pooled = self.dropout(self.pool(outputs))
        if self.loss_type == "cor":
            logits = self.head(pooled) + self.coral_bias
        else:
            logits = self.head(pooled)
        result = {"logits": logits}
        if labels is None:
            return result

        if self.loss_type == "ce":
            weights = self.class_weights if USE_CLASS_WEIGHTS else None
            result["loss"] = F.cross_entropy(logits, label_indices, weight=weights)
        elif self.loss_type == "mse":
            targets = labels.float()
            preds = logits.squeeze(-1)
            weights = sample_weights if USE_CLASS_WEIGHTS else torch.ones_like(targets)
            result["loss"] = (weights * (preds - targets) ** 2).mean()
        elif self.loss_type == "cor":
            thresholds = torch.arange(MIN_LABEL + 1, MAX_LABEL + 1, device=labels.device)
            ordinal_targets = (labels.unsqueeze(1) >= thresholds.unsqueeze(0)).float()
            loss_matrix = F.binary_cross_entropy_with_logits(logits, ordinal_targets, reduction="none")
            if USE_CLASS_WEIGHTS and sample_weights is not None:
                loss_matrix = loss_matrix * sample_weights.unsqueeze(1)
            result["loss"] = loss_matrix.mean()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        return result

    def predict_from_logits(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.loss_type == "ce":
            probs = F.softmax(logits, dim=-1)
            confidence, pred_index = probs.max(dim=-1)
            pred_label = pred_index + MIN_LABEL
            pred_score = (probs * torch.arange(MIN_LABEL, MAX_LABEL + 1, device=logits.device).float()).sum(dim=-1)
            return pred_label.float(), pred_score, confidence
        if self.loss_type == "mse":
            raw_score = logits.squeeze(-1)
            rounded = torch.round(raw_score).clamp(MIN_LABEL, MAX_LABEL)
            confidence = 1.0 / (1.0 + torch.abs(raw_score - rounded))
            return rounded, raw_score, confidence
        probs = torch.sigmoid(logits)
        pred_label = MIN_LABEL + (probs > 0.5).sum(dim=-1)
        pred_label = pred_label.clamp(MIN_LABEL, MAX_LABEL).float()
        pred_score = MIN_LABEL + probs.sum(dim=-1)
        confidence = (torch.abs(probs - 0.5) * 2.0).mean(dim=-1)
        return pred_label, pred_score, confidence


def save_model_checkpoint(
    model: nn.Module,
    tokenizer: Any,
    checkpoint_dir: Path,
    model_key: str,
    loss_type: str,
    best_qwk: float,
) -> None:
    ensure_dir(checkpoint_dir)
    model_to_save = model.module if isinstance(model, DDP) else model
    torch.save(model_to_save.state_dict(), checkpoint_dir / "model_state.pt")
    tokenizer.save_pretrained(checkpoint_dir)
    metadata = {
        "model_key": model_key,
        "checkpoint_name": MODEL_REGISTRY[model_key],
        "loss_type": loss_type,
        "best_qwk": best_qwk,
        "num_labels": NUM_LABELS,
    }
    (checkpoint_dir / "model_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_model_checkpoint(
    checkpoint_dir: Path,
    model_key: str,
    loss_type: str,
    class_weights: torch.Tensor,
    device: torch.device,
) -> Tuple[ReadabilityModel, Any]:
    tokenizer_source = checkpoint_dir if (checkpoint_dir / "tokenizer_config.json").exists() else MODEL_REGISTRY[model_key]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    model = ReadabilityModel(MODEL_REGISTRY[model_key], loss_type, class_weights)
    state_path = checkpoint_dir / "model_state.pt"
    if state_path.exists():
        model.load_state_dict(torch.load(state_path, map_location=device))
    model.to(device)
    return model, tokenizer


# ============================================================
# TRAINING AND PREDICTION
# ============================================================


def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def build_loader(
    dataset: Dataset,
    batch_size: int,
    rank: int,
    world_size: int,
    shuffle: bool,
    for_training: bool,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return loader, sampler


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    return {
        "qwk": float(cohen_kappa_score(y_true_arr, y_pred_arr, weights="quadratic")),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "adjacent_accuracy": float(np.mean(np.abs(y_true_arr - y_pred_arr) <= 1)),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
    }


def predict_from_model(
    model: nn.Module,
    dataset: ReadabilityDataset,
    rank: int,
    world_size: int,
    device: torch.device,
    split_name: str,
) -> Optional[pd.DataFrame]:
    """
    Run distributed inference and return the gathered prediction frame on rank 0.
    """
    loader, _ = build_loader(dataset, PER_GPU_BATCH_SIZE, rank, world_size, shuffle=False, for_training=False)
    model.eval()
    local_rows: List[Dict[str, Any]] = []
    model_for_prediction = model.module if isinstance(model, DDP) else model
    progress = tqdm(loader, desc=f"Predict {split_name}", disable=not is_rank_zero(rank))
    with torch.no_grad():
        for batch in progress:
            sample_ids = batch["sample_id"]
            row_indices = batch["row_index"].cpu().numpy().tolist()
            batch = batch_to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
            pred_label, pred_score, confidence = model_for_prediction.predict_from_logits(outputs["logits"])
            for offset, sample_id in enumerate(sample_ids):
                local_rows.append(
                    {
                        "row_index": int(row_indices[offset]),
                        "Sentence ID": sample_id,
                        "pred_label": int(clamp_labels([pred_label[offset].item()])[0]),
                        "pred_score": float(pred_score[offset].detach().cpu().item()),
                        "confidence": float(confidence[offset].detach().cpu().item()),
                    }
                )

    if world_size > 1:
        gathered: List[List[Dict[str, Any]]] = [None for _ in range(world_size)]  # type: ignore[list-item]
        dist.all_gather_object(gathered, local_rows)
        if not is_rank_zero(rank):
            return None
        rows = [row for worker_rows in gathered for row in worker_rows]
    else:
        rows = local_rows
    pred_df = pd.DataFrame(rows)
    pred_df = pred_df.drop_duplicates(subset=["row_index"]).sort_values("row_index").reset_index(drop=True)
    return pred_df.drop(columns=["row_index"])


def train_one_combo(
    model_key: str,
    loss_type: str,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_weights: torch.Tensor,
    rank: int,
    world_size: int,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Train one model/loss combination, save checkpoint, and write predictions.
    """
    device = get_device(rank)
    combo_name = f"{model_key}_{loss_type}"
    checkpoint_dir = Path(OUTPUT_DIR) / "checkpoints" / combo_name
    prediction_dir = ensure_dir(Path(OUTPUT_DIR) / "predictions")
    resume_path = checkpoint_dir / "resume_state.pt"

    if checkpoint_dir.exists() and (checkpoint_dir / "model_state.pt").exists() and not RETRAIN_IF_EXISTS and not resume_path.exists():
        log(f"[{combo_name}] Existing checkpoint found; skipping training.", rank)
        model, tokenizer = load_model_checkpoint(checkpoint_dir, model_key, loss_type, class_weights, device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REGISTRY[model_key])
        model = ReadabilityModel(MODEL_REGISTRY[model_key], loss_type, class_weights).to(device)
        if world_size > 1:
            model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        train_dataset = ReadabilityDataset(train_df, tokenizer, has_labels=True, class_weights=class_weights)
        dev_dataset = ReadabilityDataset(dev_df, tokenizer, has_labels=True, class_weights=class_weights)
        train_loader, train_sampler = build_loader(
            train_dataset, PER_GPU_BATCH_SIZE, rank, world_size, shuffle=True, for_training=True
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        steps_per_epoch = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)
        total_steps = max(1, EPOCHS * steps_per_epoch)
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16 and torch.cuda.is_available())

        best_qwk = -1.0
        bad_epochs = 0
        start_epoch = 0
        if resume_path.exists():
            log(f"[{combo_name}] Resuming training from {resume_path}", rank)
            checkpoint = torch.load(resume_path, map_location=device)
            model_to_load = model.module if isinstance(model, DDP) else model
            model_to_load.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_qwk = checkpoint["best_qwk"]
            bad_epochs = checkpoint["bad_epochs"]

        for epoch in range(start_epoch, EPOCHS):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            progress = tqdm(train_loader, desc=f"{combo_name} epoch {epoch + 1}", disable=not is_rank_zero(rank))
            for step, batch in enumerate(progress, start=1):
                batch = batch_to_device(batch, device)
                with torch.cuda.amp.autocast(enabled=USE_FP16 and torch.cuda.is_available()):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_type_ids=batch.get("token_type_ids"),
                        labels=batch["label"],
                        label_indices=batch["label_index"],
                        sample_weights=batch["sample_weight"],
                    )
                    loss = outputs["loss"] / GRADIENT_ACCUMULATION_STEPS
                scaler.scale(loss).backward()
                running_loss += float(loss.detach().cpu().item()) * GRADIENT_ACCUMULATION_STEPS
                if step % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                if is_rank_zero(rank):
                    progress.set_postfix(loss=running_loss / step)

            dev_pred_df = predict_from_model(model, dev_dataset, rank, world_size, device, "dev")
            improved = False
            stop_now = False
            if is_rank_zero(rank) and dev_pred_df is not None:
                metrics = compute_metrics(dev_df["_label"].tolist(), dev_pred_df["pred_label"].tolist())
                log(
                    f"[{combo_name}] epoch={epoch + 1} "
                    f"qwk={metrics['qwk']:.5f} acc={metrics['accuracy']:.5f} "
                    f"adj_acc={metrics['adjacent_accuracy']:.5f} mae={metrics['mae']:.5f}",
                    rank,
                )
                improved = metrics["qwk"] > best_qwk
                if improved:
                    best_qwk = metrics["qwk"]
                    bad_epochs = 0
                    save_model_checkpoint(model, tokenizer, checkpoint_dir, model_key, loss_type, best_qwk)
                else:
                    bad_epochs += 1
                stop_now = bad_epochs >= EARLY_STOPPING_PATIENCE
                
                resume_state = {
                    "epoch": epoch,
                    "model_state_dict": (model.module if isinstance(model, DDP) else model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_qwk": best_qwk,
                    "bad_epochs": bad_epochs,
                }
                torch.save(resume_state, resume_path)
            stop_now = broadcast_bool(stop_now, rank, world_size)
            distributed_barrier(world_size)
            if stop_now:
                log(f"[{combo_name}] Early stopping.", rank)
                break

        distributed_barrier(world_size)
        if is_rank_zero(rank) and resume_path.exists():
            resume_path.unlink()
        model_for_load = model.module if isinstance(model, DDP) else model
        state_path = checkpoint_dir / "model_state.pt"
        if state_path.exists():
            model_for_load.load_state_dict(torch.load(state_path, map_location=device))

    if not isinstance(model, DDP) and world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    dev_dataset = ReadabilityDataset(dev_df, tokenizer, has_labels=True, class_weights=class_weights)
    test_dataset = ReadabilityDataset(test_df, tokenizer, has_labels="_label" in test_df.columns, class_weights=class_weights)
    dev_pred_df = predict_from_model(model, dev_dataset, rank, world_size, device, "dev")
    test_pred_df = predict_from_model(model, test_dataset, rank, world_size, device, "test")
    if is_rank_zero(rank):
        assert dev_pred_df is not None and test_pred_df is not None
        dev_pred_df.to_csv(prediction_dir / f"dev_{combo_name}.csv", index=False)
        test_pred_df.to_csv(prediction_dir / f"test_{combo_name}.csv", index=False)
    return dev_pred_df, test_pred_df


def predict_one_combo(*args: Any, **kwargs: Any) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Compatibility wrapper for combo inference after training or checkpoint load.
    """
    return train_one_combo(*args, **kwargs)


# ============================================================
# ENSEMBLE AND SUBMISSION
# ============================================================


def post_process_predictions(predictions: Sequence[float], fallback_label: int) -> np.ndarray:
    values = np.asarray(predictions, dtype=float)
    values = np.nan_to_num(values, nan=float(fallback_label))
    return clamp_labels(values)


def confidence_weighted_ensemble(
    prediction_frames: Sequence[pd.DataFrame],
    fallback_label: int,
) -> pd.DataFrame:
    """
    Ensemble model scores with confidence weights.

    If confidence weighting is disabled, this becomes a simple arithmetic mean
    over predicted scores.
    """
    if not prediction_frames:
        raise ValueError("No prediction frames were provided for ensembling.")
    base_ids = prediction_frames[0]["Sentence ID"].astype(str).tolist()
    score_stack = []
    confidence_stack = []
    for frame in prediction_frames:
        if frame["Sentence ID"].astype(str).tolist() != base_ids:
            raise ValueError("Prediction frames have different sample ordering or IDs.")
        score_stack.append(frame["pred_score"].astype(float).to_numpy())
        confidence_stack.append(frame["confidence"].astype(float).to_numpy())
    scores = np.vstack(score_stack)
    confidences = np.vstack(confidence_stack)
    if USE_CONFIDENCE_WEIGHTED_ENSEMBLE:
        confidences = np.clip(np.nan_to_num(confidences, nan=0.0), 1e-6, None)
        ensemble_score = np.sum(scores * confidences, axis=0) / np.sum(confidences, axis=0)
    else:
        ensemble_score = np.nanmean(scores, axis=0)
    final_pred = post_process_predictions(ensemble_score, fallback_label)
    return pd.DataFrame(
        {
            "Sentence ID": base_ids,
            "pred_score": ensemble_score,
            "pred_label": final_pred.astype(int),
            "confidence": np.mean(confidences, axis=0),
        }
    )


def aggregate_document_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    if TASK_LEVEL != "document":
        return pred_df
    df = pred_df.copy()
    df["Document ID"] = df["Sentence ID"].astype(str).str[:DOC_ID_PREFIX_LENGTH]
    aggregated = df.groupby("Document ID", as_index=False)["pred_label"].max()
    return aggregated.rename(columns={"Document ID": "Sentence ID"})


def create_submission(test_ids: Sequence[str], final_predictions: Sequence[int], submission_dir: str) -> Path:
    """
    Create the official BAREC submission file and prediction.zip archive.
    """
    submission_path = ensure_dir(submission_dir)
    prediction_file = submission_path / "prediction"
    zip_file = submission_path / "prediction.zip"
    predictions = clamp_labels(final_predictions)
    if len(test_ids) != len(predictions):
        raise ValueError(f"Expected {len(test_ids)} predictions, received {len(predictions)}.")
    if not all(MIN_LABEL <= int(pred) <= MAX_LABEL for pred in predictions):
        raise ValueError("All predictions must be integers in [1, 19].")

    output_df = pd.DataFrame({"Sentence ID": list(test_ids), "Prediction": predictions.astype(int)})
    output_df.to_csv(prediction_file, index=False)
    if zip_file.exists():
        zip_file.unlink()
    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(prediction_file, arcname="prediction")
    validate_submission(prediction_file, zip_file, expected_rows=len(test_ids))
    print(f"Submission zip written to: {zip_file.resolve()}")
    return zip_file


def validate_submission(prediction_file: Path, zip_file: Path, expected_rows: int) -> None:
    if not prediction_file.exists():
        raise FileNotFoundError(f"Missing submission file: {prediction_file}")
    if not zip_file.exists():
        raise FileNotFoundError(f"Missing submission zip: {zip_file}")
    with zipfile.ZipFile(zip_file, "r") as archive:
        names = archive.namelist()
        if names != ["prediction"]:
            raise ValueError(f"prediction.zip must contain exactly ['prediction']; found {names}")
        if "prediction.csv" in names:
            raise ValueError("prediction.zip must not contain prediction.csv")
    df = pd.read_csv(prediction_file)
    if list(df.columns) != ["Sentence ID", "Prediction"]:
        raise ValueError("Submission header must be: Sentence ID,Prediction")
    if len(df) != expected_rows:
        raise ValueError(f"Submission row count {len(df)} != test row count {expected_rows}")
    parsed = df["Prediction"].map(parse_label).tolist()
    if not all(MIN_LABEL <= label <= MAX_LABEL for label in parsed):
        raise ValueError("Submission predictions must be in [1, 19].")


# ============================================================
# MAIN PIPELINE
# ============================================================


def run_pipeline(rank: int, world_size: int) -> None:
    """
    Main end-to-end pipeline for one process.

    With multiple GPUs, python train.py launches one process per GPU and DDP
    divides each per-GPU batch evenly. Global batch size is:
    PER_GPU_BATCH_SIZE * number_of_gpus * GRADIENT_ACCUMULATION_STEPS.
    """
    if world_size > 1:
        setup_distributed(rank, world_size)
    set_seed(SEED + rank)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        global_batch_size = PER_GPU_BATCH_SIZE * gpu_count * GRADIENT_ACCUMULATION_STEPS
        log(f"Detected {gpu_count} GPU(s). Global batch size: {global_batch_size}", rank)
    else:
        log("No GPU detected. CPU fallback is available but will be slow.", rank)

    ensure_dir(OUTPUT_DIR)
    ensure_dir(Path(OUTPUT_DIR) / "checkpoints")
    ensure_dir(Path(OUTPUT_DIR) / "predictions")

    train_raw = load_table(TRAIN_PATH)
    dev_raw = load_table(DEV_PATH)
    test_raw = load_table(TEST_PATH)
    train_cols = infer_columns(train_raw, "train")
    dev_cols = infer_columns(dev_raw, "dev")
    test_cols = infer_columns(test_raw, "test")

    train_df = prepare_split(train_raw, train_cols, "train", MAX_TRAIN_ROWS, rank)
    dev_df = prepare_split(dev_raw, dev_cols, "dev", MAX_DEV_ROWS, rank)
    test_df = prepare_split(test_raw, test_cols, "test", MAX_TEST_ROWS, rank)
    validate_config_and_data(train_df, dev_df, test_df, train_cols, dev_cols, test_cols)

    train_labels = train_df["_label"].astype(int).tolist()
    median_label = int(np.median(train_labels))
    class_weights = compute_class_weights(train_labels, NUM_LABELS)

    dev_predictions: List[pd.DataFrame] = []
    test_predictions: List[pd.DataFrame] = []
    for model_key, loss_type in ENSEMBLE_COMBOS:
        dev_pred_df, test_pred_df = train_one_combo(
            model_key=model_key,
            loss_type=loss_type,
            train_df=train_df,
            dev_df=dev_df,
            test_df=test_df,
            class_weights=class_weights,
            rank=rank,
            world_size=world_size,
        )
        if is_rank_zero(rank):
            assert dev_pred_df is not None and test_pred_df is not None
            dev_predictions.append(dev_pred_df)
            test_predictions.append(test_pred_df)

    if is_rank_zero(rank):
        prediction_dir = ensure_dir(Path(OUTPUT_DIR) / "predictions")
        dev_ensemble = confidence_weighted_ensemble(dev_predictions, fallback_label=median_label)
        test_ensemble = confidence_weighted_ensemble(test_predictions, fallback_label=median_label)
        if USE_POST_PROCESSING:
            dev_ensemble["pred_label"] = post_process_predictions(dev_ensemble["pred_label"], median_label)
            test_ensemble["pred_label"] = post_process_predictions(test_ensemble["pred_label"], median_label)
        dev_ensemble.to_csv(prediction_dir / "dev_ensemble.csv", index=False)
        test_ensemble.to_csv(prediction_dir / "test_ensemble.csv", index=False)
        metrics = compute_metrics(dev_df["_label"].tolist(), dev_ensemble["pred_label"].astype(int).tolist())
        log(
            f"[ensemble] qwk={metrics['qwk']:.5f} acc={metrics['accuracy']:.5f} "
            f"adj_acc={metrics['adjacent_accuracy']:.5f} mae={metrics['mae']:.5f}",
            rank,
        )
        submission_df = aggregate_document_predictions(test_ensemble)
        create_submission(
            submission_df["Sentence ID"].astype(str).tolist(),
            submission_df["pred_label"].astype(int).tolist(),
            SUBMISSION_DIR,
        )
    cleanup_distributed()


def main() -> None:
    """
    Entrypoint that automatically uses DDP when multiple GPUs are available.
    """
    start = time.time()
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if USE_DDP_AUTO and gpu_count > 1:
        mp.spawn(run_pipeline, args=(gpu_count,), nprocs=gpu_count, join=True)
    else:
        run_pipeline(rank=0, world_size=1)
    print(f"Finished in {(time.time() - start) / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
