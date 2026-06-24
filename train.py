import math
import random
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class Configuration:
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 2e-5
    EPOCHS: int = 3
    OPTIMIZER: str = "AdamW"
    EARLY_STOPPING_PATIENCE: int = 2
    MAX_LENGTH: int = 256
    DATA_PATHS: Dict[str, str] = field(
        default_factory=lambda: {
            "train": "data/barec-corpus-v1/train.csv",
            "dev": "data/barec-corpus-v1/dev.csv",
            "test": "data/barec-corpus-v1/test.csv",
        }
    )
    OUTPUT_DIR: str = "outputs"

    TRACK: str = "strict"
    TASK_LEVEL: str = "sentence"
    BASE_MODELS: Tuple[str, ...] = (
        "aubmindlab/bert-base-arabertv2",
        "aubmindlab/araelectra-base-discriminator",
    )
    LOSS_FUNCTIONS: Tuple[str, ...] = ("CE", "MSE")
    NUM_LABELS: int = 19
    SEED: int = 42
    USE_D3TOK: bool = True
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS: int = 2
    WARMUP_RATIO: float = 0.1
    STATIC_MSE_CONFIDENCE: float = 1.0

    @property
    def MODEL_LOSS_COMBINATIONS(self) -> List[Tuple[str, str]]:
        return [(model_name, loss_name) for model_name in self.BASE_MODELS for loss_name in self.LOSS_FUNCTIONS]


CFG = Configuration()


# =============================================================================
# Utilities
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_name(model_name: str, loss_name: str) -> str:
    return f"{model_name.replace('/', '__')}_{loss_name}"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"ID": str})


# =============================================================================
# Data preprocessing
# =============================================================================


ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
REDUNDANT_PUNCT_RE = re.compile(r"([؟?!.,،؛:])\1+")
EXTRA_SPACES_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    text = text.replace("\u0640", "")
    text = ARABIC_DIACRITICS_RE.sub("", text)
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    text = text.replace("ؤ", "و").replace("ئ", "ي")
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = REDUNDANT_PUNCT_RE.sub(r"\1", text)
    text = re.sub(r"([؟?!.,،؛:])", r" \1 ", text)
    text = EXTRA_SPACES_RE.sub(" ", text)
    return text.strip()


def build_d3_tokenizer():
    #  khởi tạo và thiết lập công cụ tách từ tiếng Ả Rập
    try:
        from camel_tools.disambig.mle import MLEDisambiguator
        from camel_tools.tokenizers.morphological import MorphologicalTokenizer

        disambiguator = MLEDisambiguator.pretrained("calima-msa-r13")
        return MorphologicalTokenizer(disambiguator=disambiguator, scheme="d3tok", split=True)
    except Exception as exc:
        print(f"Warning: D3TOK unavailable ({exc}). Continuing with cleaned text only.")
        return None


def d3_tokenize(text: str, tokenizer) -> str:
    if tokenizer is None:
        return text
    try:
        return " ".join(tokenizer.tokenize(text))
    except Exception:
        return text


def preprocess_dataframe(df: pd.DataFrame, d3_tokenizer=None) -> pd.DataFrame:
    processed = df.copy()
    processed["ID"] = processed["ID"].astype(str)
    processed["clean_sentence"] = processed["Sentence"].map(clean_text)
    if d3_tokenizer is not None:
        processed["model_text"] = [d3_tokenize(text, d3_tokenizer) for text in tqdm(processed["clean_sentence"], desc="D3TOK")]
    else:
        processed["model_text"] = processed["clean_sentence"]
    return processed


def compute_class_weights(labels_1_to_19: Iterable[int], num_labels: int = 19) -> torch.Tensor:
    labels = np.asarray(list(labels_1_to_19), dtype=np.int64)
    total = len(labels)
    weights = np.zeros(num_labels, dtype=np.float32)
    for level in range(1, num_labels + 1):
        count = int((labels == level).sum())
        weights[level - 1] = total / (num_labels * count) if count > 0 else 0.0
    return torch.tensor(weights, dtype=torch.float32)


class BARECDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int,
        label_column: Optional[str] = "Readability_Level_19",
    ) -> None:
        self.ids = df["ID"].astype(str).tolist()
        self.texts = df["model_text"].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = None
        if label_column and label_column in df.columns:
            self.labels = df[label_column].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["id"] = self.ids[idx]
        if self.labels is not None:
            label_level = int(self.labels[idx])
            item["labels"] = torch.tensor(label_level - 1, dtype=torch.long)
        return item


# =============================================================================
# Models and losses
# =============================================================================


def coral_targets(labels_zero_based: torch.Tensor, num_labels: int) -> torch.Tensor:
    thresholds = torch.arange(num_labels - 1, device=labels_zero_based.device).unsqueeze(0)
    return (labels_zero_based.unsqueeze(1) > thresholds).float()


class ReadabilityModel(nn.Module):
    def __init__(self, model_name: str, loss_type: str, num_labels: int, class_weights: torch.Tensor) -> None:
        super().__init__()
        self.loss_type = loss_type.upper()
        self.num_labels = num_labels
        self.register_buffer("class_weights", class_weights.float())

        if self.loss_type == "CE":
            self.backbone = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            self.dropout = None
            self.head = None
        elif self.loss_type in {"MSE", "COR"}:
            self.backbone = AutoModel.from_pretrained(model_name)
            hidden_size = self.backbone.config.hidden_size
            self.dropout = nn.Dropout(getattr(self.backbone.config, "hidden_dropout_prob", 0.1))
            output_dim = 1 if self.loss_type == "MSE" else num_labels - 1
            self.head = nn.Linear(hidden_size, output_dim)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids

        if self.loss_type == "CE":
            outputs = self.backbone(**model_inputs)
            logits = outputs.logits
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)
            return {"loss": loss, "logits": logits}

        outputs = self.backbone(**model_inputs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        logits = self.head(self.dropout(pooled))

        loss = None
        if labels is not None:
            sample_weights = self.class_weights[labels]
            if self.loss_type == "MSE":
                pred = logits.squeeze(-1)
                target = labels.float() + 1.0
                per_sample = (pred - target).pow(2)
                loss = (per_sample * sample_weights).sum() / sample_weights.clamp_min(1e-8).sum()
            else:
                target = coral_targets(labels, self.num_labels)
                per_threshold = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
                per_sample = per_threshold.mean(dim=1)
                loss = (per_sample * sample_weights).sum() / sample_weights.clamp_min(1e-8).sum()
        return {"loss": loss, "logits": logits}


def batch_to_device(batch: Dict[str, torch.Tensor | str], device: str) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items() if key != "id"}


def logits_to_predictions(loss_type: str, logits: torch.Tensor, cfg: Configuration) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loss_type = loss_type.upper()
    if loss_type == "CE":
        probs = torch.softmax(logits, dim=1)
        levels = torch.arange(1, cfg.NUM_LABELS + 1, device=logits.device, dtype=torch.float32)
        scores = (probs * levels.unsqueeze(0)).sum(dim=1)
        pred_labels = probs.argmax(dim=1).float() + 1.0
        confidence = probs.max(dim=1).values
    elif loss_type == "MSE":
        scores = logits.squeeze(-1).clamp(1, cfg.NUM_LABELS)
        pred_labels = scores.round().clamp(1, cfg.NUM_LABELS)
        confidence = torch.full_like(scores, fill_value=cfg.STATIC_MSE_CONFIDENCE)
    elif loss_type == "COR":
        probs = torch.sigmoid(logits)
        pred_labels = 1.0 + (probs > 0.5).sum(dim=1).float()
        scores = 1.0 + probs.sum(dim=1)
        confidence = (torch.abs(probs - 0.5) * 2.0).mean(dim=1).clamp_min(1e-4)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    return (
        scores.detach().cpu().numpy(),
        pred_labels.detach().cpu().numpy(),
        confidence.detach().cpu().numpy(),
    )


# =============================================================================
# Training and inference
# =============================================================================


def evaluate(model: nn.Module, dataloader: DataLoader, loss_type: str, cfg: Configuration) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            labels = batch["labels"].numpy() + 1
            inputs = batch_to_device(batch, cfg.DEVICE)
            outputs = model(**inputs)
            loss = outputs["loss"]
            if isinstance(model, nn.DataParallel):
                loss = loss.mean()
            batch_size = len(labels)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_rows += batch_size
            _, pred_labels, _ = logits_to_predictions(loss_type, outputs["logits"], cfg)
            y_true.extend(labels.tolist())
            y_pred.extend(pred_labels.astype(int).tolist())

    return {
        "loss": total_loss / max(total_rows, 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def predict(model: nn.Module, dataloader: DataLoader, loss_type: str, cfg: Configuration) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predict", leave=False):
            ids = batch["id"]
            inputs = batch_to_device(batch, cfg.DEVICE)
            outputs = model(**inputs)
            scores, labels, confidences = logits_to_predictions(loss_type, outputs["logits"], cfg)
            for sample_id, score, label, confidence in zip(ids, scores, labels, confidences):
                rows.append(
                    {
                        "ID": str(sample_id),
                        "score": float(score),
                        "label": int(np.clip(round(float(label)), 1, cfg.NUM_LABELS)),
                        "confidence": float(confidence),
                    }
                )
    return pd.DataFrame(rows)


def save_checkpoint(model: nn.Module, tokenizer, checkpoint_dir: Path, metadata: Dict[str, str]) -> None:
    ensure_dir(checkpoint_dir)
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({"state_dict": base_model.state_dict(), "metadata": metadata}, checkpoint_dir / "model.pt")
    tokenizer.save_pretrained(checkpoint_dir / "tokenizer")


def load_checkpoint(model: nn.Module, checkpoint_dir: Path, device: str) -> bool:
    checkpoint_path = checkpoint_dir / "model.pt"
    if not checkpoint_path.exists():
        return False
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return True


def train_one_model(
    model_name: str,
    loss_type: str,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_weights: torch.Tensor,
    cfg: Configuration,
) -> pd.DataFrame:
    run_name = safe_name(model_name, loss_type)
    checkpoint_dir = Path(cfg.OUTPUT_DIR) / "checkpoints" / run_name
    predictions_path = Path(cfg.OUTPUT_DIR) / "predictions" / f"{run_name}.csv"
    ensure_dir(predictions_path.parent)

    if predictions_path.exists():
        print(f"Loading cached predictions: {predictions_path}")
        return pd.read_csv(predictions_path, dtype={"ID": str})

    print(f"\n=== Training {model_name} with {loss_type} ===")
    tokenizer_path = checkpoint_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path.exists() else model_name)

    train_dataset = BARECDataset(train_df, tokenizer, cfg.MAX_LENGTH)
    dev_dataset = BARECDataset(dev_df, tokenizer, cfg.MAX_LENGTH)
    test_dataset = BARECDataset(test_df, tokenizer, cfg.MAX_LENGTH, label_column=None)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    dev_loader = DataLoader(dev_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    model = ReadabilityModel(model_name, loss_type, cfg.NUM_LABELS, class_weights).to(cfg.DEVICE)

    is_loaded = load_checkpoint(model, checkpoint_dir, cfg.DEVICE)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        model = nn.DataParallel(model)

    if not is_loaded:
        if cfg.OPTIMIZER != "AdamW":
            raise ValueError("This implementation supports OPTIMIZER='AdamW'.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
        scaler = torch.cuda.amp.GradScaler()
        total_steps = len(train_loader) * cfg.EPOCHS
        warmup_steps = int(total_steps * cfg.WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        best_dev_loss = float("inf")
        patience = 0
        best_state = None

        for epoch in range(1, cfg.EPOCHS + 1):
            model.train()
            running_loss = 0.0
            seen = 0
            progress = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}")
            for batch in progress:
                inputs = batch_to_device(batch, cfg.DEVICE)
                optimizer.zero_grad(set_to_none=True)
                
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(**inputs)
                    loss = outputs["loss"]
                    if isinstance(model, nn.DataParallel):
                        loss = loss.mean()
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                batch_size = inputs["input_ids"].size(0)
                running_loss += float(loss.detach().cpu()) * batch_size
                seen += batch_size
                progress.set_postfix(loss=running_loss / max(seen, 1))

            metrics = evaluate(model, dev_loader, loss_type, cfg)
            print(
                f"Epoch {epoch}: dev_loss={metrics['loss']:.4f}, "
                f"acc={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, mae={metrics['mae']:.4f}"
            )

            if metrics["loss"] < best_dev_loss:
                best_dev_loss = metrics["loss"]
                patience = 0
                base_model = model.module if isinstance(model, nn.DataParallel) else model
                best_state = {key: value.detach().cpu().clone() for key, value in base_model.state_dict().items()}
            else:
                patience += 1
                if patience >= cfg.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

        if best_state is not None:
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            base_model.load_state_dict(best_state)
        save_checkpoint(model, tokenizer, checkpoint_dir, {"model_name": model_name, "loss_type": loss_type})
    else:
        print(f"Loaded checkpoint: {checkpoint_dir}")

    predictions = predict(model, test_loader, loss_type, cfg)
    predictions["model_name"] = model_name
    predictions["loss_type"] = loss_type
    predictions.to_csv(predictions_path, index=False)
    return predictions


# =============================================================================
# Ensembling and submission
# =============================================================================


class Ensembler:
    def __init__(self, num_labels: int) -> None:
        self.num_labels = num_labels

    def _combine_sentence_group(self, group: pd.DataFrame) -> int:
        labels = group["label"].astype(int).tolist()
        high_labels = [label for label in labels if label in {16, 17}]
        if high_labels:
            return max(high_labels)

        unique_labels = sorted(set(labels))
        if len(unique_labels) == 2 and abs(unique_labels[0] - unique_labels[1]) == 1:
            return int(unique_labels[1])

        confidences = group["confidence"].astype(float).clip(lower=1e-8).to_numpy()
        scores = group["score"].astype(float).to_numpy()
        weighted_score = float(np.sum(scores * confidences) / np.sum(confidences))
        return int(np.clip(math.floor(weighted_score), 1, self.num_labels))

    def combine(self, model_predictions: List[pd.DataFrame]) -> pd.DataFrame:
        all_predictions = pd.concat(model_predictions, ignore_index=True)

        sentence_rows = []
        for sentence_id, group in all_predictions.groupby("ID", sort=False):
            sentence_rows.append({"Sentence ID": sentence_id, "label": self._combine_sentence_group(group)})
        sentence_predictions = pd.DataFrame(sentence_rows)
        sentence_predictions["label"] = sentence_predictions["label"].astype(int).clip(1, self.num_labels)
        return sentence_predictions


def write_submission(predictions: pd.DataFrame, output_dir: str) -> Path:
    output_path = ensure_dir(output_dir)
    submission_path = output_path / "prediction"
    zip_path = Path("prediction.zip")

    predictions[["Sentence ID", "label"]].to_csv(submission_path, index=False)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(submission_path, arcname="prediction")
    return zip_path


def main() -> None:
    set_seed(CFG.SEED)
    ensure_dir(CFG.OUTPUT_DIR)

    print(f"Using device: {CFG.DEVICE}")
    print("Loading datasets...")
    train_df = read_csv(CFG.DATA_PATHS["train"])
    dev_df = read_csv(CFG.DATA_PATHS["dev"])
    test_df = read_csv(CFG.DATA_PATHS["test"])

    d3_tokenizer = build_d3_tokenizer() if CFG.USE_D3TOK else None
    print("Preprocessing train/dev/test...")
    train_df = preprocess_dataframe(train_df, d3_tokenizer)
    dev_df = preprocess_dataframe(dev_df, d3_tokenizer)
    test_df = preprocess_dataframe(test_df, d3_tokenizer)

    class_weights = compute_class_weights(train_df["Readability_Level_19"], CFG.NUM_LABELS).to(CFG.DEVICE)
    print(f"Class weights: {class_weights.detach().cpu().numpy().round(4).tolist()}")

    model_predictions: List[pd.DataFrame] = []
    combinations = CFG.MODEL_LOSS_COMBINATIONS
    print(f"\n==============================================")
    print(f"Bắt đầu huấn luyện {len(combinations)} tổ hợp mô hình...")
    print(f"==============================================\n")
    for model_name, loss_type in combinations:
        preds = train_one_model(model_name, loss_type, train_df, dev_df, test_df, class_weights, CFG)
        model_predictions.append(preds)

    ensembler = Ensembler(CFG.NUM_LABELS)
    final_predictions = ensembler.combine(model_predictions)
    zip_path = write_submission(final_predictions, CFG.OUTPUT_DIR)

    print(f"Generated {len(final_predictions)} sentence-level predictions for the {CFG.TRACK} track.")
    print(f"Training complete. Submission file saved as {zip_path}")


if __name__ == "__main__":
    main()
