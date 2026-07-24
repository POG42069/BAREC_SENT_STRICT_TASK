from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
import torch.nn as nn

import train
import hierarchical
from hierarchical import HierarchicalArabicReadabilityRegressor


class DummyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 12, vocabulary_size: int = 64) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del attention_mask, token_type_ids
        hidden = self.linear(self.embedding(input_ids))
        return SimpleNamespace(last_hidden_state=hidden)


class DummyTokenizer:
    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        max_length: int,
        padding: bool = False,
        return_tensors: str | None = None,
    ) -> dict[str, object]:
        del truncation, padding
        values = [min(ord(character), 63) for character in text][:max_length] or [1]
        result: dict[str, object] = {
            "input_ids": values,
            "attention_mask": [1] * len(values),
        }
        if return_tensors == "pt":
            return {
                key: torch.tensor([value], dtype=torch.long)
                for key, value in result.items()
            }
        return result

    def pad(
        self,
        features: list[dict[str, object]],
        *,
        padding: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del padding, return_tensors
        width = max(len(feature["input_ids"]) for feature in features)  # type: ignore[arg-type]
        output: dict[str, list[list[int]]] = {
            "input_ids": [],
            "attention_mask": [],
        }
        for feature in features:
            for key in output:
                values = list(feature[key])  # type: ignore[arg-type]
                output[key].append(values + [0] * (width - len(values)))
        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in output.items()
        }

    def save_pretrained(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "dummy_tokenizer.json").write_text("{}\n", encoding="utf-8")


def make_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "_processed_text": ["abc", "xy"],
            "_id": ["a", "b"],
            "_original_index": [0, 1],
            "_label": [1, 19],
            "_label3": [0, 2],
            "_label5": [0, 4],
            "_label7": [0, 6],
        }
    )
    frame.attrs["has_labels"] = True
    return frame


def make_processed_frame(
    labels: list[int] | None,
    *,
    prefix: str,
) -> pd.DataFrame:
    size = len(labels) if labels is not None else 4
    frame = pd.DataFrame(
        {
            "_processed_text": [f"sentence {prefix} {index}" for index in range(size)],
            "_id": [f"{prefix}-{index}" for index in range(size)],
            "_original_index": list(range(size)),
        }
    )
    frame.attrs["has_labels"] = labels is not None
    if labels is not None:
        auxiliary = hierarchical.derive_auxiliary_labels(torch.tensor(labels))
        frame["_label"] = labels
        frame["_label3"] = auxiliary.label3.numpy()
        frame["_label5"] = auxiliary.label5.numpy()
        frame["_label7"] = auxiliary.label7.numpy()
    return frame


def make_hmtl() -> HierarchicalArabicReadabilityRegressor:
    return HierarchicalArabicReadabilityRegressor(
        model_name="unused",
        dropout=0.0,
        output_bias=10.5,
        projection_size=4,
        fusion_hidden_size=8,
        encoder=DummyEncoder(),
    )


class DataAndStageTests(unittest.TestCase):
    def test_dataset_and_collator_emit_all_target_dtypes(self) -> None:
        tokenizer = DummyTokenizer()
        dataset = train.BARECDataset(make_frame(), tokenizer, max_length=8)
        batch = train.BARECCollator(tokenizer)([dataset[0], dataset[1]])

        self.assertEqual(tuple(batch["label19"].shape), (2,))
        self.assertEqual(batch["label19"].dtype, torch.float32)
        self.assertIs(batch["labels"], batch["label19"])
        for key, maximum in (("label3", 2), ("label5", 4), ("label7", 6)):
            self.assertEqual(batch[key].dtype, torch.long)
            self.assertEqual(batch[key].tolist(), [0, maximum])
        self.assertNotIn("label19", {"input_ids", "attention_mask"})

    def test_locked_stage_defaults_and_selection_fallback(self) -> None:
        config = train.Config()
        train.validate_config(config)
        stage1 = train.make_stage_spec(config, "stage1")
        stage2 = train.make_stage_spec(config, "stage2")

        self.assertEqual(config.PIPELINE_MODE, "two_stage")
        self.assertEqual(config.SAMPLER_ALPHA, 0.5)
        self.assertTrue(stage1.weighted_sampling)
        self.assertEqual(stage1.gradient_accumulation_steps, 2)
        self.assertFalse(stage2.weighted_sampling)
        self.assertEqual(stage2.gradient_accumulation_steps, 1)
        self.assertEqual(stage2.encoder_lr, 4e-6)
        self.assertEqual(stage2.head_lr, 2e-5)
        self.assertFalse(
            train.is_better_checkpoint(
                0.80,
                1.20,
                0.81,
                1.30,
                has_selected_model=True,
            )
        )
        self.assertTrue(
            train.is_better_checkpoint(
                0.81,
                1.10,
                0.81,
                1.20,
                has_selected_model=True,
            )
        )
        with self.assertRaisesRegex(ValueError, "Choose one resume point"):
            train.validate_config(
                train.Config(
                    STAGE1_RESUME_FROM_CHECKPOINT="stage1.pt",
                    STAGE2_RESUME_FROM_CHECKPOINT="stage2.pt",
                )
            )
        with self.assertRaisesRegex(ValueError, "legacy baseline"):
            train.validate_config(train.Config(RESUME_FROM_CHECKPOINT="old.pt"))


class OptimizerAndCheckpointTests(unittest.TestCase):
    def test_hmtl_optimizer_covers_every_parameter_and_stage2_starts_empty(self) -> None:
        config = train.Config()
        stage1_model = make_hmtl()
        stage1_optimizer = train.create_optimizer(stage1_model, config)
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        output = stage1_model(input_ids, torch.ones_like(input_ids))
        output.scores.sum().backward()
        stage1_optimizer.step()
        self.assertTrue(stage1_optimizer.state)

        stage2_model = make_hmtl()
        stage2_model.load_state_dict(stage1_model.state_dict(), strict=True)
        stage2_optimizer = train.create_optimizer(
            stage2_model,
            config,
            encoder_lr=config.STAGE2_ENCODER_LR,
            head_lr=config.STAGE2_HEAD_LR,
        )
        self.assertEqual(stage2_optimizer.state, {})
        optimized = {
            id(parameter)
            for group in stage2_optimizer.param_groups
            for parameter in group["params"]
        }
        trainable = {
            id(parameter)
            for parameter in stage2_model.parameters()
            if parameter.requires_grad
        }
        self.assertEqual(optimized, trainable)

    def test_stage_checkpoint_round_trip_and_stage_guard(self) -> None:
        config = train.Config()
        context = train.DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            distributed=False,
            device=torch.device("cpu"),
            backend=None,
        )
        model = make_hmtl()
        optimizer = train.create_optimizer(model, config)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scaler = train.make_grad_scaler(False)
        original = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last.pt"
            train.save_training_checkpoint(
                checkpoint,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=0,
                global_step=3,
                best_qwk=0.7,
                best_mae=1.2,
                bad_epochs=0,
                config=config,
                rng_states=[train.local_rng_state()],
                stage_name="stage1",
            )
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(1.0)
            restored = train.resume_training(
                checkpoint,
                model,
                optimizer,
                scheduler,
                scaler,
                context,
                expected_stage="stage1",
                current_config=config,
            )
            self.assertEqual(restored[:2], (1, 3))
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, original[name]))
            with self.assertRaisesRegex(ValueError, "does not match"):
                train.resume_training(
                    checkpoint,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    context,
                    expected_stage="stage2",
                    current_config=config,
                )
            changed_config = train.Config(STAGE1_CE3_WEIGHT=0.2)
            with self.assertRaisesRegex(ValueError, "resume config differs"):
                train.resume_training(
                    checkpoint,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    context,
                    expected_stage="stage1",
                    current_config=changed_config,
                )

    def test_old_baseline_model_state_still_strict_loads(self) -> None:
        with patch.object(
            train.AutoModel,
            "from_pretrained",
            side_effect=lambda *args, **kwargs: DummyEncoder(),
        ):
            old_model = train.ArabicReadabilityRegressor("dummy", 0.1)
            checkpoint_state = old_model.state_dict()
            current_model = train.ArabicReadabilityRegressor(
                "dummy",
                0.1,
                output_bias=10.5,
            )
        current_model.load_state_dict(checkpoint_state, strict=True)


class TinyEndToEndTests(unittest.TestCase):
    def _run_tiny_two_stage(self, *, blind: bool) -> None:
        train_frame = make_processed_frame(
            [1, 4, 8, 12, 15, 19],
            prefix="train",
        )
        dev_frame = make_processed_frame([1, 8, 12, 19], prefix="dev")
        test_frame = make_processed_frame(
            None if blind else [2, 9, 13, 18],
            prefix="blind" if blind else "test",
        )
        context = train.DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            distributed=False,
            device=torch.device("cpu"),
            backend=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = train.Config(
                OUTPUT_DIR=str(root / "outputs"),
                CHECKPOINT_DIR=str(root / "outputs" / "checkpoints"),
                SUBMISSION_DIR=str(root / "outputs"),
                CACHE_DIR=str(root / "cache"),
                NUM_EPOCHS=1,
                STAGE2_NUM_EPOCHS=1,
                PER_DEVICE_BATCH_SIZE=2,
                STAGE2_PER_DEVICE_BATCH_SIZE=2,
                EVAL_BATCH_SIZE=2,
                GRADIENT_ACCUMULATION_STEPS=1,
                STAGE2_GRADIENT_ACCUMULATION_STEPS=1,
                NUM_WORKERS=0,
                PIN_MEMORY=False,
                USE_FP16=False,
                LOG_EVERY_N_STEPS=1,
                SMOKE_MAX_TRAIN_STEPS=1,
                AUX_HIDDEN_SIZE=4,
                FUSION_HIDDEN_SIZE=8,
            )
            train.validate_config(config)
            with (
                patch.object(
                    train.AutoTokenizer,
                    "from_pretrained",
                    return_value=DummyTokenizer(),
                ),
                patch.object(
                    hierarchical.AutoModel,
                    "from_pretrained",
                    side_effect=lambda *args, **kwargs: DummyEncoder(),
                ),
            ):
                train.train_hierarchical_select_and_predict(
                    train_frame,
                    dev_frame,
                    test_frame,
                    config,
                    context,
                    smoke_test=True,
                )

            self.assertTrue(
                (root / "outputs" / "stage1" / "best_model" / "model_state.pt").is_file()
            )
            self.assertTrue(
                (root / "outputs" / "stage2" / "best_model" / "model_state.pt").is_file()
            )
            self.assertTrue(
                (root / "outputs" / "best_model" / "model_state.pt").is_file()
            )
            self.assertTrue((root / "outputs" / "selection.json").is_file())
            self.assertTrue((root / "outputs" / "prediction").is_file())
            self.assertTrue((root / "outputs" / "prediction.zip").is_file())

    def test_open_two_stage_checkpoint_selection_and_submission(self) -> None:
        self._run_tiny_two_stage(blind=False)

    def test_unlabelled_blind_inference_path(self) -> None:
        self._run_tiny_two_stage(blind=True)


if __name__ == "__main__":
    unittest.main()
