from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import train
import train_blind


class DummyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 8, vocab_size: int = 32) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del attention_mask, token_type_ids
        hidden = self.projection(self.embedding(input_ids))
        return SimpleNamespace(last_hidden_state=hidden)


def make_model(dropout: float = 0.0) -> train.CascadedHierarchicalReadabilityRegressor:
    with patch.object(
        train.AutoModel,
        "from_pretrained",
        side_effect=lambda *args, **kwargs: DummyEncoder(),
    ):
        return train.CascadedHierarchicalReadabilityRegressor(
            "offline-dummy", dropout=dropout, temperature=1.0
        )


class MappingAndLoadingTests(unittest.TestCase):
    def test_official_mapping_and_zero_based_columns(self) -> None:
        train.run_auxiliary_mapping_check()
        labels19 = np.arange(1, 20, dtype=np.int64)
        mapped = train.auxiliary_labels_from_19(labels19)
        frame = pd.DataFrame(
            {
                "ID": [f"id-{value}" for value in labels19],
                "Sentence": ["جملة" for _ in labels19],
                "Readability_Level_19": labels19,
                "Readability_Level_3": mapped[3],
                "Readability_Level_5": mapped[5],
                "Readability_Level_7": mapped[7],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.csv"
            frame.to_csv(path, index=False)
            loaded = train.load_split(
                path, "probe", train.Config(), require_label=True
            )
        self.assertEqual((loaded["_label3"].min(), loaded["_label3"].max()), (0, 2))
        self.assertEqual((loaded["_label5"].min(), loaded["_label5"].max()), (0, 4))
        self.assertEqual((loaded["_label7"].min(), loaded["_label7"].max()), (0, 6))

    def test_missing_or_mismatched_auxiliary_label_fails(self) -> None:
        frame = pd.DataFrame(
            {
                "ID": ["x"],
                "Sentence": ["جملة"],
                "Readability_Level_19": [12],
                "Readability_Level_3": [2],
                "Readability_Level_5": [3],
                "Readability_Level_7": [5],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing_path = root / "missing.csv"
            frame.drop(columns="Readability_Level_5").to_csv(missing_path, index=False)
            with self.assertRaisesRegex(ValueError, "5-level label"):
                train.load_split(
                    missing_path, "probe", train.Config(), require_label=True
                )

            mismatch_path = root / "mismatch.csv"
            frame.assign(Readability_Level_7=4).to_csv(mismatch_path, index=False)
            with self.assertRaisesRegex(ValueError, "official 19→7 mapping"):
                train.load_split(
                    mismatch_path, "probe", train.Config(), require_label=True
                )

    def test_fractional_label_is_never_treated_as_integer(self) -> None:
        labels = pd.Series(["19.0001"])
        with self.assertRaisesRegex(ValueError, "non-integer"):
            train.validated_integer_labels(labels, "probe", "19-level", 1, 19)


class PreprocessingProgressTests(unittest.TestCase):
    def test_long_cache_fingerprint_does_not_hide_batch_counter(self) -> None:
        cache_path = Path("train-" + "a" * 64 + ".parquet")
        options = train.d3tok_progress_options(cache_path, 215)
        self.assertEqual(options["desc"], "BERT D3Tok train")
        self.assertEqual(options["total"], 215)
        self.assertEqual(options["unit"], "batch")
        self.assertTrue(options["dynamic_ncols"])

        rendered = train.tqdm.format_meter(
            1,
            options["total"],
            3.0,
            ncols=80,
            prefix=options["desc"],
            unit=options["unit"],
        )
        self.assertIn("1/215", rendered)


class CascadedModelTests(unittest.TestCase):
    def test_shapes_and_soft_conditioning(self) -> None:
        for batch_size in (1, 4):
            model = make_model()
            model.eval()
            captured: dict[str, torch.Tensor] = {}

            def capture(name: str):
                def hook(module: nn.Module, arguments: tuple[torch.Tensor, ...]) -> None:
                    del module
                    captured[name] = arguments[0].detach().clone()

                return hook

            handles = [
                model.head5.register_forward_pre_hook(capture("head5")),
                model.head7.register_forward_pre_hook(capture("head7")),
                model.head19.register_forward_pre_hook(capture("head19")),
            ]
            inputs = {
                "input_ids": torch.randint(0, 32, (batch_size, 6)),
                "attention_mask": torch.ones(batch_size, 6, dtype=torch.long),
                "token_type_ids": torch.zeros(batch_size, 6, dtype=torch.long),
            }
            outputs = model(**inputs)
            for handle in handles:
                handle.remove()

            self.assertEqual(outputs["score19"].shape, torch.Size([batch_size]))
            self.assertEqual(outputs["logits3"].shape, torch.Size([batch_size, 3]))
            self.assertEqual(outputs["logits5"].shape, torch.Size([batch_size, 5]))
            self.assertEqual(outputs["logits7"].shape, torch.Size([batch_size, 7]))
            torch.testing.assert_close(
                captured["head5"][:, -3:], torch.softmax(outputs["logits3"], -1)
            )
            torch.testing.assert_close(
                captured["head7"][:, -5:], torch.softmax(outputs["logits5"], -1)
            )
            torch.testing.assert_close(
                captured["head19"][:, -7:], torch.softmax(outputs["logits7"], -1)
            )

    def test_mse_gradient_reaches_every_stage(self) -> None:
        torch.manual_seed(7)
        model = make_model()
        with torch.no_grad():
            model.head5.weight[:, -3:] = torch.arange(15).reshape(5, 3) / 10
            model.head7.weight[:, -5:] = torch.arange(35).reshape(7, 5) / 10
            model.head19.weight[:, -7:] = torch.arange(7).reshape(1, 7) / 10
        outputs = model(
            input_ids=torch.randint(0, 32, (4, 6)),
            attention_mask=torch.ones(4, 6, dtype=torch.long),
            token_type_ids=torch.zeros(4, 6, dtype=torch.long),
        )
        nn.functional.mse_loss(outputs["score19"], torch.full((4,), 10.0)).backward()
        for name, module in {
            "encoder": model.encoder,
            "head3": model.head3,
            "head5": model.head5,
            "head7": model.head7,
            "head19": model.head19,
        }.items():
            gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            self.assertTrue(all(gradient is not None for gradient in gradients), name)
            self.assertTrue(
                all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None),
                name,
            )
            self.assertGreater(
                sum(float(gradient.abs().sum()) for gradient in gradients if gradient is not None),
                0.0,
                name,
            )

    def test_joint_loss_is_fp32_and_uses_half_weight(self) -> None:
        config = train.Config()
        outputs = {
            "score19": torch.tensor([4.0, 7.0], dtype=torch.float16),
            "logits3": torch.tensor([[2, 1, 0], [0, 1, 2]], dtype=torch.float16),
            "logits5": torch.randn(2, 5, dtype=torch.float16),
            "logits7": torch.randn(2, 7, dtype=torch.float16),
        }
        batch = {
            "labels": torch.tensor([5.0, 8.0]),
            "labels3": torch.tensor([0, 2]),
            "labels5": torch.tensor([1, 4]),
            "labels7": torch.tensor([2, 6]),
        }
        losses = train.compute_training_losses(outputs, batch, config)
        expected = losses["mse19"] + 0.5 * (
            losses["ce3"] + losses["ce5"] + losses["ce7"]
        )
        torch.testing.assert_close(losses["total"], expected)
        self.assertEqual(losses["total"].dtype, torch.float32)

        baseline_config = train.Config()
        baseline_config.MODEL_MODE = "baseline_mse"
        baseline_losses = train.compute_training_losses(
            {"score19": outputs["score19"]}, batch, baseline_config
        )
        torch.testing.assert_close(
            baseline_losses["total"], baseline_losses["mse19"]
        )
        self.assertEqual(float(baseline_losses["ce3"]), 0.0)

    def test_optimizer_contains_every_parameter_once(self) -> None:
        model = make_model()
        config = train.Config()
        optimizer = train.create_optimizer(model, config)
        optimizer_parameter_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        model_parameter_ids = [
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        ]
        self.assertEqual(len(optimizer_parameter_ids), len(set(optimizer_parameter_ids)))
        self.assertEqual(set(optimizer_parameter_ids), set(model_parameter_ids))
        for group in optimizer.param_groups:
            expected_lr = (
                config.ENCODER_LR
                if str(group["group_name"]).startswith("encoder")
                else config.HEAD_LR
            )
            self.assertEqual(group["lr"], expected_lr)

    def test_cascaded_resume_signature_locks_behavior(self) -> None:
        config = train.Config()
        signature = train.resume_compatibility_signature(config)
        self.assertEqual(
            signature["CHECKPOINT_SELECTION_POLICY"],
            train.CHECKPOINT_SELECTION_POLICY,
        )
        self.assertEqual(
            signature["FINAL_DISCRETIZATION_POLICY"],
            train.FINAL_DISCRETIZATION_POLICY,
        )
        self.assertEqual(signature["MIN_LABEL"], config.MIN_LABEL)
        self.assertEqual(signature["MAX_LABEL"], config.MAX_LABEL)
        checkpoint = {
            "config": {"MODEL_MODE": config.MODEL_MODE},
            "resume_signature": signature,
        }
        train.validate_resume_compatibility(checkpoint, config)

        changed = train.Config()
        changed.CASCADE_TEMPERATURE = 0.5
        with self.assertRaisesRegex(RuntimeError, "CASCADE_TEMPERATURE"):
            train.validate_resume_compatibility(checkpoint, changed)

        unsigned = {"config": {"MODEL_MODE": "cascaded_hmtl"}}
        with self.assertRaisesRegex(RuntimeError, "lacks a compatibility signature"):
            train.validate_resume_compatibility(unsigned, config)

        legacy_signature = dict(signature)
        legacy_signature.pop("CHECKPOINT_SELECTION_POLICY")
        legacy_signature.pop("FINAL_DISCRETIZATION_POLICY")
        legacy = {
            "config": {"MODEL_MODE": config.MODEL_MODE},
            "resume_signature": legacy_signature,
        }
        with self.assertRaisesRegex(RuntimeError, "floor-selection pipeline"):
            train.validate_resume_compatibility(legacy, config)

        wrong_range_signature = dict(signature)
        wrong_range_signature["MAX_LABEL"] = 18
        wrong_range = {
            "config": {"MODEL_MODE": config.MODEL_MODE},
            "resume_signature": wrong_range_signature,
        }
        with self.assertRaisesRegex(RuntimeError, "MAX_LABEL"):
            train.validate_resume_compatibility(wrong_range, config)


class EnsembleAndSubmissionTests(unittest.TestCase):
    def test_weighted_sampler_defaults_remain_locked(self) -> None:
        config = train.Config()
        self.assertTrue(config.USE_WEIGHTED_SAMPLER)
        self.assertEqual(config.SAMPLER_ALPHA, 0.5)

    def test_discretization_and_average_before_floor(self) -> None:
        config = train.Config()
        train.run_discretization_checks(config)
        outputs = []
        for raw in ([1.2, 7.2], [2.0, 7.8]):
            raw_array = np.asarray(raw, dtype=np.float64)
            outputs.append(
                train.EvaluationOutput(
                    ids=["a", "b"],
                    indices=np.asarray([0, 1]),
                    raw_predictions=raw_array,
                    round_predictions=train.round_and_clip(raw_array, config),
                    up_predictions=train.ceil_and_clip(raw_array, config),
                    down_predictions=train.floor_and_clip(raw_array, config),
                    labels=None,
                    metrics=None,
                )
            )
        ensemble = train.average_evaluation_outputs(outputs, config)
        np.testing.assert_allclose(ensemble.raw_predictions, [1.6, 7.5])
        self.assertEqual(ensemble.round_predictions.tolist(), [2, 8])
        self.assertEqual(ensemble.up_predictions.tolist(), [2, 8])
        self.assertEqual(ensemble.down_predictions.tolist(), [1, 7])

    def test_down_checkpoint_selection_ignores_better_round_qwk(self) -> None:
        first = {
            "down_qwk": 0.80,
            "down_mae": 1.0,
            "round_qwk": 0.95,
        }
        selection_qwk, selection_mae, improved = train.down_checkpoint_decision(
            first,
            best_down_qwk=0.81,
            best_down_mae=1.2,
            has_selected_model=True,
        )
        self.assertEqual(selection_qwk, 0.80)
        self.assertEqual(selection_mae, 1.0)
        self.assertFalse(improved)

        tied = {"down_qwk": 0.81, "down_mae": 1.1, "round_qwk": 0.1}
        _, _, improved = train.down_checkpoint_decision(
            tied,
            best_down_qwk=0.81,
            best_down_mae=1.2,
            has_selected_model=True,
        )
        self.assertTrue(improved)

    def test_calculated_floor_metrics_drive_checkpoint_selection(self) -> None:
        config = train.Config()
        metrics = train.calculate_metrics(
            labels=[1, 2, 3, 4],
            raw_predictions=[1.9, 2.9, 3.9, 4.9],
            config=config,
        )
        self.assertEqual(metrics["down_qwk"], 1.0)
        self.assertEqual(metrics["down_mae"], 0.0)
        self.assertLess(metrics["round_qwk"], metrics["down_qwk"])
        _, _, improved = train.down_checkpoint_decision(
            metrics,
            best_down_qwk=0.99,
            best_down_mae=0.0,
            has_selected_model=True,
        )
        self.assertTrue(improved)

    def test_one_valid_down_zip_with_official_internal_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = train.Config()
            config.SUBMISSION_DIR = directory
            for stale_name in (
                "prediction",
                "prediction.zip",
                "prediction_round",
                "prediction_round.zip",
                "prediction_up",
                "prediction_up.zip",
            ):
                (Path(directory) / stale_name).write_text("stale", encoding="utf-8")
            paths = train.create_submissions(["id-1", "id-2"], [1, 2], config)
            source_path, zip_path = paths["down"]
            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(archive.namelist(), ["prediction"])
                rows = archive.read("prediction").decode("utf-8").splitlines()
            self.assertEqual(rows[0], "Sentence ID,Prediction")
            self.assertEqual([row.split(",")[1] for row in rows[1:]], ["1", "2"])
            self.assertEqual(archive_bytes(zip_path), source_path.read_bytes())
            root = Path(directory)
            for stale_name in (
                "prediction",
                "prediction.zip",
                "prediction_round",
                "prediction_round.zip",
                "prediction_up",
                "prediction_up.zip",
            ):
                self.assertFalse((root / stale_name).exists())

    def test_blind_label_aliases_are_removed(self) -> None:
        frame = pd.DataFrame(
            {
                "ID": ["x"],
                "Sentence": ["جملة"],
                "Domain": ["STEM"],
                "Text_Class": ["Advanced"],
                "label3": [1],
                "label_5": [2],
                "Readability_Level_7": [3],
            }
        )
        stripped = train_blind.strip_blind_labels(frame)
        self.assertEqual(
            stripped.columns.tolist(), ["ID", "Sentence", "Domain", "Text_Class"]
        )

    def test_blind_cache_requires_all_structured_metadata(self) -> None:
        complete = pd.DataFrame(
            {
                "ID": ["x"],
                "Sentence": ["جملة"],
                "Document": ["doc-1"],
                "Domain": ["STEM"],
                "Text_Class": ["Advanced"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete_path = root / "complete.parquet"
            complete.to_parquet(complete_path, index=False)
            row_count, columns = train_blind.validate_local_blind(complete_path)
            self.assertEqual(row_count, 1)
            self.assertEqual(columns, complete.columns.tolist())

            incomplete_path = root / "incomplete.parquet"
            complete.drop(columns="Domain").to_parquet(incomplete_path, index=False)
            with self.assertRaisesRegex(RuntimeError, "Domain"):
                train_blind.validate_local_blind(incomplete_path)


def archive_bytes(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        return archive.read("prediction")


if __name__ == "__main__":
    unittest.main()
