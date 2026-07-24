"""Fast structural tests for the scalar two-stage training configuration."""

from __future__ import annotations

import unittest
from dataclasses import asdict

import torch.nn as nn

from train import (
    ArabicReadabilityRegressor,
    Config,
    create_optimizer,
    validate_config,
    validate_qwk_resume_config,
)


def tiny_scalar_regressor() -> ArabicReadabilityRegressor:
    """Construct the repository model type without downloading a PLM."""

    model = ArabicReadabilityRegressor.__new__(ArabicReadabilityRegressor)
    nn.Module.__init__(model)
    model.encoder = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    model.dropout = nn.Dropout(0.1)
    model.regression_head = nn.Linear(4, 1)
    return model


class TwoStageConfigTests(unittest.TestCase):
    def test_locked_defaults_match_requested_pipeline(self) -> None:
        config = Config()
        validate_config(config)

        self.assertTrue(config.USE_WEIGHTED_SAMPLER)
        self.assertEqual(config.SAMPLER_ALPHA, 0.5)
        self.assertTrue(config.QWK_FINETUNE_ENABLED)
        self.assertFalse(config.QWK_FINETUNE_USE_WEIGHTED_SAMPLER)
        self.assertEqual(config.QWK_FINETUNE_NUM_EPOCHS, 2)
        self.assertEqual(config.QWK_FINETUNE_PER_DEVICE_BATCH_SIZE, 16)
        self.assertEqual(config.QWK_FINETUNE_GRADIENT_ACCUMULATION_STEPS, 1)
        self.assertEqual(config.QWK_FINETUNE_MSE_WEIGHT, 0.05)

    def test_invalid_stage_boundaries_fail_fast(self) -> None:
        invalid_configs = []

        weighted_stage2 = Config()
        weighted_stage2.QWK_FINETUNE_USE_WEIGHTED_SAMPLER = True
        invalid_configs.append(weighted_stage2)

        accumulated_stage2 = Config()
        accumulated_stage2.QWK_FINETUNE_GRADIENT_ACCUMULATION_STEPS = 2
        invalid_configs.append(accumulated_stage2)

        conflicting_resume = Config()
        conflicting_resume.RESUME_FROM_CHECKPOINT = "stage1.pt"
        conflicting_resume.QWK_FINETUNE_RESUME_FROM_CHECKPOINT = "stage2.pt"
        invalid_configs.append(conflicting_resume)

        wrong_alpha = Config()
        wrong_alpha.SAMPLER_ALPHA = 0.25
        invalid_configs.append(wrong_alpha)

        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_config(config)

    def test_stage2_optimizer_is_fresh_and_uses_lower_learning_rates(self) -> None:
        config = Config()
        model = tiny_scalar_regressor()
        optimizer = create_optimizer(
            model,
            config,
            encoder_lr=config.QWK_FINETUNE_ENCODER_LR,
            head_lr=config.QWK_FINETUNE_HEAD_LR,
        )

        self.assertEqual(len(optimizer.state), 0)
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(len(parameters), len({id(parameter) for parameter in parameters}))
        self.assertEqual(
            {group["lr"] for group in optimizer.param_groups if group["group_name"].startswith("encoder")},
            {config.QWK_FINETUNE_ENCODER_LR},
        )
        self.assertEqual(
            {group["lr"] for group in optimizer.param_groups if group["group_name"].startswith("head")},
            {config.QWK_FINETUNE_HEAD_LR},
        )

    def test_resume_rejects_material_stage2_config_change(self) -> None:
        config = Config()
        checkpoint_config = asdict(config)
        checkpoint_config["QWK_FINETUNE_PER_DEVICE_BATCH_SIZE"] = 8

        with self.assertRaises(ValueError):
            validate_qwk_resume_config(checkpoint_config, config)


if __name__ == "__main__":
    unittest.main()
