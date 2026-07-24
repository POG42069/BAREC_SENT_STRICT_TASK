"""Hierarchical model and losses for BAREC sentence readability.

This module deliberately contains no training-loop or data-loading policy.  It
provides the official 19-to-{3, 5, 7} mappings, the HMTL model, and loss
building blocks used by both ``train.py`` and ``train_blind.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.nn.functional import all_reduce as autograd_all_reduce
from transformers import AutoModel


AuxiliaryLevel = Literal[3, 5, 7]

# Index 0 corresponds to BAREC level 1.  Values are zero-based so they can be
# passed directly to torch.nn.functional.cross_entropy.
OFFICIAL_19_TO_3: tuple[int, ...] = (
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
)
OFFICIAL_19_TO_5: tuple[int, ...] = (
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    2,
    2,
    3,
    3,
    4,
    4,
    4,
    4,
)
OFFICIAL_19_TO_7: tuple[int, ...] = (
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    2,
    3,
    3,
    4,
    4,
    5,
    5,
    6,
    6,
    6,
    6,
)
OFFICIAL_HIERARCHY: dict[int, tuple[int, ...]] = {
    3: OFFICIAL_19_TO_3,
    5: OFFICIAL_19_TO_5,
    7: OFFICIAL_19_TO_7,
}


@dataclass(frozen=True)
class AuxiliaryLabels:
    """Official zero-based auxiliary labels derived from 19-level labels."""

    label3: torch.Tensor
    label5: torch.Tensor
    label7: torch.Tensor


@dataclass(frozen=True)
class HierarchicalModelOutput:
    """All outputs required by the regression and auxiliary objectives."""

    scores: torch.Tensor
    logits3: torch.Tensor
    logits5: torch.Tensor
    logits7: torch.Tensor
    cls_embedding: torch.Tensor
    z3: torch.Tensor
    z5: torch.Tensor
    z7: torch.Tensor


@dataclass(frozen=True)
class HierarchicalLossOutput:
    """Weighted Stage-1 loss and its unweighted components."""

    total: torch.Tensor
    huber: torch.Tensor
    ce3: torch.Tensor
    ce5: torch.Tensor
    ce7: torch.Tensor


@dataclass(frozen=True)
class SoftQWKOutput:
    """Differentiable SoftQWK loss plus finite fallback diagnostics."""

    loss: torch.Tensor
    used_fallback: bool
    fallback_reason: Optional[str]
    n_samples: int
    n_gold_classes: int
    observed_disagreement: float
    expected_disagreement: float


@dataclass(frozen=True)
class Stage2LossOutput:
    """Weighted Stage-2 objective and its unweighted components."""

    total: torch.Tensor
    soft_qwk: SoftQWKOutput
    huber: torch.Tensor
    ce3: torch.Tensor
    ce5: torch.Tensor
    ce7: torch.Tensor


def _as_tensor(values: Any) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values
    if hasattr(values, "to_numpy"):
        try:
            values = values.to_numpy(copy=True)
        except TypeError:
            values = values.to_numpy()
    return torch.as_tensor(values)


def _coerce_integral_tensor(values: Any, name: str) -> torch.Tensor:
    tensor = _as_tensor(values)
    if tensor.dtype == torch.bool or tensor.is_complex():
        raise ValueError(f"{name} must contain integer labels")
    if tensor.is_floating_point():
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} contains a non-finite label")
        rounded = torch.round(tensor)
        if not bool(torch.equal(tensor, rounded)):
            raise ValueError(f"{name} contains a non-integer label")
    return tensor.to(dtype=torch.long)


def _validate_range(
    labels: torch.Tensor,
    minimum: int,
    maximum: int,
    name: str,
) -> None:
    if labels.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    invalid = (labels < minimum) | (labels > maximum)
    if bool(invalid.any().item()):
        examples = labels[invalid].detach().reshape(-1).cpu().tolist()[:8]
        raise ValueError(
            f"{name} must be in [{minimum}, {maximum}]; invalid values: {examples}"
        )


def map_label19_to_auxiliary(
    labels19: Any,
    levels: AuxiliaryLevel,
) -> torch.Tensor:
    """Map 1-based 19-level labels to official zero-based auxiliary labels."""

    if levels not in OFFICIAL_HIERARCHY:
        raise ValueError(f"levels must be one of 3, 5, or 7; got {levels!r}")
    labels = _coerce_integral_tensor(labels19, "labels19")
    _validate_range(labels, 1, 19, "labels19")
    lookup = torch.tensor(
        OFFICIAL_HIERARCHY[levels],
        dtype=torch.long,
        device=labels.device,
    )
    return lookup[labels - 1]


def derive_auxiliary_labels(labels19: Any) -> AuxiliaryLabels:
    """Derive all official zero-based hierarchy labels."""

    labels = _coerce_integral_tensor(labels19, "labels19")
    _validate_range(labels, 1, 19, "labels19")
    return AuxiliaryLabels(
        label3=map_label19_to_auxiliary(labels, 3),
        label5=map_label19_to_auxiliary(labels, 5),
        label7=map_label19_to_auxiliary(labels, 7),
    )


def validate_official_hierarchy_columns(
    labels19: Any,
    labels3: Any,
    labels5: Any,
    labels7: Any,
    *,
    auxiliary_one_based: bool = True,
) -> None:
    """Fail if supplied auxiliary columns disagree with the official mapping.

    Dataset CSV columns are one-based, hence the default.  Set
    ``auxiliary_one_based=False`` when validating already-prepared CE targets.
    """

    source19 = _coerce_integral_tensor(labels19, "labels19")
    supplied = {
        3: _coerce_integral_tensor(labels3, "labels3"),
        5: _coerce_integral_tensor(labels5, "labels5"),
        7: _coerce_integral_tensor(labels7, "labels7"),
    }
    for levels, values in supplied.items():
        if values.shape != source19.shape:
            raise ValueError(
                f"labels{levels} shape {tuple(values.shape)} does not match "
                f"labels19 shape {tuple(source19.shape)}"
            )
        expected = map_label19_to_auxiliary(source19, levels)
        actual = values - 1 if auxiliary_one_based else values
        _validate_range(actual, 0, levels - 1, f"labels{levels}")
        mismatch = actual != expected
        if bool(mismatch.any().item()):
            flat_positions = (
                torch.nonzero(mismatch.reshape(-1), as_tuple=False)
                .reshape(-1)
                .detach()
                .cpu()
                .tolist()[:8]
            )
            expected_values = (
                expected.reshape(-1)[flat_positions].detach().cpu().tolist()
            )
            actual_values = actual.reshape(-1)[flat_positions].detach().cpu().tolist()
            display_expected = (
                [value + 1 for value in expected_values]
                if auxiliary_one_based
                else expected_values
            )
            display_actual = (
                [value + 1 for value in actual_values]
                if auxiliary_one_based
                else actual_values
            )
            raise ValueError(
                f"labels{levels} disagrees with the official mapping at flat "
                f"positions {flat_positions}: expected={display_expected}, "
                f"actual={display_actual}"
            )


class _Projection(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class HierarchicalArabicReadabilityRegressor(nn.Module):
    """AraBERT CLS encoder with 3/5/7-level auxiliary hierarchy heads."""

    def __init__(
        self,
        model_name: str,
        dropout: float,
        output_bias: float,
        projection_size: int = 64,
        fusion_hidden_size: int = 256,
        *,
        encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if projection_size <= 0 or fusion_hidden_size <= 0:
            raise ValueError("projection and fusion sizes must be positive")
        if not math.isfinite(float(output_bias)):
            raise ValueError("output_bias must be finite")

        self.encoder = (
            encoder
            if encoder is not None
            else AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        )
        trainable_pooler_parameters = [
            name
            for name, parameter in self.encoder.named_parameters()
            if parameter.requires_grad and "pooler" in name.lower()
        ]
        if trainable_pooler_parameters:
            raise RuntimeError(
                "CLS pooling must not retain trainable encoder-pooler parameters: "
                f"{trainable_pooler_parameters}"
            )
        try:
            hidden_size = int(self.encoder.config.hidden_size)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("encoder.config.hidden_size must be available") from exc

        self.projection3 = _Projection(hidden_size, projection_size, dropout)
        self.projection5 = _Projection(hidden_size, projection_size, dropout)
        self.projection7 = _Projection(hidden_size, projection_size, dropout)
        self.classifier3 = nn.Linear(projection_size, 3)
        self.classifier5 = nn.Linear(projection_size, 5)
        self.classifier7 = nn.Linear(projection_size, 7)

        fusion_size = hidden_size + 3 * projection_size
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_size, fusion_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, 1),
        )
        self.initialize_output_bias(output_bias)

    def initialize_output_bias(self, value: float) -> None:
        """Initialize the scalar head bias, normally with the Train-label mean."""

        if not math.isfinite(float(value)):
            raise ValueError("output bias must be finite")
        output_layer = self.regression_head[-1]
        if not isinstance(output_layer, nn.Linear) or output_layer.bias is None:
            raise RuntimeError("regression output layer has no bias")
        with torch.no_grad():
            output_layer.bias.fill_(float(value))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> HierarchicalModelOutput:
        encoder_arguments: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_arguments["token_type_ids"] = token_type_ids
        encoder_output = self.encoder(**encoder_arguments)
        cls_embedding = encoder_output.last_hidden_state[:, 0, :]
        z3 = self.projection3(cls_embedding)
        z5 = self.projection5(cls_embedding)
        z7 = self.projection7(cls_embedding)
        logits3 = self.classifier3(z3)
        logits5 = self.classifier5(z5)
        logits7 = self.classifier7(z7)
        fused = torch.cat((cls_embedding, z3, z5, z7), dim=-1)
        scores = self.regression_head(fused).squeeze(-1)
        return HierarchicalModelOutput(
            scores=scores,
            logits3=logits3,
            logits5=logits5,
            logits7=logits7,
            cls_embedding=cls_embedding,
            z3=z3,
            z5=z5,
            z7=z7,
        )


def _validate_loss_shapes(
    output: HierarchicalModelOutput,
    labels19: torch.Tensor,
    labels3: torch.Tensor,
    labels5: torch.Tensor,
    labels7: torch.Tensor,
) -> None:
    batch_size = output.scores.shape[0]
    expected_shapes = {
        "scores": (batch_size,),
        "logits3": (batch_size, 3),
        "logits5": (batch_size, 5),
        "logits7": (batch_size, 7),
        "labels19": (batch_size,),
        "labels3": (batch_size,),
        "labels5": (batch_size,),
        "labels7": (batch_size,),
    }
    actual_shapes = {
        "scores": tuple(output.scores.shape),
        "logits3": tuple(output.logits3.shape),
        "logits5": tuple(output.logits5.shape),
        "logits7": tuple(output.logits7.shape),
        "labels19": tuple(labels19.shape),
        "labels3": tuple(labels3.shape),
        "labels5": tuple(labels5.shape),
        "labels7": tuple(labels7.shape),
    }
    for name, expected in expected_shapes.items():
        if actual_shapes[name] != expected:
            raise ValueError(
                f"{name} must have shape {expected}, got {actual_shapes[name]}"
            )


def hierarchical_huber_aux_loss(
    output: HierarchicalModelOutput,
    labels19: Any,
    labels3: Any,
    labels5: Any,
    labels7: Any,
    *,
    huber_delta: float = 1.0,
    ce3_weight: float = 0.1,
    ce5_weight: float = 0.1,
    ce7_weight: float = 0.1,
) -> HierarchicalLossOutput:
    """Compute FP32 Huber plus weighted 3/5/7-level cross-entropies."""

    if huber_delta <= 0.0:
        raise ValueError("huber_delta must be positive")
    weights = (ce3_weight, ce5_weight, ce7_weight)
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("auxiliary CE weights must be finite and non-negative")

    target19 = _coerce_integral_tensor(labels19, "labels19").to(output.scores.device)
    target3 = _coerce_integral_tensor(labels3, "labels3").to(output.scores.device)
    target5 = _coerce_integral_tensor(labels5, "labels5").to(output.scores.device)
    target7 = _coerce_integral_tensor(labels7, "labels7").to(output.scores.device)
    _validate_loss_shapes(output, target19, target3, target5, target7)
    _validate_range(target19, 1, 19, "labels19")
    _validate_range(target3, 0, 2, "labels3")
    _validate_range(target5, 0, 4, "labels5")
    _validate_range(target7, 0, 6, "labels7")

    huber = F.huber_loss(
        output.scores.float(),
        target19.float(),
        delta=float(huber_delta),
    )
    ce3 = F.cross_entropy(output.logits3.float(), target3)
    ce5 = F.cross_entropy(output.logits5.float(), target5)
    ce7 = F.cross_entropy(output.logits7.float(), target7)
    total = (
        huber
        + float(ce3_weight) * ce3
        + float(ce5_weight) * ce5
        + float(ce7_weight) * ce7
    )
    return HierarchicalLossOutput(total=total, huber=huber, ce3=ce3, ce5=ce5, ce7=ce7)


def _soft_qwk_sufficient_statistics(
    scores: torch.Tensor,
    labels19: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores32 = scores.float()
    centers = torch.arange(1, 20, dtype=torch.float32, device=scores.device)
    logits = -torch.square(scores32.unsqueeze(-1) - centers) / float(temperature)
    probabilities = torch.softmax(logits, dim=-1)
    gold = F.one_hot(labels19 - 1, num_classes=19).to(dtype=torch.float32)
    observed = gold.transpose(0, 1) @ probabilities
    true_histogram = gold.sum(dim=0)
    predicted_histogram = probabilities.sum(dim=0)
    return observed, true_histogram, predicted_histogram


def soft_qwk_loss(
    scores: torch.Tensor,
    labels19: Any,
    *,
    temperature: float = 1.0,
    distributed: bool = True,
    eps: float = 1e-8,
) -> SoftQWKOutput:
    """Return differentiable ``1 - SoftQWK`` using global sufficient stats.

    When DDP is initialized, the observed matrix and histograms are summed with
    the autograd-aware distributed collective.  This is intentionally not a
    mean of per-rank QWK values.
    """

    if scores.ndim != 1:
        raise ValueError(f"scores must have shape [batch], got {tuple(scores.shape)}")
    if scores.numel() == 0:
        raise ValueError("scores must not be empty")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")

    targets = _coerce_integral_tensor(labels19, "labels19").to(scores.device)
    if targets.shape != scores.shape:
        raise ValueError(
            f"labels19 shape {tuple(targets.shape)} does not match "
            f"scores shape {tuple(scores.shape)}"
        )
    _validate_range(targets, 1, 19, "labels19")

    # Autocast is explicitly disabled: SoftQWK's histograms and ratio are more
    # numerically sensitive than the encoder forward pass.
    device_type = scores.device.type
    autocast_device = device_type if device_type in {"cuda", "cpu", "xpu"} else "cpu"
    with torch.autocast(device_type=autocast_device, enabled=False):
        observed, true_histogram, predicted_histogram = (
            _soft_qwk_sufficient_statistics(scores, targets, temperature)
        )
        packed = torch.cat(
            (observed.reshape(-1), true_histogram, predicted_histogram),
            dim=0,
        )
        use_distributed = (
            distributed
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )
        if use_distributed:
            packed = autograd_all_reduce(packed, op=dist.ReduceOp.SUM)
        # Check after the collective so every rank observes and raises for a
        # rank-local NaN/Inf instead of leaving peers blocked in all_reduce.
        if not bool(torch.isfinite(packed).all().item()):
            raise ValueError("scores produced non-finite SoftQWK statistics")

        observed_count = 19 * 19
        global_observed = packed[:observed_count].reshape(19, 19)
        global_true = packed[observed_count : observed_count + 19]
        global_predicted = packed[observed_count + 19 :]
        n_samples_tensor = global_true.sum()

        indices = torch.arange(19, dtype=torch.float32, device=scores.device)
        weights = torch.square(indices[:, None] - indices[None, :]) / float(18**2)
        expected = torch.outer(global_true, global_predicted) / n_samples_tensor
        observed_disagreement_tensor = (weights * global_observed).sum()
        expected_disagreement_tensor = (weights * expected).sum()

        n_samples = int(round(float(n_samples_tensor.detach().cpu().item())))
        n_gold_classes = int((global_true.detach() > 0).sum().cpu().item())
        observed_disagreement = float(
            observed_disagreement_tensor.detach().cpu().item()
        )
        expected_disagreement = float(
            expected_disagreement_tensor.detach().cpu().item()
        )

        fallback_reason: Optional[str] = None
        if n_gold_classes <= 1:
            fallback_reason = "single_gold_class"
        elif (
            not math.isfinite(expected_disagreement)
            or expected_disagreement <= float(eps)
        ):
            fallback_reason = "expected_disagreement_too_small"

        if fallback_reason is not None:
            loss = scores.float().sum() * 0.0
        else:
            loss = observed_disagreement_tensor / expected_disagreement_tensor.clamp_min(
                float(eps)
            )

    return SoftQWKOutput(
        loss=loss,
        used_fallback=fallback_reason is not None,
        fallback_reason=fallback_reason,
        n_samples=n_samples,
        n_gold_classes=n_gold_classes,
        observed_disagreement=observed_disagreement,
        expected_disagreement=expected_disagreement,
    )


def combine_stage2_loss(
    soft_qwk: SoftQWKOutput,
    components: HierarchicalLossOutput,
    *,
    qwk_weight: float = 1.0,
    huber_weight: float = 0.1,
    ce3_weight: float = 0.03,
    ce5_weight: float = 0.03,
    ce7_weight: float = 0.03,
) -> Stage2LossOutput:
    """Combine SoftQWK with unweighted Huber/CE components for Stage 2."""

    weights = (qwk_weight, huber_weight, ce3_weight, ce5_weight, ce7_weight)
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("Stage-2 loss weights must be finite and non-negative")
    total = (
        float(qwk_weight) * soft_qwk.loss
        + float(huber_weight) * components.huber
        + float(ce3_weight) * components.ce3
        + float(ce5_weight) * components.ce5
        + float(ce7_weight) * components.ce7
    )
    return Stage2LossOutput(
        total=total,
        soft_qwk=soft_qwk,
        huber=components.huber,
        ce3=components.ce3,
        ce5=components.ce5,
        ce7=components.ce7,
    )


__all__ = [
    "AuxiliaryLabels",
    "HierarchicalArabicReadabilityRegressor",
    "HierarchicalLossOutput",
    "HierarchicalModelOutput",
    "OFFICIAL_19_TO_3",
    "OFFICIAL_19_TO_5",
    "OFFICIAL_19_TO_7",
    "OFFICIAL_HIERARCHY",
    "SoftQWKOutput",
    "Stage2LossOutput",
    "combine_stage2_loss",
    "derive_auxiliary_labels",
    "hierarchical_huber_aux_loss",
    "map_label19_to_auxiliary",
    "soft_qwk_loss",
    "validate_official_hierarchy_columns",
]
