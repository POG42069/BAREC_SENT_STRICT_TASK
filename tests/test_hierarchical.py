from __future__ import annotations

import os
import queue
import socket
import sys
import time
import traceback
import unittest
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from hierarchical import (
    OFFICIAL_19_TO_3,
    OFFICIAL_19_TO_5,
    OFFICIAL_19_TO_7,
    HierarchicalArabicReadabilityRegressor,
    derive_auxiliary_labels,
    hierarchical_huber_aux_loss,
    map_label19_to_auxiliary,
    soft_qwk_loss,
    validate_official_hierarchy_columns,
)


class DummyEncoder(nn.Module):
    """Small trainable encoder with the Hugging Face output contract."""

    def __init__(self, hidden_size: int = 12, vocabulary_size: int = 32) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.transform = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del attention_mask, token_type_ids
        hidden = torch.tanh(self.transform(self.embedding(input_ids)))
        return SimpleNamespace(last_hidden_state=hidden)


def make_model() -> HierarchicalArabicReadabilityRegressor:
    torch.manual_seed(7)
    return HierarchicalArabicReadabilityRegressor(
        model_name="unused-dummy-model",
        dropout=0.0,
        output_bias=10.5,
        projection_size=5,
        fusion_hidden_size=9,
        encoder=DummyEncoder(hidden_size=12),
    )


def make_inputs(batch_size: int, sequence_length: int = 4) -> dict[str, torch.Tensor]:
    input_ids = torch.arange(batch_size * sequence_length, dtype=torch.long)
    input_ids = input_ids.reshape(batch_size, sequence_length) % 32
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "token_type_ids": torch.zeros_like(input_ids),
    }


_DISTRIBUTED_LABEL_PARTS = (
    (1, 4, 8, 11),
    (12, 14, 17, 19),
)
_DISTRIBUTED_SCORE_PARTS = (
    (1.2, 3.7, 8.5, 10.3),
    (12.2, 13.4, 16.6, 18.8),
)
_DISTRIBUTED_TEMPERATURE = 0.7


def _find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _gloo_soft_qwk_worker(
    rank: int,
    world_size: int,
    port: int,
    result_queue: mp.Queue,
) -> None:
    """Compute one rank's global SoftQWK and return only serializable values."""

    try:
        torch.set_num_threads(1)
        if not sys.platform.startswith("win"):
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=30),
        )
        local_scores = torch.tensor(
            _DISTRIBUTED_SCORE_PARTS[rank],
            dtype=torch.float32,
            requires_grad=True,
        )
        local_labels = torch.tensor(
            _DISTRIBUTED_LABEL_PARTS[rank],
            dtype=torch.long,
        )
        output = soft_qwk_loss(
            local_scores,
            local_labels,
            temperature=_DISTRIBUTED_TEMPERATURE,
            distributed=True,
        )
        output.loss.backward()
        if local_scores.grad is None:
            raise AssertionError("Distributed SoftQWK did not produce local gradients")
        result_queue.put(
            {
                "status": "ok",
                "rank": rank,
                "loss": float(output.loss.detach()),
                "gradient": local_scores.grad.detach().cpu().tolist(),
                "used_fallback": output.used_fallback,
                "n_samples": output.n_samples,
                "n_gold_classes": output.n_gold_classes,
            }
        )
    except BaseException:
        result_queue.put(
            {
                "status": "error",
                "rank": rank,
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


class OfficialHierarchyMappingTests(unittest.TestCase):
    def test_all_19_levels_map_to_official_zero_based_targets(self) -> None:
        labels19 = torch.arange(1, 20, dtype=torch.long)
        expected = {
            3: torch.tensor(OFFICIAL_19_TO_3),
            5: torch.tensor(OFFICIAL_19_TO_5),
            7: torch.tensor(OFFICIAL_19_TO_7),
        }

        for levels, expected_targets in expected.items():
            with self.subTest(levels=levels):
                actual = map_label19_to_auxiliary(labels19, levels)
                self.assertTrue(torch.equal(actual, expected_targets))
                self.assertEqual(actual.dtype, torch.long)
                self.assertGreaterEqual(int(actual.min()), 0)
                self.assertLess(int(actual.max()), levels)

        derived = derive_auxiliary_labels(labels19)
        self.assertTrue(torch.equal(derived.label3, expected[3]))
        self.assertTrue(torch.equal(derived.label5, expected[5]))
        self.assertTrue(torch.equal(derived.label7, expected[7]))

    def test_official_columns_validate_and_one_mismatch_fails(self) -> None:
        labels19 = torch.arange(1, 20, dtype=torch.long)
        auxiliary = derive_auxiliary_labels(labels19)
        labels3 = auxiliary.label3 + 1
        labels5 = auxiliary.label5 + 1
        labels7 = auxiliary.label7 + 1

        validate_official_hierarchy_columns(labels19, labels3, labels5, labels7)

        mismatched5 = labels5.clone()
        mismatched5[11] = 4
        with self.assertRaisesRegex(ValueError, "disagrees with the official mapping"):
            validate_official_hierarchy_columns(
                labels19,
                labels3,
                mismatched5,
                labels7,
            )


class HierarchicalModelTests(unittest.TestCase):
    def test_batch_one_shapes_and_outputs_are_raw_logits(self) -> None:
        model = make_model()
        model.eval()
        known_biases = {
            "classifier3": torch.tensor([-2.0, 0.5, 3.0]),
            "classifier5": torch.tensor([-3.0, -1.0, 0.25, 2.0, 4.0]),
            "classifier7": torch.tensor(
                [-4.0, -2.0, -0.5, 0.25, 1.5, 3.0, 5.0]
            ),
        }
        with torch.no_grad():
            for name, biases in known_biases.items():
                classifier = getattr(model, name)
                classifier.weight.zero_()
                classifier.bias.copy_(biases)

        output = model(**make_inputs(batch_size=1))

        self.assertEqual(tuple(output.cls_embedding.shape), (1, 12))
        self.assertEqual(tuple(output.z3.shape), (1, 5))
        self.assertEqual(tuple(output.z5.shape), (1, 5))
        self.assertEqual(tuple(output.z7.shape), (1, 5))
        self.assertEqual(tuple(output.logits3.shape), (1, 3))
        self.assertEqual(tuple(output.logits5.shape), (1, 5))
        self.assertEqual(tuple(output.logits7.shape), (1, 7))
        self.assertEqual(tuple(output.scores.shape), (1,))

        for name, biases in known_biases.items():
            logits = getattr(output, name.replace("classifier", "logits"))
            self.assertTrue(torch.equal(logits[0], biases))
            self.assertTrue(bool((logits < 0).any()))
            self.assertFalse(
                torch.allclose(
                    logits.sum(dim=-1),
                    torch.ones(1, dtype=logits.dtype),
                )
            )

        ce = F.cross_entropy(output.logits3, torch.tensor([2]))
        self.assertTrue(bool(torch.isfinite(ce)))

    def test_regression_and_auxiliary_gradients_are_connected(self) -> None:
        model = make_model()
        labels19 = torch.tensor([1, 7, 13, 19])
        auxiliary = derive_auxiliary_labels(labels19)

        output = model(**make_inputs(batch_size=4))
        huber = F.huber_loss(output.scores, labels19.float(), delta=1.0)
        huber.backward()
        for projection_name in ("projection3", "projection5", "projection7"):
            projection = getattr(model, projection_name)
            gradients = [
                parameter.grad
                for parameter in projection.parameters()
                if parameter.requires_grad
            ]
            self.assertTrue(all(gradient is not None for gradient in gradients))
            self.assertGreater(
                sum(float(gradient.abs().sum()) for gradient in gradients), 0.0
            )

        model.zero_grad(set_to_none=True)
        output = model(**make_inputs(batch_size=4))
        F.cross_entropy(output.logits3, auxiliary.label3).backward()
        for module_name in ("projection3", "classifier3"):
            module = getattr(model, module_name)
            gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            self.assertTrue(all(gradient is not None for gradient in gradients))
            self.assertGreater(
                sum(float(gradient.abs().sum()) for gradient in gradients), 0.0
            )

        model.zero_grad(set_to_none=True)
        output = model(**make_inputs(batch_size=4))
        losses = hierarchical_huber_aux_loss(
            output,
            labels19,
            auxiliary.label3,
            auxiliary.label5,
            auxiliary.label7,
        )
        self.assertTrue(bool(torch.isfinite(losses.total)))
        losses.total.backward()
        disconnected = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        self.assertEqual(disconnected, [])
        non_finite = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
        ]
        self.assertEqual(non_finite, [])


class SoftQWKTests(unittest.TestCase):
    def test_missing_classes_is_finite_and_differentiable(self) -> None:
        labels = torch.tensor([1, 3, 7, 11, 15, 19])
        scores = (labels.float() + torch.tensor([0.1, -0.2, 0.3, -0.1, 0.2, -0.3]))
        scores.requires_grad_()

        output = soft_qwk_loss(scores, labels, distributed=False)

        self.assertFalse(output.used_fallback)
        self.assertEqual(output.n_gold_classes, 6)
        self.assertTrue(bool(torch.isfinite(output.loss)))
        output.loss.backward()
        self.assertIsNotNone(scores.grad)
        self.assertTrue(bool(torch.isfinite(scores.grad).all()))
        self.assertGreater(float(scores.grad.abs().sum()), 0.0)

    def test_single_gold_class_uses_finite_connected_fallback(self) -> None:
        labels = torch.full((5,), 12, dtype=torch.long)
        scores = torch.linspace(10.0, 14.0, steps=5, requires_grad=True)

        output = soft_qwk_loss(scores, labels, distributed=False)

        self.assertTrue(output.used_fallback)
        self.assertEqual(output.fallback_reason, "single_gold_class")
        self.assertEqual(output.n_gold_classes, 1)
        self.assertTrue(bool(torch.isfinite(output.loss)))
        output.loss.backward()
        self.assertIsNotNone(scores.grad)
        self.assertTrue(bool(torch.isfinite(scores.grad).all()))

    def test_extreme_fp16_scores_remain_finite(self) -> None:
        labels = torch.tensor([1, 5, 15, 19])
        scores = torch.tensor(
            [-60000.0, -1000.0, 1000.0, 60000.0],
            dtype=torch.float16,
            requires_grad=True,
        )

        output = soft_qwk_loss(scores, labels, distributed=False)

        self.assertFalse(output.used_fallback)
        self.assertEqual(output.loss.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(output.loss)))
        output.loss.backward()
        self.assertIsNotNone(scores.grad)
        self.assertTrue(bool(torch.isfinite(scores.grad).all()))

    def test_loss_orders_perfect_shifted_and_reversed_predictions(self) -> None:
        labels = torch.arange(1, 20, dtype=torch.long)
        perfect = labels.float()
        shifted = torch.clamp(labels.float() + 3.0, 1.0, 19.0)
        reversed_scores = 20.0 - labels.float()

        perfect_loss = soft_qwk_loss(
            perfect, labels, temperature=0.5, distributed=False
        ).loss
        shifted_loss = soft_qwk_loss(
            shifted, labels, temperature=0.5, distributed=False
        ).loss
        reversed_loss = soft_qwk_loss(
            reversed_scores, labels, temperature=0.5, distributed=False
        ).loss

        self.assertLess(float(perfect_loss), float(shifted_loss))
        self.assertLess(float(shifted_loss), float(reversed_loss))

    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(),
        "PyTorch Gloo backend is unavailable",
    )
    @unittest.skipIf(
        sys.platform.startswith("win"),
        (
            "This Windows PyTorch build advertises Gloo but cannot construct a "
            "Gloo network device; the real two-rank test runs on Linux/Kaggle"
        ),
    )
    def test_two_rank_gloo_matches_global_reference_and_gradient_scaling(self) -> None:
        world_size = 2
        all_scores = [
            score for rank_scores in _DISTRIBUTED_SCORE_PARTS for score in rank_scores
        ]
        all_labels = [
            label for rank_labels in _DISTRIBUTED_LABEL_PARTS for label in rank_labels
        ]
        reference_scores = torch.tensor(
            all_scores,
            dtype=torch.float32,
            requires_grad=True,
        )
        reference_labels = torch.tensor(all_labels, dtype=torch.long)
        reference = soft_qwk_loss(
            reference_scores,
            reference_labels,
            temperature=_DISTRIBUTED_TEMPERATURE,
            distributed=False,
        )
        reference.loss.backward()
        self.assertIsNotNone(reference_scores.grad)
        reference_gradient = reference_scores.grad.detach()

        context = mp.get_context("spawn")
        result_queue = context.Queue()
        port = _find_free_local_port()
        processes = [
            context.Process(
                target=_gloo_soft_qwk_worker,
                args=(rank, world_size, port, result_queue),
            )
            for rank in range(world_size)
        ]
        started: list[mp.Process] = []
        try:
            for process in processes:
                process.start()
                started.append(process)

            deadline = time.monotonic() + 45.0
            for process in started:
                process.join(timeout=max(0.0, deadline - time.monotonic()))

            timed_out = [process for process in started if process.is_alive()]
            if timed_out:
                for process in timed_out:
                    process.terminate()
                for process in timed_out:
                    process.join(timeout=5.0)
                self.fail(
                    "Two-rank Gloo SoftQWK test exceeded its 45-second timeout"
                )

            messages = []
            for _ in range(world_size):
                try:
                    messages.append(result_queue.get(timeout=5.0))
                except queue.Empty:
                    break

            self.assertEqual(
                len(messages),
                world_size,
                msg=(
                    "Gloo workers did not all report results; "
                    f"exit codes={[process.exitcode for process in started]}"
                ),
            )
            errors = [
                message for message in messages if message.get("status") != "ok"
            ]
            self.assertEqual(
                errors,
                [],
                msg="\n".join(message.get("traceback", str(message)) for message in errors),
            )
            self.assertTrue(all(process.exitcode == 0 for process in started))

            chunk_size = len(_DISTRIBUTED_SCORE_PARTS[0])
            for message in sorted(messages, key=lambda item: item["rank"]):
                rank = int(message["rank"])
                self.assertFalse(message["used_fallback"])
                self.assertEqual(message["n_samples"], len(all_labels))
                self.assertEqual(message["n_gold_classes"], len(set(all_labels)))
                self.assertAlmostEqual(
                    message["loss"],
                    float(reference.loss.detach()),
                    places=6,
                )
                start = rank * chunk_size
                expected_gradient = (
                    reference_gradient[start : start + chunk_size] * world_size
                )
                actual_gradient = torch.tensor(message["gradient"])
                self.assertTrue(
                    torch.allclose(
                        actual_gradient,
                        expected_gradient,
                        rtol=1e-5,
                        atol=1e-6,
                    ),
                    msg=(
                        f"rank {rank} gradient mismatch: "
                        f"actual={actual_gradient.tolist()} "
                        f"expected={expected_gradient.tolist()}"
                    ),
                )
        finally:
            for process in started:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5.0)
            result_queue.close()
            result_queue.join_thread()


if __name__ == "__main__":
    unittest.main()
