"""Focused tests for the baseline's QWK fine-tuning objective.

The tests import only pure loss/selection helpers from ``train.py``.  They do
not instantiate a tokenizer or model, so running this module never downloads
Hugging Face or CAMeL resources.
"""

from __future__ import annotations

import math
import os
import queue
import socket
import sys
import time
import traceback
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from train import (
    QWKFinetuneLossOutput,
    SoftQWKOutput,
    is_better_checkpoint,
    qwk_finetune_loss,
    soft_qwk_loss,
)


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
    """Return one rank's global loss and local score gradients."""

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
        local_labels = torch.tensor(_DISTRIBUTED_LABEL_PARTS[rank])
        output = soft_qwk_loss(
            local_scores,
            local_labels,
            temperature=_DISTRIBUTED_TEMPERATURE,
            distributed=True,
        )
        output.loss.backward()
        if local_scores.grad is None:
            raise AssertionError("Distributed SoftQWK produced no local gradient")
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


class SoftQWKLossTests(unittest.TestCase):
    def test_perfect_is_better_than_shifted_and_reversed(self) -> None:
        labels = torch.arange(1, 20, dtype=torch.long)
        perfect = labels.float()
        shifted = torch.clamp(labels.float() + 3.0, 1.0, 19.0)
        reversed_scores = 20.0 - labels.float()

        perfect_output = soft_qwk_loss(
            perfect, labels, temperature=0.5, distributed=False
        )
        shifted_output = soft_qwk_loss(
            shifted, labels, temperature=0.5, distributed=False
        )
        reversed_output = soft_qwk_loss(
            reversed_scores, labels, temperature=0.5, distributed=False
        )

        self.assertIsInstance(perfect_output, SoftQWKOutput)
        self.assertLess(float(perfect_output.loss), float(shifted_output.loss))
        self.assertLess(float(shifted_output.loss), float(reversed_output.loss))

    def test_missing_classes_is_finite_and_differentiable(self) -> None:
        labels = torch.tensor([1, 3, 7, 11, 15, 19])
        offsets = torch.tensor([0.1, -0.2, 0.3, -0.1, 0.2, -0.3])
        scores = (labels.float() + offsets).requires_grad_()

        output = soft_qwk_loss(scores, labels, distributed=False)

        self.assertFalse(output.used_fallback)
        self.assertIsNone(output.fallback_reason)
        self.assertEqual(output.n_samples, labels.numel())
        self.assertEqual(output.n_gold_classes, 6)
        self.assertTrue(bool(torch.isfinite(output.loss)))
        output.loss.backward()
        self.assertIsNotNone(scores.grad)
        self.assertTrue(bool(torch.isfinite(scores.grad).all()))
        self.assertGreater(float(scores.grad.abs().sum()), 0.0)

    def test_single_gold_class_returns_connected_zero(self) -> None:
        labels = torch.full((5,), 12, dtype=torch.long)
        scores = torch.linspace(10.0, 14.0, steps=5, requires_grad=True)

        output = soft_qwk_loss(scores, labels, distributed=False)

        self.assertTrue(output.used_fallback)
        self.assertEqual(output.fallback_reason, "single_gold_class")
        self.assertEqual(output.n_gold_classes, 1)
        self.assertEqual(float(output.loss.detach()), 0.0)
        self.assertTrue(output.loss.requires_grad)
        output.loss.backward()
        self.assertIsNotNone(scores.grad)
        self.assertTrue(bool(torch.isfinite(scores.grad).all()))
        self.assertEqual(float(scores.grad.abs().sum()), 0.0)

    def test_extreme_fp16_scores_remain_finite_in_fp32(self) -> None:
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

    def test_rejects_invalid_scores_labels_and_hyperparameters(self) -> None:
        valid_scores = torch.tensor([1.0, 2.0])
        valid_labels = torch.tensor([1, 2])
        invalid_calls = (
            lambda: soft_qwk_loss(
                torch.ones(2, 1), valid_labels, distributed=False
            ),
            lambda: soft_qwk_loss(
                torch.empty(0), torch.empty(0, dtype=torch.long), distributed=False
            ),
            lambda: soft_qwk_loss(
                valid_scores, torch.tensor([1]), distributed=False
            ),
            lambda: soft_qwk_loss(
                valid_scores, torch.tensor([1.0, 2.5]), distributed=False
            ),
            lambda: soft_qwk_loss(
                valid_scores, torch.tensor([0, 20]), distributed=False
            ),
            lambda: soft_qwk_loss(
                torch.tensor([1.0, math.inf]), valid_labels, distributed=False
            ),
            lambda: soft_qwk_loss(
                valid_scores, valid_labels, temperature=0.0, distributed=False
            ),
            lambda: soft_qwk_loss(
                valid_scores, valid_labels, temperature=math.nan, distributed=False
            ),
            lambda: soft_qwk_loss(
                valid_scores, valid_labels, eps=0.0, distributed=False
            ),
            lambda: soft_qwk_loss(
                valid_scores, valid_labels, eps=math.nan, distributed=False
            ),
        )

        for index, invalid_call in enumerate(invalid_calls):
            with self.subTest(case=index), self.assertRaises(ValueError):
                invalid_call()

    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(),
        "PyTorch Gloo backend is unavailable",
    )
    @unittest.skipIf(
        sys.platform.startswith("win"),
        (
            "This Windows PyTorch build may advertise Gloo but cannot reliably "
            "construct a two-rank network device; run this test on Linux/Kaggle"
        ),
    )
    def test_two_rank_gloo_matches_concatenated_reference(self) -> None:
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
        reference_labels = torch.tensor(all_labels)
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
                    process.join(timeout=5.0)
                self.fail("Two-rank Gloo SoftQWK test exceeded 45 seconds")

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
                    "Not all Gloo workers reported; "
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

            chunk_size = len(_DISTRIBUTED_SCORE_PARTS[0])
            for message in sorted(messages, key=lambda item: item["rank"]):
                rank = int(message["rank"])
                self.assertFalse(message["used_fallback"])
                self.assertEqual(message["n_samples"], len(all_labels))
                self.assertEqual(message["n_gold_classes"], len(set(all_labels)))
                self.assertAlmostEqual(
                    message["loss"], float(reference.loss.detach()), places=6
                )
                start = rank * chunk_size
                # autograd-aware all_reduce sums the identical loss gradient
                # emitted by each rank. DDP later averages parameter gradients.
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
                        f"rank {rank}: actual={actual_gradient.tolist()} "
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


class QWKFinetuneLossTests(unittest.TestCase):
    def test_exact_soft_qwk_plus_weighted_mse_composition(self) -> None:
        labels = torch.tensor([1, 4, 8, 12, 16, 19])
        scores = torch.tensor(
            [1.4, 3.7, 8.8, 11.2, 15.6, 18.1],
            requires_grad=True,
        )
        mse_weight = 0.05

        output = qwk_finetune_loss(
            scores,
            labels,
            mse_weight=mse_weight,
            temperature=0.8,
            distributed=False,
        )

        self.assertIsInstance(output, QWKFinetuneLossOutput)
        self.assertIsInstance(output.soft_qwk, SoftQWKOutput)
        expected_mse = torch.mean(torch.square(scores.float() - labels.float()))
        expected_total = output.soft_qwk.loss + mse_weight * expected_mse
        self.assertTrue(torch.allclose(output.mse, expected_mse, rtol=0.0, atol=1e-7))
        self.assertTrue(
            torch.allclose(output.total, expected_total, rtol=0.0, atol=1e-7)
        )
        output.total.backward()
        self.assertIsNotNone(scores.grad)
        self.assertTrue(bool(torch.isfinite(scores.grad).all()))
        self.assertGreater(float(scores.grad.abs().sum()), 0.0)

    def test_single_class_fallback_keeps_exact_mse_gradient(self) -> None:
        labels = torch.full((4,), 12, dtype=torch.long)
        scores = torch.tensor([9.0, 11.0, 13.0, 15.0], requires_grad=True)
        mse_weight = 0.05

        output = qwk_finetune_loss(
            scores,
            labels,
            mse_weight=mse_weight,
            distributed=False,
        )

        self.assertTrue(output.soft_qwk.used_fallback)
        self.assertEqual(output.soft_qwk.fallback_reason, "single_gold_class")
        self.assertEqual(float(output.soft_qwk.loss.detach()), 0.0)
        self.assertTrue(
            torch.allclose(
                output.total,
                mse_weight * output.mse,
                rtol=0.0,
                atol=1e-7,
            )
        )
        output.total.backward()
        expected_gradient = (
            mse_weight
            * 2.0
            * (scores.detach() - labels.float())
            / float(labels.numel())
        )
        self.assertIsNotNone(scores.grad)
        self.assertTrue(
            torch.allclose(scores.grad, expected_gradient, rtol=1e-6, atol=1e-7)
        )

    def test_extreme_fp16_total_and_components_are_finite_fp32(self) -> None:
        labels = torch.tensor([1, 5, 15, 19])
        scores = torch.tensor(
            [-60000.0, -1000.0, 1000.0, 60000.0],
            dtype=torch.float16,
            requires_grad=True,
        )

        output = qwk_finetune_loss(scores, labels, distributed=False)

        self.assertEqual(output.soft_qwk.loss.dtype, torch.float32)
        self.assertEqual(output.mse.dtype, torch.float32)
        self.assertEqual(output.total.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(output.soft_qwk.loss)))
        self.assertTrue(bool(torch.isfinite(output.mse)))
        self.assertTrue(bool(torch.isfinite(output.total)))

    def test_rejects_invalid_mse_weight(self) -> None:
        scores = torch.tensor([1.0, 2.0])
        labels = torch.tensor([1, 2])
        for mse_weight in (-0.01, math.inf, math.nan):
            with self.subTest(mse_weight=mse_weight), self.assertRaises(ValueError):
                qwk_finetune_loss(
                    scores,
                    labels,
                    mse_weight=mse_weight,
                    distributed=False,
                )


class CheckpointSelectionTests(unittest.TestCase):
    def test_first_candidate_is_selected(self) -> None:
        self.assertTrue(
            is_better_checkpoint(
                0.10,
                3.0,
                0.90,
                0.5,
                has_selected_model=False,
            )
        )

    def test_qwk_has_priority_over_mae(self) -> None:
        self.assertTrue(is_better_checkpoint(0.81, 2.0, 0.80, 0.5))
        self.assertFalse(is_better_checkpoint(0.79, 0.1, 0.80, 3.0))

    def test_mae_breaks_exact_qwk_tie(self) -> None:
        self.assertTrue(is_better_checkpoint(0.80, 1.1, 0.80, 1.2))
        self.assertFalse(is_better_checkpoint(0.80, 1.3, 0.80, 1.2))

    def test_full_tie_keeps_existing_checkpoint(self) -> None:
        self.assertFalse(is_better_checkpoint(0.80, 1.2, 0.80, 1.2))

    def test_non_finite_candidate_qwk_cannot_replace_selected_model(self) -> None:
        self.assertFalse(is_better_checkpoint(math.nan, 0.1, 0.80, 1.2))


if __name__ == "__main__":
    unittest.main()
