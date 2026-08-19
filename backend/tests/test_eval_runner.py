"""Tests for `evals/eval_runner.py`'s `score_answer` expected_divergence relabeling."""

from __future__ import annotations

from evals.eval_runner import Verdict, score_answer


def test_expected_divergence_pass_is_labelled_pass_divergent() -> None:
    """An answer matching ground truth despite `expected_divergence` gets its own label."""
    question = {
        "expected_answer_shape": "plain_value",
        "ground_truth_values": ["42"],
        "expected_divergence": "synthetic: answer is expected to diverge from ground truth",
    }
    result = score_answer(question, "The reported value is 42.")

    assert result.verdict is Verdict.PASS_DIVERGENT
    assert "expected divergence" in result.detail


def test_expected_divergence_fail_is_still_labelled_divergent() -> None:
    """Regression guard: a genuine mismatch on an `expected_divergence` entry still relabels
    to DIVERGENT, unchanged from before PASS_DIVERGENT was added."""
    question = {
        "expected_answer_shape": "plain_value",
        "ground_truth_values": ["42"],
        "expected_divergence": "synthetic: answer is expected to diverge from ground truth",
    }
    result = score_answer(question, "The reported value is 7.")

    assert result.verdict is Verdict.DIVERGENT
    assert "expected divergence" in result.detail
