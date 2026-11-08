
from src.evaluate_and_register import evaluate_quality_gates


def test_quality_gate_pass():
    result = evaluate_quality_gates(
        rmse=0.80,
        baseline_rmse=1.00
    )

    assert result["absolute_gate_passed"] is True
    assert result["relative_gate_status"] == "PASS"
    assert result["passed"] is True
    assert result["status"] == "PASS"

def test_quality_gate_fail_relative():
    result = evaluate_quality_gates(
        rmse=0.95,
        baseline_rmse=0.96
    )

    assert result["absolute_gate_passed"] is True
    assert result["relative_gate_status"] == "FAIL"
    assert result["passed"] is False
    assert result["status"] == "FAIL"

def test_quality_gate_blocked_without_baseline():
    result = evaluate_quality_gates(rmse=0.80)

    assert result["absolute_gate_passed"] is True
    assert result["relative_gate_status"] == "NOT_EVALUATED"
    assert result["passed"] is False
    assert result["status"] == "BLOCKED"


def test_quality_gate_fail_absolute():
    result = evaluate_quality_gates(
        rmse=0.99,
        baseline_rmse=1.00
    )

    assert result["absolute_gate_passed"] is False
    assert result["relative_gate_status"] == "FAIL"
    assert result["passed"] is False
    assert result["status"] == "FAIL"
