import pytest
from src.evaluation.metrics import (
    calculate_hit_rate,
    calculate_mrr,
    calculate_recall_at_k,
    calculate_precision_at_k
)

def test_hit_rate():
    assert calculate_hit_rate(["c1", "c2"], ["c2", "c3"]) == True
    assert calculate_hit_rate(["c1", "c2"], ["c3", "c4"]) == False

def test_mrr():
    assert calculate_mrr(["c1", "c2", "c3"], ["c2"]) == 0.5
    assert calculate_mrr(["c1", "c2", "c3"], ["c1"]) == 1.0
    assert calculate_mrr(["c1", "c2", "c3"], ["c4"]) == 0.0

def test_recall_at_k():
    assert calculate_recall_at_k(["c1", "c2", "c3"], ["c2", "c3", "c4"]) == 2.0 / 3.0

def test_precision_at_k():
    assert calculate_precision_at_k(["c1", "c2", "c3"], ["c2", "c3", "c4"]) == 2.0 / 3.0
