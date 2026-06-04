from __future__ import annotations

import importlib.util
from pathlib import Path


def load_eval_module():
    module_path = Path(__file__).resolve().parents[1] / "eval" / "eval.py"
    spec = importlib.util.spec_from_file_location("eval_metrics", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_em_uses_normalized_exact_match() -> None:
    module = load_eval_module()

    assert module.cal_em([["The Hoboken!"]], ["hoboken"]) == 1.0
    assert module.cal_em([["Hoboken"]], ["Hoboken, New Jersey"]) == 0.0
    assert module.cal_em([["no"]], ["No, they do not have the same nationality."]) == 0.0


def test_f1_uses_normalized_token_overlap() -> None:
    module = load_eval_module()

    assert module.cal_f1([["Hoboken"]], ["Hoboken, New Jersey"]) == 0.5
