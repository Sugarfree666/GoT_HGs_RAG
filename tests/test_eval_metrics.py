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


def test_hyperrag_em_uses_substring_match() -> None:
    module = load_eval_module()

    assert module.cal_em([["Hoboken"]], ["Hoboken, New Jersey"]) == 1.0
    assert module.cal_em([["no"]], ["No, they do not have the same nationality."]) == 1.0
    assert module.cal_em([["Jorge Ledezma"]], ["Jorge Ledezma was born first on 24 August 1963."]) == 1.0


def test_hyperrag_f1_remains_token_level() -> None:
    module = load_eval_module()

    assert module.cal_f1([["Hoboken"]], ["Hoboken, New Jersey"]) == 0.5
