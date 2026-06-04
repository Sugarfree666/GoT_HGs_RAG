from __future__ import annotations

import re
import string
from collections import Counter
from typing import Callable

import numpy as np


def normalize_answer(answer: str) -> str:
    """Lowercase, remove punctuation/articles, and normalize whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(str(answer).lower())))


def exact_match(predicted: str, gold: str) -> float:
    """Normalized exact match."""

    predicted_normalized = normalize_answer(predicted)
    gold_normalized = normalize_answer(gold)
    if not predicted_normalized or not gold_normalized:
        return 0.0
    return 1.0 if predicted_normalized == gold_normalized else 0.0


def token_f1(predicted: str, gold: str) -> float:
    """Token-level F1 over normalized answer strings."""

    predicted_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()
    if not predicted_tokens or not gold_tokens:
        return 0.0

    common = Counter(predicted_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(predicted_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def calculate_metric_scores_em(
    gold_answers: list[list[str]],
    predicted_answers: list[str],
    aggregation_fn: Callable[[list[float]], float],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    assert len(gold_answers) == len(predicted_answers), (
        "Length of gold answers and predicted answers should be the same."
    )

    example_eval_results: list[dict[str, float]] = []
    total_em = 0.0

    for gold_list, predicted in zip(gold_answers, predicted_answers):
        em_scores = [exact_match(predicted, gold) for gold in gold_list]
        aggregated_em = float(aggregation_fn(em_scores)) if em_scores else 0.0
        example_eval_results.append({"ExactMatch": aggregated_em})
        total_em += aggregated_em

    avg_em = total_em / len(gold_answers) if gold_answers else 0.0
    return {"ExactMatch": avg_em}, example_eval_results


def calculate_metric_scores_f1(
    gold_answers: list[list[str]],
    predicted_answers: list[str],
    aggregation_fn: Callable[[list[float]], float],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    assert len(gold_answers) == len(predicted_answers), (
        "Length of gold answers and predicted answers should be the same."
    )

    example_eval_results: list[dict[str, float]] = []
    total_f1 = 0.0

    for gold_list, predicted in zip(gold_answers, predicted_answers):
        f1_scores = [token_f1(predicted, gold) for gold in gold_list]
        aggregated_f1 = float(aggregation_fn(f1_scores)) if f1_scores else 0.0
        example_eval_results.append({"F1": aggregated_f1})
        total_f1 += aggregated_f1

    avg_f1 = total_f1 / len(gold_answers) if gold_answers else 0.0
    return {"F1": avg_f1}, example_eval_results


def cal_em(gold_answers: list[list[str]], predicted_answers: list[str]) -> float:
    overall_qa_em_result, _ = calculate_metric_scores_em(
        gold_answers=gold_answers,
        predicted_answers=predicted_answers,
        aggregation_fn=np.max,
    )
    return overall_qa_em_result["ExactMatch"]


def cal_f1(gold_answers: list[list[str]], predicted_answers: list[str]) -> float:
    overall_qa_f1_result, _ = calculate_metric_scores_f1(
        gold_answers=gold_answers,
        predicted_answers=predicted_answers,
        aggregation_fn=np.max,
    )
    return overall_qa_f1_result["F1"]
