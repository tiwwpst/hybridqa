from __future__ import annotations

import argparse
import json
import re
import string
import urllib.request
from pathlib import Path


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize(prediction).split()
    truth_tokens = normalize(ground_truth).split()
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1
    num_same = 0
    for token in truth_tokens:
        if common.get(token, 0) > 0:
            num_same += 1
            common[token] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize(prediction) == normalize(ground_truth))


def best_score(prediction: str, answers: list[str]) -> tuple[float, float]:
    em = 0.0
    f1 = 0.0
    for ans in answers:
        em = max(em, exact_match(prediction, ans))
        f1 = max(f1, f1_score(prediction, ans))
    return em, f1


def call_api(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate API on HybridQA dev split.")
    parser.add_argument("--data", default="data/hybridqa/HybridQA/released_data/dev.json")
    parser.add_argument("--api-url", default="http://localhost:8000/ask")
    parser.add_argument("--out", default="predictions.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Missing data file: {data_path}")

    data = json.loads(data_path.read_text(encoding="utf-8"))
    if args.limit:
        data = data[: args.limit]

    predictions = []
    total_em = 0.0
    total_f1 = 0.0

    for item in data:
        payload = {
            "question": item["question"],
            "table_id": item["table_id"],
        }
        result = call_api(args.api_url, payload, args.timeout)
        pred = result.get("payload", {}).get("final_answer", "")
        answers = item.get("answer-text")
        if isinstance(answers, list):
            answer_list = [str(a) for a in answers]
        else:
            answer_list = [str(answers)] if answers is not None else [""]
        em, f1 = best_score(pred, answer_list)
        total_em += em
        total_f1 += f1
        predictions.append({"question_id": item.get("question_id"), "prediction": pred})

    count = max(len(data), 1)
    metrics = {"em": total_em / count, "f1": total_f1 / count, "count": count}
    Path(args.out).write_text(json.dumps({"metrics": metrics, "predictions": predictions}, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
