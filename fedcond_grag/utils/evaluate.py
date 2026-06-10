import re
import string
from collections import Counter

import pandas as pd


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    return normalize(s2) in normalize(s1)


def exact_match(prediction: str, answer: str) -> bool:
    """SQuAD-style EM: normalized strings must be identical."""
    return normalize(prediction) == normalize(answer)


def token_f1(prediction: str, answer: str) -> float:
    """SQuAD/MuSiQue-style token-level F1 on normalized strings."""
    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(answer).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    return 2 * precision * recall / (precision + recall), precision, recall


def eval_acc(prediction, answer):
    matched = 0.0
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)


def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0


def get_accuracy_fedcond_qa(path):
    df = pd.read_json(path, lines=True)

    acc_list, hit_list, f1_list = [], [], []
    precission_list, recall_list = [], []

    for prediction, answer in zip(df.pred.tolist(), df.label.tolist()):
        prediction = prediction.replace("|", "\n")
        answer = answer.split("|")

        prediction = prediction.split("\n")
        f1_score, precision_score, recall_score = eval_f1(prediction, answer)
        f1_list.append(f1_score)
        precission_list.append(precision_score)
        recall_list.append(recall_score)
        prediction_str = " ".join(prediction)
        acc_list.append(eval_acc(prediction_str, answer))
        hit_list.append(eval_hit(prediction_str, answer))

    acc = sum(acc_list) * 100 / len(acc_list)
    hit = sum(hit_list) * 100 / len(hit_list)
    f1 = sum(f1_list) * 100 / len(f1_list)
    pre = sum(precission_list) * 100 / len(precission_list)
    recall = sum(recall_list) * 100 / len(recall_list)

    print(f"Accuracy: {acc:.4f}")
    print(f"Hit: {hit:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    return hit


eval_funcs = {
    "fedcond_qa": get_accuracy_fedcond_qa,
}
