"""Question-ownership + partition-statistics report for the topic-skew
partitioning experiment (see fedcond_grag/dataloader/topic_partition.py).

Question-to-client assignment is FIXED (question index % num_clients, the
same rule this repo already uses by default for training splits and
eval routing) and reused identically across every partition setting -- only
which client owns which article/passage changes. A question is
local-answerable if every one of its gold_titles is owned by its assigned
client under the given partition; otherwise it is cross-client. Recomputed
per partition setting from partition_diagnostics_<mode>_<alpha>.json's
article_to_client manifest (dataset/linearrag/<ds>/questions.json's
gold_titles, fixed across settings).

Usage:
    python scripts/compute_partition_question_stats.py \
        --datasets hotpotqa 2wikimultihop musique --num-clients 3 \
        --settings random dirichlet_1.0 dirichlet_0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LINEARRAG_ROOT = Path(__file__).resolve().parent.parent / "dataset" / "linearrag"
PROCESSED_ROOT = Path(__file__).resolve().parent.parent / "processed"

SETTINGS = {
    "random": ("random", None),
    "dirichlet_1.0": ("dirichlet", 1.0),
    "dirichlet_0.1": ("dirichlet", 0.1),
}


def load_questions(dataset: str) -> list[dict]:
    return json.loads((LINEARRAG_ROOT / dataset / "questions.json").read_text())


def load_diagnostics(dataset: str, mode: str, alpha: float | None) -> dict:
    path = PROCESSED_ROOT / dataset / f"partition_diagnostics_{mode}_{alpha}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/preprocess_data.py --dataset {dataset} "
            f"--partition-mode {mode}" + (f" --alpha {alpha}" if alpha is not None else "")
        )
    return json.loads(path.read_text())


def _split_name(i: int, n: int) -> str:
    """Positional 80/10/10 split, matching scripts/build_fedcond_qa_dataset.py
    exactly (train=[0, 0.8n), val=[0.8n, 0.9n), test=[0.9n, n)) -- no
    shuffling, so this is computable directly from question position without
    running that script."""
    if i < int(0.8 * n):
        return "train"
    if i < int(0.9 * n):
        return "val"
    return "test"


def classify_questions(questions: list[dict], article_to_client: dict, num_clients: int) -> dict:
    """Returns per-question records plus aggregate counts.

    Availability categories, made explicit and non-overlapping:
      - "found":   every gold_title for this question is a known article
                   (owned by exactly one client each) under this partition.
      - "missing": at least one gold_title is not present as an article in
                   this partition's corpus at all (a benchmark/corpus gap,
                   not a partitioning artifact -- the same titles are
                   missing regardless of which client would own them).
    A question can only be "local_answerable" if its category is "found"
    AND all owning clients equal its assigned client; every "missing"
    question is conservatively counted as cross-client (local-answerability
    cannot be confirmed).
    """
    n = len(questions)
    records = []
    for i, q in enumerate(questions):
        assigned_client = i % num_clients
        gold_titles = q.get("gold_titles", [])
        missing_titles = [t for t in gold_titles if t not in article_to_client]
        owning_clients = sorted({article_to_client[t] for t in gold_titles if t in article_to_client})
        availability = "missing" if missing_titles else "found"
        is_local = availability == "found" and owning_clients == [assigned_client]
        records.append({
            "id": q["id"],
            "split": _split_name(i, n),
            "assigned_client": assigned_client,
            "gold_titles": gold_titles,
            "owning_clients": owning_clients,
            "missing_gold_titles": missing_titles,
            "availability": availability,
            "local_answerable": bool(is_local),
        })

    def _agg(recs: list[dict]) -> dict:
        n_recs = len(recs)
        n_local = sum(1 for r in recs if r["local_answerable"])
        n_questions_missing = sum(1 for r in recs if r["availability"] == "missing")
        distinct_missing_titles = {t for r in recs for t in r["missing_gold_titles"]}
        return {
            "num_questions": n_recs,
            "num_local_answerable": n_local,
            "num_cross_client": n_recs - n_local,
            "num_questions_with_missing_gold_title": n_questions_missing,
            "num_distinct_missing_gold_titles": len(distinct_missing_titles),
        }

    result = _agg(records)
    result["by_split"] = {s: _agg([r for r in records if r["split"] == s]) for s in ("train", "val", "test")}
    result["records"] = records
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimultihop", "musique"])
    p.add_argument("--settings", nargs="+", default=list(SETTINGS.keys()), choices=list(SETTINGS.keys()))
    p.add_argument("--num-clients", type=int, default=3)
    p.add_argument("--out-dir", default="experiments/topic_skew")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    partition_rows = []
    question_rows = []
    full_results = {}

    for ds in args.datasets:
        questions = load_questions(ds)
        for setting in args.settings:
            mode, alpha = SETTINGS[setting]
            diag = load_diagnostics(ds, mode, alpha)
            qstats = classify_questions(questions, diag["article_to_client"], args.num_clients)
            full_results[f"{ds}/{setting}"] = {"partition": diag, "questions": qstats}

            partition_rows.append({
                "dataset": ds, "setting": setting,
                "num_articles": diag["num_articles"],
                "num_passages": diag["num_passages_after_dedup"],
                "client_sizes_articles": diag["client_sizes_articles"],
                "client_sizes_passages": diag["client_sizes_passages"],
                "mean_pairwise_js_divergence": diag.get("mean_pairwise_js_divergence"),
            })
            question_rows.append({
                "dataset": ds, "setting": setting,
                "num_questions": qstats["num_questions"],
                "num_local_answerable": qstats["num_local_answerable"],
                "num_cross_client": qstats["num_cross_client"],
                "pct_local": round(100 * qstats["num_local_answerable"] / qstats["num_questions"], 1),
                "num_questions_with_missing_gold_title": qstats["num_questions_with_missing_gold_title"],
                "num_distinct_missing_gold_titles": qstats["num_distinct_missing_gold_titles"],
                "by_split": qstats["by_split"],
            })
            print(f"{ds:16s} {setting:14s} local={qstats['num_local_answerable']:4d} "
                  f"cross={qstats['num_cross_client']:4d} "
                  f"questions_missing_gold={qstats['num_questions_with_missing_gold_title']:3d} "
                  f"distinct_missing_titles={qstats['num_distinct_missing_gold_titles']:3d} "
                  f"JS={diag.get('mean_pairwise_js_divergence')}")

    (out_dir / "partition_statistics.json").write_text(json.dumps(partition_rows, indent=2))
    (out_dir / "question_ownership.json").write_text(json.dumps(question_rows, indent=2))
    # Full per-question records, kept out of the summary tables but saved for audit.
    for key, val in full_results.items():
        safe = key.replace("/", "__")
        (out_dir / f"detail_{safe}.json").write_text(json.dumps(val, indent=2, default=str))

    print(f"\nWrote {out_dir}/partition_statistics.json, {out_dir}/question_ownership.json, "
          f"and per-combo detail_*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
