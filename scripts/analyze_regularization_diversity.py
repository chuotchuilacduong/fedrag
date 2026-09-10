"""Post-hoc analysis for the "Effect of Regularization on Node Diversity"
ablation: builds the paper-ready CSV/JSON/Markdown outputs, the shared-PCA
figure set, and (optionally) uploads a W&B artifact.

Usage:
    python scripts/analyze_regularization_diversity.py \
        --experiment-dir experiments/regularization_diversity/hotpotqa_seed1 \
        --with-reg-run-id <wandb_run_id> --without-reg-run-id <wandb_run_id> \
        --with-reg-run-url <url> --without-reg-run-url <url> \
        [--with-reg-qa-metrics path.jsonl --without-reg-qa-metrics path.jsonl]

Reads <experiment-dir>/{with_reg,without_reg}/diversity_checkpoints.jsonl and
checkpoints/ckpt_XXXX.pt (written by FedTrainer when --experiment-dir is
passed to `main.py fl-train`), and writes everything under
<experiment-dir>/analysis/ and <experiment-dir>/figures/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from fedcond_grag.analysis.pca_viz import fit_shared_pca, render_figures, select_checkpoints

VARIANTS = ("with_reg", "without_reg")


def _load_checkpoint_index(variant_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((variant_dir / "checkpoints").glob("ckpt_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows.append({
            "checkpoint_index": payload["checkpoint_index"],
            "phase": payload["phase"],
            "round": payload["round"],
            "server_update": payload["server_update"],
            "path": path,
        })
    return rows


def _load_diversity_csv(experiment_dir: Path) -> pd.DataFrame:
    frames = []
    for variant in VARIANTS:
        p = experiment_dir / variant / "diversity_checkpoints.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p} -- run main.py fl-train with --experiment-dir first")
        frames.append(pd.read_json(p, lines=True))
    return pd.concat(frames, ignore_index=True)


def _load_qa_results(experiment_dir: Path, overrides: dict[str, str | None],
                      pred_overrides: dict[str, str | None] | None = None) -> pd.DataFrame:
    pred_overrides = pred_overrides or {}
    rows = []
    for variant in VARIANTS:
        override_path = overrides.get(variant)
        if override_path:
            # Held-out eval-only run's --metrics-path (one JSONL line, round 0).
            with open(override_path) as f:
                lines = [json.loads(l) for l in f if l.strip()]
            m = lines[-1]
            n_eval = None
            pred_path = pred_overrides.get(variant)
            if pred_path and Path(pred_path).exists():
                with open(pred_path) as f:
                    n_eval = sum(1 for l in f if l.strip())
            rows.append({
                "variant": variant, "source": "held_out_eval_only",
                "f1": m.get("test_f1"), "em": m.get("test_em"), "hit": m.get("test_acc"),
                "num_eval_questions": n_eval,
            })
        else:
            qa_path = experiment_dir / variant / "qa_summary.json"
            if not qa_path.exists():
                continue
            m = json.loads(qa_path.read_text())
            rows.append({
                "variant": variant, "source": "training_run_eval",
                "f1": m.get("f1"), "em": m.get("em"), "hit": m.get("hit"),
                "num_eval_questions": m.get("num_eval_questions"),
            })
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", required=True)
    p.add_argument("--dataset", default="hotpotqa")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--with-reg-qa-metrics", default=None,
                   help="--metrics-path from the held-out eval-only run for the with_reg arm")
    p.add_argument("--without-reg-qa-metrics", default=None,
                   help="--metrics-path from the held-out eval-only run for the without_reg arm")
    p.add_argument("--with-reg-qa-predictions", default=None,
                   help="--dump-predictions path from the with_reg held-out eval (for num_eval_questions)")
    p.add_argument("--without-reg-qa-predictions", default=None,
                   help="--dump-predictions path from the without_reg held-out eval (for num_eval_questions)")
    p.add_argument("--with-reg-run-url", default=None)
    p.add_argument("--without-reg-run-url", default=None)
    p.add_argument("--with-reg-run-id", default=None)
    p.add_argument("--without-reg-run-id", default=None)
    p.add_argument("--wandb-project", default="fedcond-graphrag")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--upload-artifact", action="store_true")
    args = p.parse_args()

    exp_dir = Path(args.experiment_dir)
    analysis_dir = exp_dir / "analysis"
    figures_dir = exp_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1) diversity_by_checkpoint.csv
    div_df = _load_diversity_csv(exp_dir)
    div_df.to_csv(analysis_dir / "diversity_by_checkpoint.csv", index=False)

    # 2) qa_results.csv
    overrides = {"with_reg": args.with_reg_qa_metrics, "without_reg": args.without_reg_qa_metrics}
    pred_overrides = {"with_reg": args.with_reg_qa_predictions, "without_reg": args.without_reg_qa_predictions}
    qa_df = _load_qa_results(exp_dir, overrides, pred_overrides)
    qa_df.to_csv(analysis_dir / "qa_results.csv", index=False)

    # 3) paired-init verification
    init_rows = div_df[(div_df["phase"] == "phase1") & (div_df["server_update"] == 0)]
    ckpt_idx = {v: _load_checkpoint_index(exp_dir / v) for v in VARIANTS}
    init_ckpt = {v: [c for c in ckpt_idx[v] if c["phase"] == "phase1" and c["server_update"] == 0][0] for v in VARIANTS}
    x_init_with = torch.load(init_ckpt["with_reg"]["path"], map_location="cpu", weights_only=False)["x"]
    x_init_without = torch.load(init_ckpt["without_reg"]["path"], map_location="cpu", weights_only=False)["x"]
    max_abs_diff_init = float((x_init_with - x_init_without).abs().max().item())

    # 4) shared PCA across the 4x2 canonical checkpoints
    selected = {v: select_checkpoints(ckpt_idx[v]) for v in VARIANTS}
    matrices = {}
    for v in VARIANTS:
        for ckpt_name, meta in selected[v].items():
            matrices[(v, ckpt_name)] = torch.load(meta["path"], map_location="cpu", weights_only=False)["x"]
    pca = fit_shared_pca(matrices)
    figure_paths = render_figures(pca, figures_dir)

    # 5) summary.json
    def _final_metrics(variant: str, rep: str) -> dict:
        end_ckpt = selected[variant]["end_phase2"]
        row = div_df[
            (div_df["variant"] == variant) & (div_df["checkpoint_index"] == end_ckpt["checkpoint_index"])
            & (div_df["representation"] == rep)
        ].iloc[0]
        return row

    qa_by_variant = {r["variant"]: r for _, r in qa_df.iterrows()} if len(qa_df) else {}

    def _variant_summary(variant: str) -> dict:
        x_final = _final_metrics(variant, "X_syn")
        h_final = _final_metrics(variant, "H_syn")
        qa = qa_by_variant.get(variant, {})
        run_id = args.with_reg_run_id if variant == "with_reg" else args.without_reg_run_id
        run_url = args.with_reg_run_url if variant == "with_reg" else args.without_reg_run_url
        return {
            "final_X_mean_cos": float(x_final["mean_offdiag_cosine"]),
            "final_X_effective_rank": float(x_final["effective_rank"]),
            "final_X_zero_rows": int(x_final["num_near_zero_rows"]),
            "final_H_mean_cos": float(h_final["mean_offdiag_cosine"]),
            "final_H_effective_rank": float(h_final["effective_rank"]),
            "final_H_zero_rows": int(h_final["num_near_zero_rows"]),
            "qa_f1": qa.get("f1"),
            "qa_em": qa.get("em"),
            "qa_hit": qa.get("hit"),
            "qa_num_eval_questions": qa.get("num_eval_questions"),
            "wandb_run_id": run_id,
            "wandb_run_url": run_url,
        }

    end_phase1_update = selected["with_reg"]["end_phase1"]["server_update"]
    early_phase1_update = selected["with_reg"]["early_phase1"]["server_update"]
    end_phase2_round = selected["with_reg"]["end_phase2"]["round"]

    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "phase1_end_update": end_phase1_update,
        "early_phase1_update": early_phase1_update,
        "phase2_end_round": end_phase2_round,
        "shared_pca_explained_variance_ratio": pca["explained_variance_ratio"],
        "max_abs_diff_X_syn_init_reg_vs_noreg": max_abs_diff_init,
        "with_reg": _variant_summary("with_reg"),
        "without_reg": _variant_summary("without_reg"),
    }
    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # 6) run_manifest.json
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent).decode().strip()
    except Exception:
        git_hash = None
    manifest = {
        "git_commit": git_hash,
        "dataset": args.dataset,
        "seed": args.seed,
        "wandb_project": args.wandb_project,
        "wandb_group": f"regularization-diversity-{args.dataset}-seed{args.seed}",
        "with_reg": {"wandb_run_id": args.with_reg_run_id, "wandb_run_url": args.with_reg_run_url},
        "without_reg": {"wandb_run_id": args.without_reg_run_id, "wandb_run_url": args.without_reg_run_url},
        "figures": {name: str(Path(path).relative_to(exp_dir)) for name, path in figure_paths.items()},
    }
    (analysis_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    # 7) Markdown report
    md = _render_markdown(summary)
    (analysis_dir / "regularization_ablation_results.md").write_text(md)

    print(f"Wrote analysis outputs to {analysis_dir}")
    print(f"Wrote figures to {figures_dir}")
    print(f"max_abs_diff(X_syn_init_reg, X_syn_init_noreg) = {max_abs_diff_init:.3e}")

    if args.upload_artifact:
        _upload_artifact(exp_dir, args)

    return 0


def _render_markdown(summary: dict) -> str:
    wr, nr = summary["with_reg"], summary["without_reg"]
    return f"""# Effect of Regularization on Node Diversity -- HotpotQA, seed {summary['seed']}

n = 1 seed; standard deviation is not defined/reported.

| | Phase-I end (update) | Early Phase-I (update) | Phase-II end (round) |
|---|---|---|---|
| checkpoints | {summary['phase1_end_update']} | {summary['early_phase1_update']} | {summary['phase2_end_round']} |

Shared PCA explained variance: PC1 {100*summary['shared_pca_explained_variance_ratio'][0]:.1f}%, \
PC2 {100*summary['shared_pca_explained_variance_ratio'][1]:.1f}%

Paired-init check: max_abs_diff(X_syn_init_reg, X_syn_init_noreg) = {summary['max_abs_diff_X_syn_init_reg_vs_noreg']:.3e}

## Final metrics

| metric | with L_reg | without L_reg |
|---|---|---|
| H_syn mean off-diag cosine | {wr['final_H_mean_cos']:.6f} | {nr['final_H_mean_cos']:.6f} |
| H_syn effective rank | {wr['final_H_effective_rank']:.6f} | {nr['final_H_effective_rank']:.6f} |
| H_syn near-zero rows | {wr['final_H_zero_rows']} | {nr['final_H_zero_rows']} |
| X_syn mean off-diag cosine | {wr['final_X_mean_cos']:.6f} | {nr['final_X_mean_cos']:.6f} |
| X_syn effective rank | {wr['final_X_effective_rank']:.6f} | {nr['final_X_effective_rank']:.6f} |
| QA F1 (cross-client) | {wr['qa_f1']} | {nr['qa_f1']} |
| QA EM | {wr['qa_em']} | {nr['qa_em']} |
| QA hit% | {wr['qa_hit']} | {nr['qa_hit']} |
| # eval questions | {wr['qa_num_eval_questions']} | {nr['qa_num_eval_questions']} |

## WandB runs

- with_reg: {wr['wandb_run_url']}
- without_reg: {nr['wandb_run_url']}

## Paste-ready values

```
Dataset: hotpotqa
Seed: {summary['seed']}
Early Phase-I checkpoint/update: {summary['early_phase1_update']}
Phase-I endpoint/update: {summary['phase1_end_update']}
Phase-II endpoint/round: {summary['phase2_end_round']}

Final H_syn mean cosine, with L_reg: {wr['final_H_mean_cos']:.6f}
Final H_syn effective rank, with L_reg: {wr['final_H_effective_rank']:.6f}
QA F1, with L_reg: {wr['qa_f1']}

Final H_syn mean cosine, without L_reg: {nr['final_H_mean_cos']:.6f}
Final H_syn effective rank, without L_reg: {nr['final_H_effective_rank']:.6f}
QA F1, without L_reg: {nr['qa_f1']}
```
"""


def _upload_artifact(exp_dir: Path, args) -> None:
    import wandb

    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        group=f"regularization-diversity-{args.dataset}-seed{args.seed}",
        job_type="regularization-diversity-ablation-analysis",
        name=f"{args.dataset}-seed{args.seed}-analysis",
    )
    artifact = wandb.Artifact(f"regularization-diversity-{args.dataset}-seed{args.seed}", type="analysis")
    for f in (exp_dir / "analysis").glob("*"):
        artifact.add_file(str(f))
    for f in (exp_dir / "figures").glob("*"):
        artifact.add_file(str(f))
    run.log_artifact(artifact)
    summary_path = exp_dir / "analysis" / "summary.json"
    summary = json.loads(summary_path.read_text())
    run.summary["pca/pc1_explained_variance"] = summary["shared_pca_explained_variance_ratio"][0]
    run.summary["pca/pc2_explained_variance"] = summary["shared_pca_explained_variance_ratio"][1]
    run.summary["pca/early_phase1_update"] = summary["early_phase1_update"]
    run.finish()
    print(f"Uploaded artifact + PCA summary to {run.url}")


if __name__ == "__main__":
    raise SystemExit(main())
