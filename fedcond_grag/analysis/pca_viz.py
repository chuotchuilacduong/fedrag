"""Shared-PCA visualization for the regularization-diversity ablation.

One PCA (2 components, no whitening) fit jointly across BOTH ablation arms'
X_syn snapshots at 4 matched checkpoints (init, early Phase-I, end Phase-I,
end Phase-II) -- 8 panels total, reusing the same mean/components/axis limits
so panels are directly comparable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fedcond_grag.analysis.diversity import rows_for_pca

CHECKPOINT_ORDER = ("init", "early_phase1", "end_phase1", "end_phase2")
VARIANTS = ("with_reg", "without_reg")


def select_checkpoints(ckpt_meta: list[dict]) -> dict[str, dict]:
    """Pick the 4 canonical checkpoints from one variant's checkpoint index.

    ckpt_meta: list of {"checkpoint_index", "phase", "round", "server_update"}
    for every saved checkpoint (phase1 in ascending server_update, phase2 in
    ascending round), as written alongside each ckpt_XXXX.pt file.

    "early Phase-I" = 10% of the Phase-I schedule, rounded to a recorded
    multiple of --syn-snapshot-every (falls back to the nearest recorded
    update if 10% isn't exactly on the snapshot cadence).
    """
    phase1 = sorted([m for m in ckpt_meta if m["phase"] == "phase1"], key=lambda m: m["server_update"])
    phase2 = sorted([m for m in ckpt_meta if m["phase"] == "phase2"], key=lambda m: m["round"])
    if not phase1:
        raise ValueError("No Phase-I checkpoints found")

    init_ckpt = phase1[0]
    end_phase1_ckpt = phase1[-1]
    num_phase1_steps = end_phase1_ckpt["server_update"]
    target_early = round(0.1 * num_phase1_steps)
    early_ckpt = min(phase1, key=lambda m: abs(m["server_update"] - target_early))
    end_phase2_ckpt = phase2[-1] if phase2 else end_phase1_ckpt

    return {
        "init": init_ckpt,
        "early_phase1": early_ckpt,
        "end_phase1": end_phase1_ckpt,
        "end_phase2": end_phase2_ckpt,
    }


def load_x_syn(ckpt_path: Path) -> torch.Tensor:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return payload["x"]


def fit_shared_pca(matrices: dict[tuple[str, str], torch.Tensor]) -> dict:
    """Fit ONE 2-component PCA (no whitening) on all 8 row-normalized,
    near-zero-row-dropped X_syn matrices concatenated together.

    matrices: {(variant, checkpoint_name): X_syn [N, D]} for all 8 panels.
    Returns {"mean": [D], "components": [2, D], "explained_variance_ratio":
    [2], "projections": {(variant, ckpt): [N', 2]}, "near_zero_counts": {...},
    "xlim": (min, max), "ylim": (min, max)}.
    """
    normalized: dict[tuple[str, str], torch.Tensor] = {}
    near_zero: dict[tuple[str, str], int] = {}
    for key, x in matrices.items():
        norm_rows, n_near_zero = rows_for_pca(x)
        normalized[key] = norm_rows
        near_zero[key] = n_near_zero

    concat = torch.cat(list(normalized.values()), dim=0).numpy().astype(np.float64)
    mean = concat.mean(axis=0)
    centered = concat - mean  # PCA's own internal centering (SVD requires it);
    # the diversity metrics elsewhere never center -- this centering is local
    # to PCA and does not feed back into mean_offdiag_cosine/effective_rank.
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]  # [2, D]
    total_var = (s ** 2).sum()
    explained_variance_ratio = (s[:2] ** 2) / total_var

    projections: dict[tuple[str, str], np.ndarray] = {}
    all_xy = []
    for key, rows in normalized.items():
        proj = (rows.numpy().astype(np.float64) - mean) @ components.T
        projections[key] = proj
        all_xy.append(proj)
    all_xy = np.concatenate(all_xy, axis=0)
    pad_x = 0.05 * (all_xy[:, 0].max() - all_xy[:, 0].min() + 1e-9)
    pad_y = 0.05 * (all_xy[:, 1].max() - all_xy[:, 1].min() + 1e-9)
    xlim = (float(all_xy[:, 0].min() - pad_x), float(all_xy[:, 0].max() + pad_x))
    ylim = (float(all_xy[:, 1].min() - pad_y), float(all_xy[:, 1].max() + pad_y))

    return {
        "mean": mean,
        "components": components,
        "explained_variance_ratio": explained_variance_ratio.tolist(),
        "projections": projections,
        "near_zero_counts": near_zero,
        "xlim": xlim,
        "ylim": ylim,
    }


def plot_panel(ax, proj: np.ndarray, xlim, ylim, pc1_pct: float, pc2_pct: float, color: str) -> None:
    ax.scatter(proj[:, 0], proj[:, 1], s=6, alpha=0.35, linewidths=0, color=color)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(f"PC1 ({pc1_pct:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_pct:.1f}%)")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_alpha(0.3)


def render_figures(pca: dict, out_dir: Path) -> dict[str, str]:
    """Write the 8 individual panel PDFs + one combined 2x4 diagnostic PNG.

    Returns {panel_name: pdf_path} for the 8 individual files.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    pc1_pct = 100.0 * pca["explained_variance_ratio"][0]
    pc2_pct = 100.0 * pca["explained_variance_ratio"][1]
    colors = {"with_reg": "#4C72B0", "without_reg": "#C44E52"}

    file_map = {
        ("with_reg", "init"): "x_reg_init.pdf",
        ("with_reg", "early_phase1"): "x_reg_early.pdf",
        ("with_reg", "end_phase1"): "x_reg_phase1.pdf",
        ("with_reg", "end_phase2"): "x_reg_final.pdf",
        ("without_reg", "init"): "x_noreg_init.pdf",
        ("without_reg", "early_phase1"): "x_noreg_early.pdf",
        ("without_reg", "end_phase1"): "x_noreg_phase1.pdf",
        ("without_reg", "end_phase2"): "x_noreg_final.pdf",
    }

    written: dict[str, str] = {}
    for (variant, ckpt_name), fname in file_map.items():
        fig, ax = plt.subplots(figsize=(2.6, 2.6))
        plot_panel(ax, pca["projections"][(variant, ckpt_name)], pca["xlim"], pca["ylim"],
                   pc1_pct, pc2_pct, colors[variant])
        fig.tight_layout()
        path = out_dir / fname
        fig.savefig(path, format="pdf")
        plt.close(fig)
        written[fname] = str(path)

    # Combined 2 (variant) x 4 (checkpoint) diagnostic figure.
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for row, variant in enumerate(VARIANTS):
        for col, ckpt_name in enumerate(CHECKPOINT_ORDER):
            ax = axes[row][col]
            plot_panel(ax, pca["projections"][(variant, ckpt_name)], pca["xlim"], pca["ylim"],
                       pc1_pct, pc2_pct, colors[variant])
            if row == 0:
                ax.set_title(ckpt_name, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{variant}\nPC2 ({pc2_pct:.1f}%)", fontsize=9)
    fig.suptitle(f"Shared PCA of X_syn -- PC1 {pc1_pct:.1f}%, PC2 {pc2_pct:.1f}%", fontsize=11)
    fig.tight_layout()
    combined_path = out_dir / "pca_combined_diagnostic.png"
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)
    written["pca_combined_diagnostic.png"] = str(combined_path)

    return written
