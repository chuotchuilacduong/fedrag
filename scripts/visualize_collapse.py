"""Visualize node-representation collapse for the regularization-diversity
ablation, using diagnostics that actually show the effect (unlike a 2D PCA of
X_syn, which only captures a few % of variance on 384-dim features).

Builds, from one checkpoint pair (default: the end-of-Phase-II checkpoint):
  1. Pairwise-cosine-similarity heatmaps of H_syn (with_reg vs without_reg),
     nodes reordered by hierarchical clustering so redundant blocks are visible.
  2. Overlaid histogram of off-diagonal cosine similarities.
  3. Overlaid singular-value spectrum (normalized, sorted) of H_syn -- steep
     decay = low effective rank = collapse; flat = high effective rank.

Usage:
    python scripts/visualize_collapse.py \
        --experiment-dir experiments/regularization_diversity/hotpotqa_seed1_kg256 \
        --checkpoint-index 52 --representation H_syn
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fedcond_grag.analysis.diversity import rows_for_pca
from fedcond_grag.analysis.pca_viz import CHECKPOINT_ORDER, select_checkpoints
from fedcond_grag.server.stage_c_aggregate.repr_align import encode_nodes_with_edge_weight


def _load_gnn_model_registry():
    gnn_file = pathlib.Path(__file__).resolve().parent.parent / "fedcond_grag" / "model" / "gnn.py"
    spec = importlib.util.spec_from_file_location("_gnn_direct", gnn_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_gnn_model


def rebuild_encoder(ckpt: dict, load_gnn_model):
    cfg = ckpt["gnn_config"]
    encoder = load_gnn_model[cfg["gnn_model_name_c"]](
        in_channels=cfg["gnn_in_dim_c"],
        out_channels=cfg["gnn_hidden_dim_c"],
        hidden_channels=cfg["gnn_hidden_dim_c"],
        num_layers=cfg["gnn_num_layers_c"],
        dropout=cfg["gnn_dropout"],
        num_heads=cfg["gnn_num_heads_c"],
    )
    encoder.load_state_dict(ckpt["encoder_state"]["repr_encoder"])
    encoder.eval()

    proj_sd = ckpt["encoder_state"]["repr_projector"]
    weight_keys = sorted(k for k in proj_sd if k.endswith(".weight"))
    in_dim = proj_sd[weight_keys[0]].shape[1]
    mid_dim = proj_sd[weight_keys[0]].shape[0]
    out_dim = proj_sd[weight_keys[-1]].shape[0]
    projector = torch.nn.Sequential(
        torch.nn.Linear(in_dim, mid_dim), torch.nn.GELU(), torch.nn.Linear(mid_dim, out_dim),
    )
    projector.load_state_dict(proj_sd)
    projector.eval()
    return encoder, projector


def compute_h_syn(ckpt: dict, load_gnn_model) -> torch.Tensor:
    encoder, projector = rebuild_encoder(ckpt, load_gnn_model)
    with torch.no_grad():
        h = encode_nodes_with_edge_weight(
            ckpt["x"], ckpt["edge_index"], ckpt["edge_weight"], encoder, projector,
        )
    return h.detach()


def cluster_order(gram: np.ndarray) -> np.ndarray:
    """Order nodes by average-linkage hierarchical clustering on (1 - cosine)
    distance, purely for visualization -- makes redundant blocks contiguous."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    dist = np.clip(1.0 - gram, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    return np.array(leaves_list(Z))


def _checkpoint_index_meta(ckpt_dir: pathlib.Path) -> list[dict]:
    rows = []
    for path in sorted(ckpt_dir.glob("ckpt_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows.append({
            "checkpoint_index": payload["checkpoint_index"],
            "phase": payload["phase"],
            "round": payload["round"],
            "server_update": payload["server_update"],
            "path": path,
        })
    return rows


def _fit_tsne_domain(reps: dict, keys: list[tuple[str, str]]) -> tuple[dict, tuple, tuple]:
    """Fit ONE t-SNE jointly over exactly the given (variant, checkpoint)
    groups -- callers must only pass groups that share the same encoder
    (Phase-I's frozen-random encoder XOR Phase-II's trained encoder), or the
    embedding will spend its layout budget separating encoder domains instead
    of showing the with/without-reg difference within one domain."""
    from sklearn.manifold import TSNE

    sizes = [reps[k].shape[0] for k in keys]
    joint = np.concatenate([reps[k] for k in keys], axis=0)
    perplexity = min(30, max(5, sizes[0] // 4))
    emb_all = TSNE(
        n_components=2, metric="cosine", perplexity=perplexity, init="pca",
        random_state=0, max_iter=2000,
    ).fit_transform(joint)

    embs, offset = {}, 0
    for k, n in zip(keys, sizes):
        embs[k] = emb_all[offset:offset + n]
        offset += n

    pad = 0.05 * (emb_all.max(axis=0) - emb_all.min(axis=0) + 1e-9)
    xlim = (emb_all[:, 0].min() - pad[0], emb_all[:, 0].max() + pad[0])
    ylim = (emb_all[:, 1].min() - pad[1], emb_all[:, 1].max() + pad[1])
    return embs, xlim, ylim


def _save_clean_panel(emb: np.ndarray, xlim: tuple, ylim: tuple, color: str, path: pathlib.Path) -> None:
    """No title, no ticks, no axis labels -- one grid cell for a paper figure
    whose row/column headers (With/Without L_reg x Init/Early/End Phase I/
    End Phase II) are added externally, matching the target layout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.scatter(emb[:, 0], emb[:, 1], s=22, alpha=0.65, color=color, linewidths=0)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=200)
    # Vector twin (same basename, .pdf) for direct LaTeX/Overleaf inclusion.
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote {path} (+ .pdf)")


def plot_tsne_evolution(exp_dir: pathlib.Path, out_dir: pathlib.Path, load_gnn_model) -> None:
    """Two SEPARATE figures (one per arm), each a 1x4 grid over
    init / early Phase-I / end Phase-I / end Phase-II.

    Phase-I is a FROZEN RANDOM encoder (identical for both arms, since it
    never trains during Phase-I) -- fit ONE shared t-SNE + axis scale across
    both arms' init/early/end_phase1 panels (6 groups) so they're a fair,
    domain-matched comparison. End_phase2 uses the swapped-in TRAINED encoder
    (also identical starting point for both arms via FedAvg) -- a
    categorically different representation domain, so it gets its OWN t-SNE
    fit + axis scale (2 groups) rather than being forced into the Phase-I
    embedding, which would just show "different encoder" instead of
    "with/without regularization"."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ckpt_meta = {v: _checkpoint_index_meta(exp_dir / v / "checkpoints") for v in ("with_reg", "without_reg")}
    selected = {v: select_checkpoints(ckpt_meta[v]) for v in ckpt_meta}

    reps: dict[tuple[str, str], np.ndarray] = {}
    labels: dict[tuple[str, str], str] = {}
    for variant in ("with_reg", "without_reg"):
        for ckpt_name in CHECKPOINT_ORDER:
            meta = selected[variant][ckpt_name]
            ckpt = torch.load(meta["path"], map_location="cpu", weights_only=False)
            h = compute_h_syn(ckpt, load_gnn_model)
            normalized, _ = rows_for_pca(h)
            reps[(variant, ckpt_name)] = normalized.numpy()
            su = meta["server_update"]
            labels[(variant, ckpt_name)] = (
                f"update {su}" if meta["phase"] == "phase1" else f"round {meta['round']}"
            )

    phase1_names = [c for c in CHECKPOINT_ORDER if c != "end_phase2"]
    phase1_keys = [(v, c) for v in ("with_reg", "without_reg") for c in phase1_names]
    phase2_keys = [(v, "end_phase2") for v in ("with_reg", "without_reg")]

    embs_p1, xlim_p1, ylim_p1 = _fit_tsne_domain(reps, phase1_keys)
    embs_p2, xlim_p2, ylim_p2 = _fit_tsne_domain(reps, phase2_keys)
    embs = {**embs_p1, **embs_p2}

    colors = {"with_reg": "#4C72B0", "without_reg": "#C44E52"}
    titles = {"with_reg": r"with $\mathcal{L}_{\mathrm{reg}}$", "without_reg": r"without $\mathcal{L}_{\mathrm{reg}}$"}
    for variant in ("with_reg", "without_reg"):
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
        for ax, ckpt_name in zip(axes, CHECKPOINT_ORDER):
            e = embs[(variant, ckpt_name)]
            ax.scatter(e[:, 0], e[:, 1], s=20, alpha=0.65, color=colors[variant])
            xlim, ylim = (xlim_p2, ylim_p2) if ckpt_name == "end_phase2" else (xlim_p1, ylim_p1)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            domain_note = "trained encoder -- own scale" if ckpt_name == "end_phase2" else "random encoder"
            ax.set_title(f"{ckpt_name} ({labels[(variant, ckpt_name)]})\n{domain_note}", fontsize=17)
            ax.set_xticks([]); ax.set_yticks([])
            if ckpt_name == "end_phase2":
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
        fig.suptitle(
            f"t-SNE evolution of H_syn -- {titles[variant]} "
            "(panels 1-3 share one embedding/scale; panel 4 uses its own -- different encoder)",
            fontsize=16,
        )
        fig.tight_layout()
        path = out_dir / f"collapse_tsne_evolution_{variant}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Wrote {path}")

        # Same 4 panels, one file each (identical embeddings/axes as above).
        for ckpt_name in CHECKPOINT_ORDER:
            e = embs[(variant, ckpt_name)]
            xlim, ylim = (xlim_p2, ylim_p2) if ckpt_name == "end_phase2" else (xlim_p1, ylim_p1)
            fig_single, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.scatter(e[:, 0], e[:, 1], s=24, alpha=0.65, color=colors[variant])
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            domain_note = "trained encoder -- own scale" if ckpt_name == "end_phase2" else "random encoder"
            ax.set_title(f"{titles[variant]} -- {ckpt_name} ({labels[(variant, ckpt_name)]})\n{domain_note}", fontsize=14)
            ax.set_xticks([]); ax.set_yticks([])
            fig_single.tight_layout()
            single_path = out_dir / f"collapse_tsne_evolution_{variant}_{ckpt_name}.png"
            fig_single.savefig(single_path, dpi=150)
            plt.close(fig_single)
            print(f"Wrote {single_path}")

            # Clean (no title/ticks) version for direct insertion into a paper
            # figure grid whose row/column labels are set externally (LaTeX).
            _save_clean_panel(e, xlim, ylim, colors[variant],
                               out_dir / f"grid_{variant}_{ckpt_name}_clean.png")


def plot_tsne_fine_grained(
    exp_dir: pathlib.Path, out_dir: pathlib.Path, load_gnn_model,
    updates: list[int] = (0, 10, 20, 50),
) -> None:
    """Two SEPARATE figures (one per arm), zooming into early Phase-I at the
    given server-update checkpoints (all share the same frozen-random
    encoder, so one shared t-SNE + axis scale across all of them is fair)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ckpt_meta = {v: _checkpoint_index_meta(exp_dir / v / "checkpoints") for v in ("with_reg", "without_reg")}

    reps: dict[tuple[str, int], np.ndarray] = {}
    for variant in ("with_reg", "without_reg"):
        by_update = {m["server_update"]: m for m in ckpt_meta[variant] if m["phase"] == "phase1"}
        missing = [u for u in updates if u not in by_update]
        if missing:
            raise ValueError(f"{variant}: no saved Phase-I checkpoint at update(s) {missing} "
                              f"(available: {sorted(by_update)})")
        for u in updates:
            ckpt = torch.load(by_update[u]["path"], map_location="cpu", weights_only=False)
            h = compute_h_syn(ckpt, load_gnn_model)
            normalized, _ = rows_for_pca(h)
            reps[(variant, u)] = normalized.numpy()

    keys = [(v, u) for v in ("with_reg", "without_reg") for u in updates]
    embs, xlim, ylim = _fit_tsne_domain(reps, keys)

    colors = {"with_reg": "#4C72B0", "without_reg": "#C44E52"}
    titles = {"with_reg": r"with $\mathcal{L}_{\mathrm{reg}}$", "without_reg": r"without $\mathcal{L}_{\mathrm{reg}}$"}
    for variant in ("with_reg", "without_reg"):
        fig, axes = plt.subplots(1, len(updates), figsize=(4.5 * len(updates), 4.6))
        for ax, u in zip(axes, updates):
            e = embs[(variant, u)]
            ax.scatter(e[:, 0], e[:, 1], s=20, alpha=0.65, color=colors[variant])
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_title(f"update {u}", fontsize=15)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"t-SNE, early Phase-I -- {titles[variant]} (shared embedding/scale)", fontsize=17)
        fig.tight_layout()
        path = out_dir / f"collapse_tsne_early_{variant}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Wrote {path}")


def plot_tsne_by_round(
    exp_dir: pathlib.Path, out_dir: pathlib.Path, load_gnn_model,
    rounds: list[int] = (0, 1, 2, 3, 4, 5),
) -> None:
    """Two SEPARATE figures (one per arm), one panel per Phase-II round
    (round 0 = right after the Phase-I-end encoder swap, before any Phase-II
    synthetic-memory adaptation; rounds >=1 = after that round's client
    adaptation + server aggregation/regularization).

    Fits ONE INDEPENDENT t-SNE per round (with_reg + without_reg jointly,
    just those two groups), rather than one giant t-SNE across all rounds:
    Phase-II barely moves X_syn/H_syn round-to-round here (K_mem=2 is a
    deliberately cheap schedule), so a single joint fit across 6 nearly-
    identical copies of each arm dilutes every point's true local neighbors
    with its near-duplicate "twins" from other rounds and washes out the
    within-round cluster structure entirely. Each round therefore gets its
    own fair axis scale -- this view answers "does the collapse persist at
    every round", not "how far did the embedding move", which the Phase-I
    fine-grained view already covers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ckpt_meta = {v: _checkpoint_index_meta(exp_dir / v / "checkpoints") for v in ("with_reg", "without_reg")}

    reps: dict[tuple[str, int], np.ndarray] = {}
    for variant in ("with_reg", "without_reg"):
        by_round = {m["round"]: m for m in ckpt_meta[variant] if m["phase"] == "phase2"}
        missing = [r for r in rounds if r not in by_round]
        if missing:
            raise ValueError(f"{variant}: no saved Phase-II checkpoint at round(s) {missing} "
                              f"(available: {sorted(by_round)}) -- rerun training with enough "
                              f"--num-rounds to cover them.")
        for r in rounds:
            ckpt = torch.load(by_round[r]["path"], map_location="cpu", weights_only=False)
            h = compute_h_syn(ckpt, load_gnn_model)
            normalized, _ = rows_for_pca(h)
            reps[(variant, r)] = normalized.numpy()

    embs: dict[tuple[str, int], np.ndarray] = {}
    xlims: dict[int, tuple] = {}
    ylims: dict[int, tuple] = {}
    for r in rounds:
        round_embs, xlim, ylim = _fit_tsne_domain(reps, [("with_reg", r), ("without_reg", r)])
        embs.update(round_embs)
        xlims[r], ylims[r] = xlim, ylim

    colors = {"with_reg": "#4C72B0", "without_reg": "#C44E52"}
    titles = {"with_reg": r"with $\mathcal{L}_{\mathrm{reg}}$", "without_reg": r"without $\mathcal{L}_{\mathrm{reg}}$"}
    for variant in ("with_reg", "without_reg"):
        fig, axes = plt.subplots(1, len(rounds), figsize=(4.0 * len(rounds), 4.4), squeeze=False)
        axes = axes[0]
        for ax, r in zip(axes, rounds):
            e = embs[(variant, r)]
            ax.scatter(e[:, 0], e[:, 1], s=20, alpha=0.65, color=colors[variant])
            ax.set_xlim(*xlims[r]); ax.set_ylim(*ylims[r])
            ax.set_title(f"round {r}", fontsize=15)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(
            f"t-SNE, Phase-II by round -- {titles[variant]} "
            "(each round fit independently vs. the other arm -- not comparable panel-to-panel)",
            fontsize=16,
        )
        fig.tight_layout()
        path = out_dir / f"collapse_tsne_rounds_{variant}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Wrote {path}")

        # One clean file per round so a specific round can be picked for a
        # paper figure's "End Phase II" cell (e.g. grid_with_reg_round2_clean.png).
        for r in rounds:
            _save_clean_panel(embs[(variant, r)], xlims[r], ylims[r], colors[variant],
                               out_dir / f"grid_{variant}_round{r}_clean.png")


def plot_tsne_xsyn_grid(exp_dir: pathlib.Path, out_dir: pathlib.Path, phase2_round: int = 2) -> None:
    """Same 3-checkpoint (Initialization / Phase I / Phase II) t-SNE view as
    plot_tsne_evolution / plot_tsne_by_round, but on raw X_syn instead of
    H_syn -- no GNN encoder pass needed, since X_syn is one continuous
    optimization variable across the Phase-I/Phase-II boundary (unlike
    H_syn, whose encoder is swapped at that boundary), so ONE joint t-SNE
    fit across all 6 groups is valid here (matching how the required shared-
    PCA figure treats X_syn: one fit reused everywhere, no domain split)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ckpt_meta = {v: _checkpoint_index_meta(exp_dir / v / "checkpoints") for v in ("with_reg", "without_reg")}

    reps: dict[tuple[str, str], np.ndarray] = {}
    for variant in ("with_reg", "without_reg"):
        phase1 = [m for m in ckpt_meta[variant] if m["phase"] == "phase1"]
        init_meta = min(phase1, key=lambda m: m["server_update"])
        end1_meta = max(phase1, key=lambda m: m["server_update"])
        phase2_meta = next(m for m in ckpt_meta[variant] if m["phase"] == "phase2" and m["round"] == phase2_round)
        for name, meta in (("init", init_meta), ("end_phase1", end1_meta), ("phase2", phase2_meta)):
            ckpt = torch.load(meta["path"], map_location="cpu", weights_only=False)
            normalized, _ = rows_for_pca(ckpt["x"])
            reps[(variant, name)] = normalized.numpy()

    col_order = ["init", "end_phase1", "phase2"]
    keys = [(v, c) for v in ("with_reg", "without_reg") for c in col_order]
    embs, xlim, ylim = _fit_tsne_domain(reps, keys)

    colors = {"with_reg": "#4C72B0", "without_reg": "#C44E52"}
    for variant in ("with_reg", "without_reg"):
        for ckpt_name in col_order:
            _save_clean_panel(embs[(variant, ckpt_name)], xlim, ylim, colors[variant],
                               out_dir / f"grid_{variant}_{ckpt_name}_xsyn_clean.png")

    # Combined reference figure with labels, same 2x3 layout as the H_syn one.
    titles = {"with_reg": r"With $\mathcal{L}_{\mathrm{reg}}$", "without_reg": r"Without $\mathcal{L}_{\mathrm{reg}}$"}
    col_labels = {"init": "Initialization", "end_phase1": "Phase I", "phase2": f"Phase II (round {phase2_round})"}
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.6))
    for row, variant in enumerate(("with_reg", "without_reg")):
        for col, ckpt_name in enumerate(col_order):
            e = embs[(variant, ckpt_name)]
            ax = axes[row][col]
            ax.scatter(e[:, 0], e[:, 1], s=20, alpha=0.65, color=colors[variant], linewidths=0)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(col_labels[ckpt_name], fontsize=17)
            if col == 0:
                ax.set_ylabel(titles[variant], fontsize=17)
    fig.suptitle(r"t-SNE of $X_{\mathrm{syn}}$ (one shared joint fit, cosine metric)", fontsize=14)
    fig.tight_layout()
    path = out_dir / "preview_grid_xsyn.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def assemble_preview_grid(out_dir: pathlib.Path, end_phase2_round: int) -> pathlib.Path:
    """Reads already-saved clean grid_*.png panels (no t-SNE recompute) and
    assembles them into the paper's target 2x4 layout (rows: With/Without
    L_reg; columns: Initialization/Early Phase I/End Phase I/End Phase II)
    with row/column labels, for previewing which End-Phase-II round to keep."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    col_names = [("init", "Initialization"), ("early_phase1", "Early Phase I"), ("end_phase1", "End Phase I")]
    row_names = [("with_reg", "With L_reg"), ("without_reg", "Without L_reg")]

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6))
    for row, (variant, row_label) in enumerate(row_names):
        for col, (ckpt_name, col_label) in enumerate(col_names):
            img = mpimg.imread(out_dir / f"grid_{variant}_{ckpt_name}_clean.png")
            ax = axes[row][col]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(col_label, fontsize=16)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=16)
        # 4th column: chosen Phase-II round
        img = mpimg.imread(out_dir / f"grid_{variant}_round{end_phase2_round}_clean.png")
        ax = axes[row][3]
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(f"End Phase II (round {end_phase2_round})", fontsize=16)

    fig.tight_layout()
    path = out_dir / f"preview_grid_round{end_phase2_round}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", required=True)
    p.add_argument("--checkpoint-index", type=int, default=None,
                   help="Checkpoint index to use for both arms (default: the last one found)")
    p.add_argument("--out-dir", default=None, help="Default: <experiment-dir>/figures")
    p.add_argument("--skip-evolution", action="store_true",
                   help="Skip the 4-checkpoint t-SNE evolution figures (they recompute H_syn "
                        "for 8 checkpoints total and refit t-SNE, so are the slowest part)")
    p.add_argument("--rounds", type=int, nargs="+", default=None,
                   help="If given, also plot Phase-II by round (e.g. --rounds 0 1 2 3 4 5) -- "
                        "requires checkpoints saved for every requested round.")
    p.add_argument("--xsyn-round", type=int, default=None,
                   help="If given, also plot the 3-checkpoint (Init/Phase I/Phase II) t-SNE "
                        "grid on raw X_syn (one shared joint fit) using this Phase-II round.")
    args = p.parse_args()

    exp_dir = pathlib.Path(args.experiment_dir)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else exp_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    load_gnn_model = _load_gnn_model_registry()

    reps = {}
    for variant in ("with_reg", "without_reg"):
        ckpt_dir = exp_dir / variant / "checkpoints"
        if args.checkpoint_index is not None:
            ckpt_path = ckpt_dir / f"ckpt_{args.checkpoint_index:04d}.pt"
        else:
            ckpt_path = sorted(ckpt_dir.glob("ckpt_*.pt"))[-1]
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        h = compute_h_syn(ckpt, load_gnn_model)
        normalized, n_near_zero = rows_for_pca(h)
        reps[variant] = normalized.numpy()
        print(f"{variant}: loaded {ckpt_path.name} (phase={ckpt['phase']}, round={ckpt['round']}, "
              f"update={ckpt['server_update']}), H_syn shape={h.shape}, near-zero rows={n_near_zero}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grams = {v: reps[v] @ reps[v].T for v in reps}
    k = grams["with_reg"].shape[0]
    offdiag_mask = ~np.eye(k, dtype=bool)

    # --- 1. Clustered cosine-similarity heatmaps ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, variant, title in zip(axes, ("with_reg", "without_reg"), (r"with $\mathcal{L}_{\mathrm{reg}}$", r"without $\mathcal{L}_{\mathrm{reg}}$")):
        order = cluster_order(grams[variant])
        im = ax.imshow(grams[variant][order][:, order], vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(f"H_syn pairwise cosine -- {title}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label="cosine similarity")
    fig.savefig(out_dir / "collapse_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 2. Overlaid histogram of off-diagonal cosine similarities ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"with_reg": "#4C72B0", "without_reg": "#C44E52"}
    for variant, label in (("with_reg", r"with $\mathcal{L}_{\mathrm{reg}}$"), ("without_reg", r"without $\mathcal{L}_{\mathrm{reg}}$")):
        vals = grams[variant][offdiag_mask]
        ax.hist(vals, bins=60, range=(-1, 1), alpha=0.55, density=True,
                color=colors[variant], label=f"{label} (mean={vals.mean():.3f})")
    ax.set_xlabel("pairwise cosine similarity (H_syn, off-diagonal)")
    ax.set_ylabel("density")
    ax.set_title("Node-representation redundancy")
    ax.legend(fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "collapse_cosine_histogram.png", dpi=150)
    plt.close(fig)

    # --- 3. Singular-value spectrum (normalized, sorted) ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for variant, label in (("with_reg", r"with $\mathcal{L}_{\mathrm{reg}}$"), ("without_reg", r"without $\mathcal{L}_{\mathrm{reg}}$")):
        s = np.linalg.svd(reps[variant], compute_uv=False)
        s = s / s.sum()
        r_eff = float(np.exp(-(s * np.log(np.clip(s, 1e-12, None))).sum()))
        ax.plot(np.arange(1, len(s) + 1), s, color=colors[variant],
                label=f"{label} (r_eff={r_eff:.1f})", linewidth=2)
    ax.set_xlabel("singular value rank")
    ax.set_ylabel("normalized singular value")
    ax.set_yscale("log")
    ax.set_title("H_syn spectral decay -- steep = collapsed, flat = diverse")
    ax.legend(fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "collapse_spectrum.png", dpi=150)
    plt.close(fig)

    # --- 4. Joint t-SNE scatter of H_syn -- nonlinear, so a collapsed cluster
    # (many near-duplicate nodes) shows up as a tight blob even though linear
    # PCA on the same data captures too little variance to reveal it.
    from sklearn.manifold import TSNE

    joint = np.concatenate([reps["with_reg"], reps["without_reg"]], axis=0)
    n = reps["with_reg"].shape[0]
    perplexity = min(30, max(5, n // 4))
    emb = TSNE(
        n_components=2, metric="cosine", perplexity=perplexity, init="pca",
        random_state=0, max_iter=2000,
    ).fit_transform(joint)
    emb_with, emb_without = emb[:n], emb[n:]

    # Flag nodes sitting in a LOCAL collapsed cluster: mean cosine to their
    # own k-nearest neighbors (not the global mean, which stays low even for
    # a node duplicated 8x among 256 total nodes -- collapse here is several
    # small tight clusters, not one global blob).
    k_nn = 5
    sorted_sims = np.sort(grams["without_reg"] - np.eye(k), axis=1)[:, ::-1]
    knn_mean_sim = sorted_sims[:, :k_nn].mean(axis=1)
    collapsed_mask = knn_mean_sim > 0.9

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(emb_with[:, 0], emb_with[:, 1], s=18, alpha=0.6,
               color=colors["with_reg"], label=r"with $\mathcal{L}_{\mathrm{reg}}$")
    ax.scatter(emb_without[~collapsed_mask, 0], emb_without[~collapsed_mask, 1], s=18, alpha=0.6,
               color=colors["without_reg"], label=r"without $\mathcal{L}_{\mathrm{reg}}$")
    if collapsed_mask.any():
        n_collapsed = int(collapsed_mask.sum())
        collapsed_label = r"without $\mathcal{L}_{\mathrm{reg}}$" + f" -- collapsed cluster (n={n_collapsed})"
        ax.scatter(emb_without[collapsed_mask, 0], emb_without[collapsed_mask, 1], s=28,
                   facecolors="none", edgecolors="black", linewidths=1.0,
                   label=collapsed_label)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE of H_syn (cosine metric, joint embedding)")
    ax.legend(fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "collapse_tsne.png", dpi=150)
    plt.close(fig)

    print(f"Wrote collapse_heatmap.png, collapse_cosine_histogram.png, collapse_spectrum.png, "
          f"collapse_tsne.png -> {out_dir}")

    if not args.skip_evolution:
        plot_tsne_evolution(exp_dir, out_dir, load_gnn_model)
        plot_tsne_fine_grained(exp_dir, out_dir, load_gnn_model, updates=[0, 10, 20, 50])

    if args.rounds is not None:
        plot_tsne_by_round(exp_dir, out_dir, load_gnn_model, rounds=args.rounds)

    if args.xsyn_round is not None:
        plot_tsne_xsyn_grid(exp_dir, out_dir, phase2_round=args.xsyn_round)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
