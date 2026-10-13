"""Standalone schema diagrams for each "trick" slide, rendered with
matplotlib so they're slide-ready PNGs (not meant to be imported/reused
generically like build_metrics_table.py -- each trick's mechanism is drawn
by its own dedicated function since they don't share a layout)."""

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


def _box(ax, xy, w, h, text, facecolor="#eef3fb", edgecolor="#2c5f9e", fontsize=10):
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.5,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )


def _arrow(
    ax, p0, p1, label=None, label_frac=0.5, label_dy=0.12, style="-|>", color="black"
):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=16, color=color, linewidth=1.5
        )
    )
    if label:
        mx = p0[0] + label_frac * (p1[0] - p0[0])
        my = p0[1] + label_frac * (p1[1] - p0[1])
        ax.text(
            mx,
            my + label_dy,
            label,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=color,
        )


def iceberg_aug_schema(output_path: str):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    _box(
        ax,
        (0.3, 2.9),
        2.1,
        0.9,
        "Training\nmolecule",
        facecolor="#fff3cd",
        edgecolor="#b8860b",
    )

    _box(
        ax,
        (3.2, 3.6),
        2.6,
        0.8,
        "Real measured\nspectra (NIST20/MSG)",
        facecolor="#eef3fb",
    )
    _box(
        ax,
        (3.2, 2.1),
        2.6,
        0.8,
        "ICEBERG-predicted\nsynthetic spectra",
        facecolor="#e8f6e8",
        edgecolor="#2e7d32",
    )

    _arrow(ax, (2.4, 3.6), (3.2, 4.0))
    _arrow(ax, (2.4, 3.1), (3.2, 2.5))

    _box(
        ax,
        (6.5, 2.9),
        3.0,
        0.9,
        "Spectrum fed to\nSIMBA for this pair",
        facecolor="#f3e5f5",
        edgecolor="#6a1b9a",
    )
    _arrow(
        ax, (5.8, 4.0), (6.5, 3.55), label="prob 1 - p", label_frac=0.35, label_dy=0.15
    )
    _arrow(
        ax,
        (5.8, 2.5),
        (6.5, 3.05),
        label="prob p = 0.5",
        label_frac=0.35,
        label_dy=-0.3,
    )

    ax.text(
        5.0,
        0.9,
        "Only applied to the TRAIN split. Synthetic spectra come from a separately\n"
        "pretrained ICEBERG fragmentation-prediction model, matched to training\n"
        "molecules by canonical SMILES: 115,314 synthetic spectra matched to\n"
        "23,107 / 23,125 (99.9%) train molecules.",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
        color="#444444",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {output_path}")


def contrastive_schema(output_path: str):
    fig, ax = plt.subplots(figsize=(14.5, 7.2))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # Batch, with the self-pair subset drawn as a nested sub-rectangle --
    # real numbers from contrastive_only_full_run's own training logs
    # (contrastive_n_pairs, filtered to genuine training steps only, mean
    # 413.6 / median 413 across all 24,000 logged training steps, stable
    # from the first step to the last -- not a transient early-training
    # artifact).
    batch_w, batch_h = 4.4, 3.6
    bx, by = 0.3, 1.7
    ax.add_patch(
        FancyBboxPatch(
            (bx, by),
            batch_w,
            batch_h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor="#fff3cd",
            edgecolor="#b8860b",
            linewidth=1.5,
        )
    )
    ax.text(
        bx + batch_w / 2,
        by + batch_h - 0.4,
        "Batch: 2048 pairs",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )

    sub_w, sub_h = 3.6, 2.2
    sx, sy = bx + (batch_w - sub_w) / 2, by + 0.45
    ax.add_patch(
        FancyBboxPatch(
            (sx, sy),
            sub_w,
            sub_h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor="#cfe8fc",
            edgecolor="#2c5f9e",
            linewidth=1.5,
        )
    )
    ax.text(
        sx + sub_w / 2,
        sy + sub_h / 2,
        "~413 self-pairs\n(avg; same molecule,\n2 different real spectra)",
        ha="center",
        va="center",
        fontsize=13,
    )
    ax.text(
        bx + batch_w / 2,
        by - 0.3,
        "rest: ordinary MCES-regression pairs (unaffected)",
        ha="center",
        fontsize=11.5,
        style="italic",
        color="#555555",
    )

    _arrow(
        ax, (bx + batch_w + 0.15, by + batch_h / 2), (6.9, by + batch_h / 2), label=""
    )
    ax.text(
        5.9,
        by + batch_h / 2 + 0.6,
        "same transformer\nembedding as MCES head\n(no projection head)",
        ha="center",
        va="bottom",
        fontsize=11.5,
        color="#444444",
    )

    # N x N grid (illustrative 6x6, standing in for the real ~413x413) of
    # pairwise cosine similarities among just the self-pair embeddings:
    # diagonal forced toward 1 (same molecule), every off-diagonal cell
    # forced toward 0 (different molecule), via cross-entropy per row
    # (symmetrically, per column too).
    n = 6
    grid_x0, grid_y0 = 7.3, 2.0
    cell = 0.58
    for i in range(n):
        for j in range(n):
            is_diag = i == j
            facecolor = "#a8e6a1" if is_diag else "#f4a6a6"
            ax.add_patch(
                Rectangle(
                    (grid_x0 + j * cell, grid_y0 + (n - 1 - i) * cell),
                    cell,
                    cell,
                    facecolor=facecolor,
                    edgecolor="white",
                )
            )
            ax.text(
                grid_x0 + j * cell + cell / 2,
                grid_y0 + (n - 1 - i) * cell + cell / 2,
                "1" if is_diag else "0",
                ha="center",
                va="center",
                fontsize=11,
            )
    ax.text(
        grid_x0 + n * cell / 2,
        grid_y0 + n * cell + 0.35,
        "cosine_sim(anchor_i, positive_j) target\n(real matrix is ~413 x 413)",
        ha="center",
        fontsize=12.5,
        fontweight="bold",
    )
    ax.text(
        grid_x0 - 0.25,
        grid_y0 + n * cell / 2,
        "anchor i",
        ha="center",
        va="center",
        fontsize=12,
        rotation=90,
    )
    ax.text(
        grid_x0 + n * cell / 2,
        grid_y0 - 0.3,
        "positive j",
        ha="center",
        va="top",
        fontsize=12,
    )

    ax.text(
        grid_x0 + n * cell + 0.4,
        grid_y0 + n * cell / 2,
        "cross-entropy\nper row\n(+ per column,\nsymmetric)",
        ha="left",
        va="center",
        fontsize=12.5,
        color="#b71c1c",
    )

    ax.text(
        7.25,
        0.55,
        "Added on top of the primary MCES-regression MSE loss (weight 1.0). No dedicated projection head --\n"
        "the discrimination pressure acts directly on the same embedding the MCES head uses.",
        ha="center",
        va="center",
        fontsize=12.5,
        style="italic",
        color="#444444",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {output_path}")


def contrastive_projhead_schema(output_path: str):
    """Same diagram as contrastive_schema, with exactly one change
    highlighted: a dedicated projection head sits between the shared
    embedding and the cosine-similarity matrix, used ONLY for the
    contrastive loss -- the primary MCES-regression head still reads the
    raw (unprojected) embedding directly, unaffected."""
    fig, ax = plt.subplots(figsize=(17.0, 8.6))
    ax.set_xlim(0, 17.0)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    batch_w, batch_h = 4.4, 3.6
    bx, by = 0.3, 4.3
    ax.add_patch(
        FancyBboxPatch(
            (bx, by),
            batch_w,
            batch_h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor="#fff3cd",
            edgecolor="#b8860b",
            linewidth=1.5,
        )
    )
    ax.text(
        bx + batch_w / 2,
        by + batch_h - 0.4,
        "Batch: 2048 pairs",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )

    sub_w, sub_h = 3.6, 2.2
    sx, sy = bx + (batch_w - sub_w) / 2, by + 0.45
    ax.add_patch(
        FancyBboxPatch(
            (sx, sy),
            sub_w,
            sub_h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor="#cfe8fc",
            edgecolor="#2c5f9e",
            linewidth=1.5,
        )
    )
    ax.text(
        sx + sub_w / 2,
        sy + sub_h / 2,
        "~413 self-pairs\n(avg; same molecule,\n2 different real spectra)",
        ha="center",
        va="center",
        fontsize=13,
    )
    ax.text(
        bx + batch_w / 2,
        by - 0.3,
        "rest: ordinary MCES-regression pairs (unaffected)",
        ha="center",
        fontsize=11.5,
        style="italic",
        color="#555555",
    )

    _arrow(ax, (bx + batch_w + 0.15, by + batch_h / 2), (6.9, by + batch_h / 2))
    ax.text(
        5.9,
        by + batch_h / 2 + 0.6,
        "same transformer\nembedding as MCES head",
        ha="center",
        va="bottom",
        fontsize=11.5,
        color="#444444",
    )

    # The one thing that changes vs. contrastive_schema: a dedicated
    # projection head (NEW, highlighted) sits between the shared embedding
    # and the similarity matrix -- only this contrastive path goes through
    # it; the primary MCES head keeps reading the raw embedding directly
    # (shown via the dashed bypass arrow below).
    proj_w, proj_h = 2.9, 1.3
    px, py = 6.9, by + batch_h / 2 - proj_h / 2
    ax.add_patch(
        FancyBboxPatch(
            (px, py),
            proj_w,
            proj_h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor="#ffe0b2",
            edgecolor="#e65100",
            linewidth=2.2,
        )
    )
    ax.text(
        px + proj_w / 2,
        py + proj_h / 2,
        "NEW: projection head\nLinear-ReLU-Linear\n(dedicated weights)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#e65100",
    )

    _arrow(ax, (px + proj_w + 0.15, by + batch_h / 2), (10.3, by + batch_h / 2))

    _arrow(
        ax,
        (bx + batch_w + 0.15, by + 0.5),
        (9.0, 2.0),
        style="-|>",
        color="#888888",
    )
    ax.text(
        6.5,
        1.8,
        "primary MCES head: bypasses the projection head,\nuses the raw shared embedding directly (unaffected)",
        ha="center",
        va="top",
        fontsize=11.5,
        style="italic",
        color="#888888",
    )

    n = 6
    grid_x0, grid_y0 = 10.3, 4.5
    cell = 0.58
    for i in range(n):
        for j in range(n):
            is_diag = i == j
            facecolor = "#a8e6a1" if is_diag else "#f4a6a6"
            ax.add_patch(
                Rectangle(
                    (grid_x0 + j * cell, grid_y0 + (n - 1 - i) * cell),
                    cell,
                    cell,
                    facecolor=facecolor,
                    edgecolor="white",
                )
            )
            ax.text(
                grid_x0 + j * cell + cell / 2,
                grid_y0 + (n - 1 - i) * cell + cell / 2,
                "1" if is_diag else "0",
                ha="center",
                va="center",
                fontsize=11,
            )
    ax.text(
        grid_x0 + n * cell / 2,
        grid_y0 + n * cell + 0.35,
        "cosine_sim(proj(anchor_i), proj(positive_j))\n(real matrix is ~413 x 413)",
        ha="center",
        fontsize=12.5,
        fontweight="bold",
    )
    ax.text(
        grid_x0 + n * cell + 0.4,
        grid_y0 + n * cell / 2,
        "cross-entropy\nper row\n(+ per column,\nsymmetric)",
        ha="left",
        va="center",
        fontsize=12.5,
        color="#b71c1c",
    )

    ax.text(
        8.2,
        0.4,
        "Only change from the no-projection-head variant: the contrastive loss now scores proj(embedding)\n"
        "instead of the raw embedding directly -- decoupling its discrimination pressure from the primary MCES space.",
        ha="center",
        va="center",
        fontsize=12.5,
        style="italic",
        color="#444444",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {output_path}")


def cosine_head_schema(output_path: str):
    """One shared embedding, forking into the existing main task and the
    new auxiliary head -- the point being emphasized is that it's the SAME
    embedding doing double duty, not a separate encoder. How the GT target
    itself is computed is deliberately left unlabeled, since that's not the
    part worth explaining here. T-shaped layout (one horizontal row, one
    vertical stem down to two more boxes) instead of one long row -- keeps
    it slide-proportioned (taller, not a thin wide strip) at a large,
    readable font size."""
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    FS = 16

    # Top row: spectrum pair -> shared embedding -> (unchanged) main task.
    _box(
        ax,
        (0.2, 5.6),
        2.1,
        1.1,
        "Spectrum\npair",
        facecolor="#fff3cd",
        edgecolor="#b8860b",
        fontsize=FS,
    )
    _box(
        ax,
        (2.7, 5.6),
        2.8,
        1.1,
        "Shared\nembedding",
        facecolor="#cfe8fc",
        edgecolor="#2c5f9e",
        fontsize=FS,
    )
    _box(
        ax,
        (6.0, 5.6),
        3.3,
        1.1,
        "Main task:\nMCES similarity head\n(unchanged)",
        facecolor="#eef3fb",
        edgecolor="#2c5f9e",
        fontsize=FS,
    )
    _arrow(ax, (2.3, 6.15), (2.7, 6.15))
    _arrow(ax, (5.5, 6.15), (6.0, 6.15))

    # Vertical stem down from the shared embedding to the new head.
    _arrow(ax, (4.1, 5.6), (4.1, 4.5))

    _box(
        ax,
        (2.7, 3.4),
        2.8,
        1.1,
        "NEW:\nspectral_cosine\nprojection (MLP)",
        facecolor="#ffe0b2",
        edgecolor="#e65100",
        fontsize=FS,
    )

    _arrow(ax, (4.1, 3.4), (4.1, 2.3))

    _box(
        ax,
        (2.4, 1.2),
        3.4,
        1.1,
        "Predicted\nspectral cosine",
        facecolor="#e8f6e8",
        edgecolor="#2e7d32",
        fontsize=FS,
    )
    _arrow(ax, (5.8, 1.75), (6.5, 1.75))

    _box(
        ax,
        (6.5, 1.2),
        2.8,
        1.1,
        "MSE on cosine",
        facecolor="#f3e5f5",
        edgecolor="#6a1b9a",
        fontsize=FS,
    )

    ax.text(
        4.75,
        0.35,
        "Same embedding, two heads: the main task is untouched; this is extra supervision on top of it.",
        ha="center",
        va="center",
        fontsize=14,
        style="italic",
        color="#444444",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {output_path}")


def corn_schema(output_path: str):
    """Three stacked rows (classification chain on top, correction chain on
    bottom, joined by one vertical connector) instead of one long wide
    strip -- keeps it slide-proportioned (taller, not a thin wide strip) at
    a large, readable font size, same approach as cosine_head_schema. The
    CORN ordinal-loss internals are compressed into one descriptive box
    ("ordinal classification head") rather than spelled out."""
    fig, ax = plt.subplots(figsize=(12.6, 9.6))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 9.6)
    ax.axis("off")

    FS = 16

    # Row 1 (top): pair -> shared embedding -> abs diff.
    row1_y = 8.1
    _box(
        ax,
        (0.2, row1_y),
        2.1,
        1.1,
        "Spectrum\npair",
        facecolor="#fff3cd",
        edgecolor="#b8860b",
        fontsize=FS,
    )
    _box(
        ax,
        (2.7, row1_y),
        2.7,
        1.1,
        "Shared\nembedding",
        facecolor="#cfe8fc",
        edgecolor="#2c5f9e",
        fontsize=FS,
    )
    _box(
        ax,
        (5.8, row1_y),
        2.7,
        1.1,
        "abs(emb0\n- emb1)",
        facecolor="#eef3fb",
        edgecolor="#2c5f9e",
        fontsize=FS,
    )
    _arrow(ax, (2.3, row1_y + 0.55), (2.7, row1_y + 0.55))
    _arrow(ax, (5.4, row1_y + 0.55), (5.8, row1_y + 0.55))

    # Row 2 (middle): NEW ordinal head -> predicted bucket.
    row2_y = 5.9
    _box(
        ax,
        (5.8, row2_y),
        2.7,
        1.1,
        "NEW:\nordinal classification\nhead",
        facecolor="#ffe0b2",
        edgecolor="#e65100",
        fontsize=FS - 2,
    )
    _box(
        ax,
        (8.9, row2_y),
        3.5,
        1.1,
        "Predicted bucket:\nself, (0,2], (2,4],\n(4,6], (6,8], (8,inf)",
        facecolor="#e8f6e8",
        edgecolor="#2e7d32",
        fontsize=FS - 3,
    )
    _arrow(ax, (7.15, row1_y), (7.15, row2_y + 1.1))
    _arrow(ax, (8.5, row2_y + 0.55), (8.9, row2_y + 0.55))

    # Row 3 (bottom): primary raw prediction -> clip into bucket -> corrected.
    row3_y = 2.4
    _box(
        ax,
        (0.2, row3_y),
        2.9,
        1.1,
        "Primary head's\nraw MCES\nprediction",
        facecolor="#cfe8fc",
        edgecolor="#2c5f9e",
        fontsize=FS - 2,
    )
    _box(
        ax,
        (3.5, row3_y),
        3.3,
        1.1,
        "Clip into the\npredicted bucket's\nrange",
        facecolor="#f3e5f5",
        edgecolor="#6a1b9a",
        fontsize=FS - 2,
    )
    _box(
        ax,
        (7.2, row3_y),
        3.3,
        1.1,
        "= CORN-corrected\nprediction",
        facecolor="#fff3cd",
        edgecolor="#b8860b",
        fontsize=FS - 2,
    )
    _arrow(ax, (3.1, row3_y + 0.55), (3.5, row3_y + 0.55))
    _arrow(ax, (6.8, row3_y + 0.55), (7.2, row3_y + 0.55))

    # Connector from the predicted-bucket box down into the clip box.
    _arrow(ax, (10.65, row2_y), (5.15, row3_y + 1.1))

    # Shared embedding also feeds the (unchanged) primary head directly.
    _arrow(ax, (4.05, row1_y), (1.65, row3_y + 1.1))

    ax.text(
        6.3,
        0.9,
        "Loss weight 1.0, added on top of the primary MCES regression. CORN is the only trick that produces a SECOND\n"
        "usable prediction (the corrected one) -- hence the 4th table row, evaluated separately from the raw regression.",
        ha="center",
        va="center",
        fontsize=13,
        style="italic",
        color="#444444",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {output_path}")


def filip_schema(output_path: str):
    """Styled after the FILIP paper's own cross-modal late-interaction
    figure, adapted to our setting: ONE shared spectrum encoder (not two
    different modality towers) run on both spectra, producing per-peak
    tokens; a token-wise cosine-similarity matrix; then, per direction, a
    2-step reduction (aggregate over the other side's tokens, then pool
    over this side's own tokens) -- mirroring the reference's "1. Max /
    2. Mean" structure exactly. Unlike the reference (two separate
    in-batch contrastive matrices), our result is a single scalar score
    that REPLACES the CLS-cosine primary head and is trained via MSE
    against true MCES -- not an auxiliary loss added on top like the
    other 5 tricks."""
    fig, ax = plt.subplots(figsize=(18.0, 14.5))
    ax.set_xlim(0, 16.0)
    ax.set_ylim(0, 12.8)
    ax.axis("off")

    # -- bottom: caption ---------------------------------------------------
    ax.text(
        8.0,
        0.5,
        "Unlike the other 5 tricks, this REPLACES the CLS-cosine primary head (not an auxiliary loss added on top) --\n"
        "trained via MSE directly against true MCES, exactly where CLS-cosine used to be.",
        ha="center",
        va="center",
        fontsize=13,
        style="italic",
        color="#444444",
    )

    # -- spectra + shared encoder, run twice --------------------------------
    _box(
        ax,
        (0.2, 1.1),
        3.2,
        0.8,
        "Spectrum 0 (peaks)",
        facecolor="#fff3cd",
        edgecolor="#b8860b",
        fontsize=12.5,
    )
    _box(
        ax,
        (10.6, 1.1),
        3.2,
        0.8,
        "Spectrum 1 (peaks)",
        facecolor="#fff3cd",
        edgecolor="#b8860b",
        fontsize=12.5,
    )
    _arrow(ax, (1.8, 1.9), (1.8, 2.3))
    _arrow(ax, (12.2, 1.9), (12.2, 2.3))
    _box(
        ax,
        (0.2, 2.3),
        3.2,
        0.9,
        "Shared Spectrum\nTransformer Encoder",
        facecolor="#cfe8fc",
        edgecolor="#2c5f9e",
        fontsize=11.5,
    )
    _box(
        ax,
        (10.6, 2.3),
        3.2,
        0.9,
        "SAME Shared Spectrum\nTransformer Encoder",
        facecolor="#cfe8fc",
        edgecolor="#2c5f9e",
        fontsize=11.5,
    )
    ax.plot([3.4, 10.6], [2.75, 2.75], color="#aaaaaa", linewidth=1, linestyle=":")
    ax.text(
        7.0,
        3.0,
        "same weights",
        ha="center",
        fontsize=11.5,
        color="#444444",
        style="italic",
    )
    _arrow(ax, (1.8, 3.2), (1.8, 3.5))
    _arrow(ax, (12.2, 3.2), (12.2, 3.5))

    # -- per-peak tokens (chips) --------------------------------------------
    for k in range(6):
        ax.add_patch(
            Rectangle(
                (0.3 + k * 0.5, 3.5), 0.4, 0.4, facecolor="#cfe8fc", edgecolor="#2c5f9e"
            )
        )
    ax.text(
        1.8, 4.1, "per-peak tokens (N)", ha="center", fontsize=11.5, color="#444444"
    )
    for k in range(6):
        ax.add_patch(
            Rectangle(
                (10.7 + k * 0.5, 3.5),
                0.4,
                0.4,
                facecolor="#cfe8fc",
                edgecolor="#2c5f9e",
            )
        )
    ax.text(
        12.2, 4.1, "per-peak tokens (M)", ha="center", fontsize=11.5, color="#444444"
    )

    _arrow(ax, (1.8, 3.9), (5.2, 4.2))
    _arrow(ax, (12.2, 3.9), (9.5, 4.2))

    # -- yellow late-interaction box -----------------------------------------
    box_x0, box_y0, box_w, box_h = 3.7, 4.2, 8.8, 8.0
    ax.add_patch(
        FancyBboxPatch(
            (box_x0, box_y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#fff8e1",
            edgecolor="#e65100",
            linewidth=1.8,
        )
    )
    ax.text(
        box_x0 + 0.25,
        box_y0 + box_h - 0.5,
        "Token-wise Late Interaction",
        fontsize=17,
        color="#c0392b",
        fontweight="bold",
    )

    # Small N x M similarity grid.
    n = 5
    cell = 0.5
    gx, gy = box_x0 + 0.5, box_y0 + 4.6
    rng = np.random.default_rng(0)
    shades = rng.uniform(0.15, 0.85, size=(n, n))
    for i in range(n):
        for j in range(n):
            ax.add_patch(
                Rectangle(
                    (gx + j * cell, gy + (n - 1 - i) * cell),
                    cell,
                    cell,
                    facecolor=str(1 - shades[i, j] * 0.6),
                    edgecolor="white",
                )
            )
    ax.text(
        gx + n * cell / 2,
        gy - 0.3,
        "token-wise cosine\nsimilarity (N x M)",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    grid_right = gx + n * cell
    row_top = gy + n * cell - 0.5
    row_bot = gy + 0.5
    agg_box_x = grid_right + 1.6
    _arrow(ax, (grid_right + 0.15, row_top), (agg_box_x - 0.15, row_top))
    ax.text(
        (grid_right + agg_box_x) / 2,
        row_top + 0.25,
        "1. aggregate",
        ha="center",
        fontsize=11.5,
    )
    _arrow(
        ax, (grid_right + 0.15, row_bot), (agg_box_x - 0.15, row_bot), color="#2980b9"
    )
    ax.text(
        (grid_right + agg_box_x) / 2,
        row_bot + 0.25,
        "2. aggregate",
        ha="center",
        fontsize=11.5,
        color="#2980b9",
    )

    _box(
        ax,
        (agg_box_x, row_top - 0.5),
        2.9,
        1.05,
        "over spec 1's\ntokens (axis M)",
        facecolor="#fde8e8",
        edgecolor="#b71c1c",
        fontsize=11.5,
    )
    _box(
        ax,
        (agg_box_x, row_bot - 0.5),
        2.9,
        1.05,
        "over spec 0's\ntokens (axis N)",
        facecolor="#e3f2fd",
        edgecolor="#2980b9",
        fontsize=11.5,
    )

    ax.text(
        box_x0 + 0.3,
        gy - 2.0,
        "2. weighted-mean pool over this side's own tokens -> score_0to1 / score_1to0\n"
        "final FILIP score = 0.5 * (score_0to1 + score_1to0)",
        ha="left",
        va="top",
        fontsize=12.5,
        color="#444444",
    )

    ax.text(
        box_x0 + box_w / 2,
        box_y0 + 1.1,
        "Max vs. Mean: differ ONLY in step 1's aggregation rule -- literal max vs. plain average\n"
        "(both use plain step-2 pooling).\n"
        "Attention: changes BOTH steps -- learned attention weights (step 1)\n"
        "+ importance-weighted pooling (step 2).",
        ha="center",
        va="center",
        fontsize=12,
        style="italic",
        color="#444444",
    )

    # -- output --------------------------------------------------------------
    out_y = (row_top + row_bot) / 2 + 0.4
    output_box_x = box_x0 + box_w + 1.2
    _arrow(ax, (box_x0 + box_w + 0.1, out_y), (output_box_x - 0.1, out_y))
    _box(
        ax,
        (output_box_x, out_y - 0.55),
        2.4,
        1.1,
        "= FILIP\nsimilarity score",
        facecolor="#a8e6a1",
        edgecolor="#2e7d32",
        fontsize=12.5,
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def molecule_set_schema(output_path: str):
    """4 anchors x k=63 nearest-by-raw-MCES neighbors = 256 molecules/batch,
    2 spectra each, FULL 256x256 cross-matrix (every pair, not just
    anchor-to-neighbor) as the MCES-regression target -- 65,536 supervised
    pairs per step vs. 2,048 random pairs in the normal flat-pair regime.
    No auxiliary heads in this experiment -- purely a batch-composition
    change; validation always uses ordinary random pairs, unaffected."""
    fig, ax = plt.subplots(figsize=(16.5, 10.2))
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 10.2)
    ax.axis("off")

    FS = 16

    _box(
        ax,
        (0.3, 8.2),
        3.6,
        1.1,
        "Pick 4 anchor\nmolecules (random)",
        facecolor="#fff3cd",
        edgecolor="#b8860b",
        fontsize=FS - 3,
    )
    _arrow(ax, (3.9, 8.75), (4.5, 8.75))
    _box(
        ax,
        (4.5, 8.2),
        4.9,
        1.1,
        "For each anchor: 63 nearest\nmolecules by raw MCES",
        facecolor="#cfe8fc",
        edgecolor="#2c5f9e",
        fontsize=FS - 3.5,
    )
    _arrow(ax, (9.4, 8.75), (10.0, 8.75))
    _box(
        ax,
        (10.0, 8.2),
        3.5,
        1.1,
        "= 256 molecules\n(4 x 64)",
        facecolor="#e8f6e8",
        edgecolor="#2e7d32",
        fontsize=FS - 3,
    )

    _arrow(ax, (2.1, 8.2), (2.1, 7.3))
    _box(
        ax,
        (0.3, 6.2),
        5.3,
        1.1,
        "Each of the 256:\n2 different real spectra",
        facecolor="#fff3cd",
        edgecolor="#b8860b",
        fontsize=FS - 3.5,
    )
    _arrow(ax, (5.6, 6.75), (6.2, 6.75))
    _box(
        ax,
        (6.2, 6.2),
        4.9,
        1.1,
        "Encode all 512\nspectra once",
        facecolor="#cfe8fc",
        edgecolor="#2c5f9e",
        fontsize=FS - 3,
    )

    box_x0, box_y0, box_w, box_h = 0.3, 1.6, 15.9, 3.9
    ax.add_patch(
        FancyBboxPatch(
            (box_x0, box_y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#fff8e1",
            edgecolor="#e65100",
            linewidth=1.8,
        )
    )
    ax.text(
        box_x0 + 3.7,
        box_y0 + box_h - 0.5,
        "Full cross-matrix supervision",
        fontsize=FS + 1,
        color="#c0392b",
        fontweight="bold",
    )

    n = 5
    cell = 0.44
    gx, gy = box_x0 + 0.6, box_y0 + 0.4
    _arrow(ax, (2.9, 6.2), (gx + n * cell / 2, gy + n * cell - 0.3))
    rng = np.random.default_rng(1)
    shades = rng.uniform(0.1, 0.9, size=(n, n))
    for i in range(n):
        for j in range(n):
            ax.add_patch(
                Rectangle(
                    (gx + j * cell, gy + (n - 1 - i) * cell),
                    cell,
                    cell,
                    facecolor=str(1 - shades[i, j] * 0.6),
                    edgecolor="white",
                )
            )
    ax.text(
        gx + n * cell / 2,
        gy + n * cell + 0.2,
        "256 x 256 cosine\ncross-matrix",
        ha="center",
        va="bottom",
        fontsize=FS - 4,
        fontweight="bold",
    )

    _arrow(
        ax,
        (gx + n * cell + 0.3, gy + n * cell / 2),
        (gx + n * cell + 1.7, gy + n * cell / 2),
    )
    _box(
        ax,
        (gx + n * cell + 1.7, gy + n * cell / 2 - 0.55),
        3.0,
        1.1,
        "ALL pairwise\ncombinations",
        facecolor="#e3f2fd",
        edgecolor="#2980b9",
        fontsize=FS - 4,
    )

    ax.text(
        gx + n * cell + 5.2,
        gy + n * cell - 0.2,
        "= 65,536 supervised pairs / step\n(vs. 2,048 random pairs normally)",
        ha="left",
        va="center",
        fontsize=FS - 3,
        fontweight="bold",
        color="#2e7d32",
    )
    ax.text(
        gx + n * cell + 5.2,
        gy + n * cell / 2 - 0.3,
        "MSE vs. precomputed GT MCES",
        ha="left",
        va="center",
        fontsize=FS - 3,
        color="#444444",
    )

    ax.text(
        box_x0 + box_w / 2,
        0.8,
        "No auxiliary heads here -- purely a batch-composition change. Trade-off: ~32x more supervised pairs per step,\n"
        "but drawn from only 4 structural neighborhoods (enriched for close/small-MCES pairs), not randomly diverse across\n"
        "the whole dataset. Validation always uses ordinary random pairs -- entirely unaffected.",
        ha="center",
        va="center",
        fontsize=FS - 4,
        style="italic",
        color="#444444",
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    import sys

    globals()[sys.argv[1]](sys.argv[2])
