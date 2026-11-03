"""Ranking, metrics and figure of the analog-discovery benchmark, shared by
05_score_simba.py, 06_evaluate.py and simba/training_diagnostics_callback.py.

Ground truth is per (query, library molecule) (gt_sim.npy of step 2, NaN =
not computed; if a queries.txt lies next to gt_sim.npy, its rows are matched to
the evaluated queries by SMILES); score matrices are per (query, library spectrum) in the order
of 01_build_sets.py, higher = better (NaN = ranked last).

Metrics (s = normalized MCES similarity, distance = 1 - s):
  best-of-10 distance: per query, min distance over the top-10 retrieved
    spectra (ignoring NaN ground truth); median and q05/q25/q75/q95 over queries.
    The oracle is each query's true best analog.
  precision@K (his obtain_precision_k_candidates, min threshold 0): the true
    best = argmax of the per-spectrum ground-truth row (first spectrum on ties);
    queries with best similarity <= 0 are skipped; hit if the best molecule is
    among the molecules of the top-K retrieved spectra.
  ROC AUC: pooled over all (query, spectrum) pairs with ground truth; positive =
    distance < 0.3; score = -rank of the spectrum in the query's ranking.
"""

import textwrap
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

K_VALUES = [5, 10, 20, 40, 80, 100]
TOP_N_ANALOG = 10
POSITIVE_DISTANCE = 0.3
QUANTILES = {"q05": 5, "q25": 25, "median": 50, "q75": 75, "q95": 95}
SEARCH_LABELS = {"A": "Search A (NIST20 + MassSpecGym)", "B": "Search B (GNPS)"}
# One color per tool (paper figure palette); "(paper)" checkpoints are drawn dashed.
ORACLE_COLOR = "#4C72B0"
TOOL_COLORS = {"modified cosine": "#DD8452", "spec2vec": "#55A868", "ms2deepscore": "#8172B2",
               "dreams": "#937860", "simba": "#C44E52"}


# colors for the 2nd, 3rd, ... variant of the same tool in one figure
EXTRA_COLORS = {"simba": ["#E377C2", "#FF7F0E", "#8B0000", "#17BECF"]}


def method_style(label):
    color = next((c for tool, c in TOOL_COLORS.items() if label.lower().startswith(tool)), "#333333")
    return color, "--" if "(paper" in label.lower() else "-"


def method_styles(labels):
    """(color, linestyle) per label; several variants of one tool get distinct colors."""
    styles, seen = {}, {}
    for label in labels:
        tool = next((t for t in TOOL_COLORS if label.lower().startswith(t)), None)
        color, linestyle = method_style(label)
        n = seen.get(tool, 0)
        if tool is not None and n > 0:
            extra = EXTRA_COLORS.get(tool, [])
            color = extra[(n - 1) % len(extra)] if extra else color
        seen[tool] = n + 1
        styles[label] = (color, linestyle)
    return styles


def compute_ranking(similarities1, similarities1_integer, similarities2_integer):
    """Sebastian's compute_ranking: per query, lexsort by (integer MCES,
    edit-distance class, MCES similarity), all descending -> rank score in
    (0, 1], higher = better."""
    ranking_total = np.zeros(similarities1.shape, dtype=float)
    for row_index, (row_sim, row_int, row_int2) in enumerate(zip(similarities1, similarities1_integer, similarities2_integer)):
        sorted_indices = np.lexsort((-row_sim, -row_int2, -row_int))
        ranking = np.empty_like(sorted_indices, dtype=float)
        ranking[sorted_indices] = np.arange(len(row_sim), 0, -1)
        ranking /= len(row_sim)
        ranking_total[row_index] = ranking
    return ranking_total


def simba_ranking(sim_mces, ed_class):
    """SIMBA's ranking matrix from its MCES similarity and edit-distance class predictions."""
    return compute_ranking(sim_mces, np.round(sim_mces * 40), ed_class)


class GroundTruth:
    """Ground truth of one search, with everything the metrics reuse precomputed."""

    def __init__(self, gt_dir, sets_dir, search):
        self.search = search
        self.gt = np.load(Path(gt_dir) / f"search_{search}" / "gt_sim.npy")
        self.mol_idx = np.load(Path(sets_dir) / f"search_{search}" / "spectrum_mol_idx.npy")
        gt_queries = Path(gt_dir) / f"search_{search}" / "queries.txt"
        if gt_queries.exists():  # ground truth computed for more queries: take the evaluated ones by SMILES
            row = {s: i for i, s in enumerate(gt_queries.read_text().split())}
            queries = (Path(sets_dir) / f"search_{search}" / "queries.txt").read_text().split()
            self.gt = self.gt[[row[q] for q in queries]]
        self.n_queries, self.n_spectra = self.gt.shape[0], len(self.mol_idx)
        gt_spec = self.gt[:, self.mol_idx]
        best_spec = np.argmax(np.where(np.isnan(gt_spec), -np.inf, gt_spec), axis=1)  # his rule: first max
        best_val = gt_spec[np.arange(self.n_queries), best_spec]
        self.best_mol = self.mol_idx[best_spec]
        self.include = best_val > 0
        self.has_gt = ~np.isnan(gt_spec)
        self.positive = (1 - gt_spec[self.has_gt]) < POSITIVE_DISTANCE
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.oracle = 1 - np.nanmax(self.gt, axis=1)

    def load_scores(self, method_dir):
        """Score matrix of a method dir (search_X/ranking.npy or similarity.npy); NaN -> -inf."""
        d = Path(method_dir) / f"search_{self.search}"
        path = d / "ranking.npy" if (d / "ranking.npy").exists() else d / "similarity.npy"
        scores = np.load(path).astype(np.float32)
        if scores.shape != (self.n_queries, self.n_spectra):
            raise ValueError(f"{path}: shape {scores.shape}, expected {(self.n_queries, self.n_spectra)}")
        return np.where(np.isnan(scores), -np.inf, scores)

    def evaluate(self, scores):
        scores = np.where(np.isnan(scores), -np.inf, scores)
        order = np.argsort(-scores, axis=1, kind="stable")
        top_mols = self.mol_idx[order[:, : max(K_VALUES + [TOP_N_ANALOG])]]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN top-10 rows -> NaN, counted below
            best10 = np.nanmax(np.take_along_axis(self.gt, top_mols[:, :TOP_N_ANALOG], axis=1), axis=1)
        dist10 = 1 - best10
        valid = ~np.isnan(dist10)

        precision = {k: float((top_mols[self.include, :k] == self.best_mol[self.include, None]).any(axis=1).mean())
                     for k in K_VALUES}

        ranks = np.empty(scores.shape, dtype=np.float32)
        ranks[np.arange(self.n_queries)[:, None], order] = np.arange(self.n_spectra, dtype=np.float32)[None, :]
        score = -ranks[self.has_gt]
        fpr, tpr, _ = roc_curve(self.positive, score)
        step = max(1, len(fpr) // 2000)
        return {
            "best_of_10_distance": {name: float(np.percentile(dist10[valid], q)) for name, q in QUANTILES.items()},
            "n_queries_without_gt_in_top10": int((~valid).sum()),
            "roc_auc": float(roc_auc_score(self.positive, score)),
            "tpr_at_fpr_0.05": float(np.interp(0.05, fpr, tpr)),
            "precision_at_k": precision,
            "_dist10": dist10[valid],
            "_roc": (fpr[::step], tpr[::step]),
        }


def axis_label(label, width=12):
    """Tool name on the first line, the "(...)" part wrapped to short lines."""
    name, _, rest = label.partition(" (")
    return name if not rest else name + "\n" + textwrap.fill("(" + rest, width)


def draw_search(axes_row, gt, results):
    """One figure row: violins of best-of-10 distance (with oracle), ROC, precision@K."""
    ax_vio, ax_roc, ax_prec = axes_row
    data = [gt.oracle[~np.isnan(gt.oracle)]] + [r["_dist10"] for r in results.values()]
    names = ["Best candidate\n(oracle)"] + [axis_label(k) for k in results]
    per_method = method_styles(list(results))
    styles = [(ORACLE_COLOR, "-")] + [per_method[k] for k in results]
    for i, (values, (color, linestyle)) in enumerate(zip(data, styles), 1):  # violin + inner box, as in the paper figure
        body = ax_vio.violinplot([values], positions=[i], showmeans=False, showmedians=False, showextrema=False)["bodies"][0]
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
        body.set_alpha(0.85)
        if linestyle == "--":  # "(paper ...)" entries: lighter and hatched, matching their dashed lines
            body.set_alpha(0.45)
            body.set_hatch("//")
    ax_vio.boxplot(data, widths=0.1, showfliers=True, medianprops=dict(color="black", linewidth=1.5),
                   flierprops=dict(marker="o", markersize=2, markerfacecolor="none"))
    ax_vio.set_xticks(range(1, len(names) + 1), names, fontsize=8)
    ax_vio.set(ylabel="Normalized MCES distance", ylim=(-0.05, 1.1),
               title=f"{SEARCH_LABELS[gt.search]}: best of top-{TOP_N_ANALOG} retrieved")
    for label, r in results.items():
        color, linestyle = per_method[label]
        fpr, tpr = r["_roc"]
        ax_roc.plot(fpr, tpr, color=color, ls=linestyle, lw=1.6, label=f"{label} (AUC {r['roc_auc']:.3f})")
        ax_prec.plot(K_VALUES, [r["precision_at_k"][k] for k in K_VALUES], color=color, ls=linestyle,
                     marker="*", markersize=7, label=label)
    ax_roc.plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":")
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1),
               title=f"ROC, positive = distance < {POSITIVE_DISTANCE}")
    ax_roc.legend(fontsize=8, loc="lower right")
    ax_prec.set(xlabel="K candidates", ylabel="Precision", xlim=(0, 100), ylim=(0, None),
                title=f"Precision@K ({int(gt.include.sum())} queries)")
    ax_prec.legend(fontsize=8, loc="lower right")
    for ax in (ax_vio, ax_roc, ax_prec):
        ax.grid(True, alpha=0.6, color="gray", linewidth=0.6)
        ax.set_axisbelow(True)
