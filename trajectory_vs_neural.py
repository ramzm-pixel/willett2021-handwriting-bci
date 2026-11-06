"""Do letters with similar pen trajectories also have similar neural patterns?

Two letter-by-letter matrices are compared:
  S  neural similarity    read from results/split_half_correlations.csv (high = similar)
  D  trajectory distance  DTW between the mouse-template velocity profiles (low = similar)

If similar-looking letters are neurally similar, Spearman(S, D) over letter pairs is NEGATIVE.
Significance comes from a Mantel permutation test: the letter labels of D are shuffled
(rows and columns together), because pair entries share letters and are not independent.

No neural data is needed. The CSV is written by letter_split_half_test.py and therefore
describes whatever time bins and electrodes that script used (by default all 201 bins).

Usage
    python trajectory_vs_neural.py
    python trajectory_vs_neural.py --n-points 0         # keep original template lengths (length-bias check)
    python trajectory_vs_neural.py --measure frechet --no-rms
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import similaritymeasures
from scipy.io import loadmat
from scipy.stats import rankdata


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, default=Path("results/split_half_correlations.csv"))
    p.add_argument("--templates", type=Path,
                   default=Path("data/handwritingBCIData/Datasets/computerMouseTemplates.mat"))
    p.add_argument("--figdir", type=Path, default=Path("figures"))
    p.add_argument("--n-points", type=int, default=100,
                   help="resample every template to this many points (0 = keep original lengths)")
    p.add_argument("--no-rms", action="store_true", help="do not scale each template to unit RMS")
    p.add_argument("--measure", choices=["dtw", "frechet"], default="dtw")
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=47)
    return p.parse_args()


# ---------------------------------------------------------------- neural side
def load_matrix(path):
    """Read the CSV written by letter_split_half_test.py -> (letter names, matrix)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    letters = rows[0][1:]
    if [r[0] for r in rows[1:]] != letters:
        raise ValueError("row labels and column labels differ, is this the right file?")
    M = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
    return letters, M


def neural_similarity(M):
    """S[i, j]: how similar letter i's and letter j's letter-specific activity is.

    M[i, j] is the correlation between the even-trial average of letter i and the odd-trial
    average of letter j, so its diagonal is the within-letter (split-half) reliability.
    S symmetrizes M by averaging (i, j) and (j, i).
    """
    return (M + M.T) / 2, np.diag(M)


# ---------------------------------------------------------------- trajectory side
def load_templates(path, names, n_points, unit_rms):
    """letter name -> (time, 2) velocity template, optionally resampled and scaled."""
    raw = loadmat(path)
    missing = [n for n in names if n not in raw]
    if missing:
        raise KeyError(f"letters missing from the template file: {missing}")

    originals = {}
    for name in names:
        arr = np.asarray(raw[name], dtype=float)
        if arr.shape[1] != 2 and arr.shape[0] == 2:
            arr = arr.T  # we want (time, 2): x-velocity and y-velocity
        originals[name] = arr

    processed = {}
    for name, arr in originals.items():
        if n_points:
            t_old = np.linspace(0, 1, len(arr))
            t_new = np.linspace(0, 1, n_points)
            arr = np.column_stack([np.interp(t_new, t_old, arr[:, c]) for c in range(2)])
        if unit_rms:
            arr = arr / np.sqrt((arr ** 2).sum(axis=1).mean())
        processed[name] = arr
    return processed, {name: len(arr) for name, arr in originals.items()}


def trajectory_distances(templates, names, measure):
    n = len(names)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = templates[names[i]], templates[names[j]]
            if measure == "dtw":
                D[i, j] = similaritymeasures.dtw(a, b)[0]
            else:
                D[i, j] = similaritymeasures.frechet_dist(a, b)
            D[j, i] = D[i, j]
    return D


# ---------------------------------------------------------------- statistics
def spearman(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]


def mantel_spearman(S, D, n_perm, seed):
    """Spearman correlation of the upper triangles of S and D, with a permutation null.

    Spearman = Pearson on ranks, so both matrices are turned into standardized ranks once.
    A permutation only changes which pair's rank lands where, so each shuffle is cheap.
    """
    n = len(S)
    iu = np.triu_indices(n, k=1)

    def standardized_ranks(M):
        r = rankdata(M[iu])
        return (r - r.mean()) / r.std()

    z_s = standardized_ranks(S)
    z_d = np.zeros((n, n))
    z_d[iu] = standardized_ranks(D)
    z_d = z_d + z_d.T

    observed = float(np.mean(z_s * z_d[iu]))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(n)  # relabel the letters of D
        null[k] = np.mean(z_s * z_d[np.ix_(p, p)][iu])

    p_negative = (1 + np.sum(null <= observed)) / (1 + n_perm)  # predicted direction
    p_two_sided = (1 + np.sum(np.abs(null) >= abs(observed))) / (1 + n_perm)
    return observed, p_negative, p_two_sided


# ---------------------------------------------------------------- reporting
def report_pairs(names, S, D, top=10):
    iu = np.triu_indices(len(names), k=1)
    labels = [f"{names[i]}-{names[j]}" for i, j in zip(*iu)]
    neural_rank = rankdata(-S[iu], method="min").astype(int)  # 1 = most neurally similar
    traj_rank = rankdata(D[iu], method="min").astype(int)     # 1 = closest trajectories
    total = len(labels)

    def show(title, order):
        print(f"\n{title}")
        for k in order[:top]:
            print(f"  {labels[k]:<24} neural rank {neural_rank[k]:>3}   trajectory rank {traj_rank[k]:>3}   (of {total})")

    show("Most neurally similar pairs:", np.argsort(neural_rank))
    show("Closest trajectories:", np.argsort(traj_rank))


def plot_matrices(names, S, D, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    S_show = S.copy()
    np.fill_diagonal(S_show, np.nan)  # the diagonal (within-letter reliability) would wash out the colors
    limit = np.nanmax(np.abs(S_show))
    panels = [
        (axes[0], S_show, "neural similarity (higher = more similar)", "RdBu_r", -limit, limit),
        (axes[1], D, "trajectory distance (lower = more similar)", "viridis_r", None, None),
    ]
    for ax, M, title, cmap, lo, hi in panels:
        im = ax.imshow(M, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(S, D, rho, path):
    iu = np.triu_indices(len(S), k=1)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(D[iu], S[iu], s=8, alpha=0.5)
    ax.set_xlabel("trajectory distance")
    ax.set_ylabel("neural similarity")
    ax.set_title(f"one point per letter pair, Spearman rho = {rho:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    args.figdir.mkdir(parents=True, exist_ok=True)

    names, M = load_matrix(args.matrix)
    S, within = neural_similarity(M)
    templates, lengths = load_templates(args.templates, names, args.n_points, not args.no_rms)
    D = trajectory_distances(templates, names, args.measure)

    iu = np.triu_indices(len(names), k=1)
    print(f"{len(names)} letters, axis order: {names}")
    print(f"neural similarity read from {args.matrix}")
    print(f"within-letter split-half r: mean {within.mean():.3f}   "
          f"across-letter (symmetrized) mean {S[iu].mean():.3f}")

    length_sums = np.array([lengths[names[i]] + lengths[names[j]] for i, j in zip(*iu)])
    print(f"length bias check: Spearman(distance, sum of original template lengths) = "
          f"{spearman(D[iu], length_sums):.2f}  (should be near 0 after resampling)")

    rho, p_negative, p_two_sided = mantel_spearman(S, D, args.n_perm, args.seed)
    print(f"\nSpearman(S, D) = {rho:.3f}   (negative = similar trajectories are neurally similar)")
    print(f"permutation p, predicted direction (rho more negative than shuffled): {p_negative:.4f}")
    print(f"permutation p, two-sided: {p_two_sided:.4f}   ({args.n_perm} shuffles)")

    report_pairs(names, S, D)
    plot_matrices(names, S, D, args.figdir / "neural_vs_trajectory_matrices.png")
    plot_scatter(S, D, rho, args.figdir / "neural_vs_trajectory_scatter.png")
    print(f"\nfigures saved in {args.figdir}/")


if __name__ == "__main__":
    main()
