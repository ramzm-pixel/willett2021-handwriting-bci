"""Split-half reliability of letter-specific neural activity (Willett et al. 2021, participant T5).

Question: is the neural activity pattern for each handwritten character consistent enough
that it can be told apart from the other characters?

Method
  1. For every character, average its trials separately over the even- and odd-numbered
     trials (two independent halves) and subtract the grand mean over all characters,
     leaving the part of the activity that is specific to the character.
  2. Correlate every even-half pattern with every odd-half pattern -> an n x n matrix
     (rows: even half, columns: odd half). The diagonal is the within-character
     reliability, the off-diagonal entries are between-character similarities.
  3. Statistic = mean(diagonal) - mean(off-diagonal).
  4. Null distribution: shuffle the character labels of the trials, redo steps 1-3 and
     see where the real statistic falls. p = (1 + #{null >= real}) / (1 + n_perm).
     The grand mean is recomputed inside every shuffle so the null goes through exactly the
     same pipeline as the real data. Note that the null is centered slightly ABOVE zero:
     subtracting the grand mean forces the letter patterns to sum to ~0, which pushes the
     off-diagonal correlations slightly negative (about -1/(n-1) of the typical size) and
     so inflates "diagonal - off-diagonal" a little even for random labels. The shuffled
     null contains this offset, so the p-value is not affected.

Usage
    python letter_split_half_test.py
    python letter_split_half_test.py --n-perm 5000 --seed 1
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.io import loadmat

NEURAL_PREFIX = "neuralActivityCube_"
EXCLUDED = {"doNothing"}  # the no-movement condition is not a character


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path,
                   default=Path("data/handwritingBCIData/Datasets/t5.2019.05.08/singleLetters.mat"))
    p.add_argument("--figdir", type=Path, default=Path("figures"))
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--seed", type=int, default=47)
    p.add_argument("--n-jobs", type=int, default=-1, help="-1 = all cores")
    return p.parse_args()


def load_trials(path):
    """Return (letter names sorted alphabetically, X (trials, time, electrodes), y (letter index per trial))."""
    raw = loadmat(path)
    letters = sorted(
        key[len(NEURAL_PREFIX):]
        for key in raw
        if key.startswith(NEURAL_PREFIX) and key[len(NEURAL_PREFIX):] not in EXCLUDED
    )
    X = np.concatenate([raw[NEURAL_PREFIX + name] for name in letters])
    y = np.concatenate([np.full(len(raw[NEURAL_PREFIX + name]), i) for i, name in enumerate(letters)])
    return letters, X, y


def split_half_matrix(X, y, n_letters):
    """M[i, j] = correlation between the even-trial pattern of letter i and the odd-trial pattern of letter j."""
    groups = [X[y == i] for i in range(n_letters)]
    grand_mean = np.mean([g.mean(axis=0) for g in groups], axis=0)  # mean of the letter means
    even = np.stack([(g[0::2].mean(axis=0) - grand_mean).ravel() for g in groups])
    odd = np.stack([(g[1::2].mean(axis=0) - grand_mean).ravel() for g in groups])
    # corrcoef treats every row of the stacked input as a variable, so the result is
    # (2n x 2n); the top-right block is even-vs-odd.
    return np.corrcoef(even, odd)[:n_letters, n_letters:]


def statistic(M):
    n = len(M)
    diagonal = np.trace(M) / n
    off_diagonal = (M.sum() - np.trace(M)) / (n * n - n)
    return diagonal - off_diagonal


def permuted_statistic(X, labels, n_letters):
    return statistic(split_half_matrix(X, labels, n_letters))


def save_matrix(path, letters, M):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["even \\ odd"] + letters)
        for name, row in zip(letters, M):
            writer.writerow([name] + [f"{v:.5f}" for v in row])


def plot_matrix(letters, M, path):
    limit = np.abs(M).max()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(letters)))
    ax.set_xticklabels(letters, rotation=90, fontsize=7)
    ax.set_yticks(range(len(letters)))
    ax.set_yticklabels(letters, fontsize=7)
    ax.set_xlabel("odd-trial average")
    ax.set_ylabel("even-trial average")
    ax.set_title("Split-half correlation between letters")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_null(null, real, p_value, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null, bins=40, color="skyblue", label="shuffled labels")
    ax.axvline(real, color="black", linestyle="--", label=f"real: {real:.3f}")
    ax.set_xlabel("mean(diagonal) - mean(off-diagonal)")
    ax.set_ylabel("count")
    ax.set_title(f"Permutation test ({len(null)} shuffles), p = {p_value:.4f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    args.figdir.mkdir(parents=True, exist_ok=True)
    args.outdir.mkdir(parents=True, exist_ok=True)

    letters, X, y = load_trials(args.data)
    n = len(letters)
    print(f"{len(X)} trials, {n} letters, {X.shape[1]} time bins, {X.shape[2]} electrodes")

    M = split_half_matrix(X, y, n)
    real = statistic(M)
    print(f"within-letter (diagonal) mean r: {np.trace(M) / n:.3f}")
    print(f"between-letter (off-diagonal) mean r: {(M.sum() - np.trace(M)) / (n * n - n):.3f}")
    print(f"statistic (diagonal - off-diagonal): {real:.3f}")

    # Shuffled label vectors are drawn here, in the main process, so the result depends only on --seed.
    rng = np.random.default_rng(args.seed)
    shuffled = [rng.permutation(y) for _ in range(args.n_perm)]
    null = np.array(Parallel(n_jobs=args.n_jobs, verbose=0)(
        delayed(permuted_statistic)(X, labels, n) for labels in shuffled))

    p_value = (1 + np.sum(null >= real)) / (1 + args.n_perm)
    print(f"null: mean {null.mean():.4f}, 95th percentile {np.percentile(null, 95):.4f}, max {null.max():.4f}")
    print(f"p = {p_value:.4f}  ({args.n_perm} shuffles; the smallest possible p is {1 / (1 + args.n_perm):.4f})")

    save_matrix(args.outdir / "split_half_correlations.csv", letters, M)
    plot_matrix(letters, M, args.figdir / "split_half_correlation.png")
    plot_null(null, real, p_value, args.figdir / "permutation_null.png")
    print(f"figures in {args.figdir}/, matrix in {args.outdir}/split_half_correlations.csv")


if __name__ == "__main__":
    main()