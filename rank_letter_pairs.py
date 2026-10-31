"""Rank letter pairs by how similar their neural activity is.

Reads the 31 x 31 split-half correlation matrix saved by letter_split_half_test.py
(results/split_half_correlations.csv, rows = even-trial average, columns = odd-trial average)
and ranks every pair of different letters. No neural data is needed.

The similarity of a pair is the average of the two cross-half correlations,
(even i vs odd j + even j vs odd i) / 2. Averaging both directions uses all four half-averages
and removes the arbitrary even/odd asymmetry.

The within-letter correlation (diagonal) of each letter is reported next to every pair as a
reliability check: a pair that involves a letter with a low diagonal is a noisier estimate.

Usage
    python rank_letter_pairs.py
    python rank_letter_pairs.py --top 20 --bottom 10
    python rank_letter_pairs.py --pair x y
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, default=Path("results/split_half_correlations.csv"))
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--figdir", type=Path, default=Path("figures"))
    p.add_argument("--top", type=int, default=15, help="how many top pairs to print and plot")
    p.add_argument("--bottom", type=int, default=5, help="how many bottom pairs to print")
    p.add_argument("--pair", nargs=2, metavar=("A", "B"), action="append", default=[],
                   help="also report the rank of this pair (can be repeated)")
    return p.parse_args()


def load_matrix(path):
    """Read the CSV written by letter_split_half_test.py -> (letter names, matrix)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    letters = rows[0][1:]
    if [r[0] for r in rows[1:]] != letters:
        raise ValueError("row labels and column labels differ, is this the right file?")
    M = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
    return letters, M


def rank_pairs(letters, M):
    """All pairs of different letters, most similar first. Returns a list of dicts."""
    S = (M + M.T) / 2
    within = np.diag(M)
    i_idx, j_idx = np.triu_indices(len(letters), k=1)
    pairs = [
        {
            "pair": f"{letters[i]}-{letters[j]}",
            "similarity": S[i, j],
            "even_i_odd_j": M[i, j],
            "even_j_odd_i": M[j, i],
            "within_i": within[i],
            "within_j": within[j],
        }
        for i, j in zip(i_idx, j_idx)
    ]
    pairs.sort(key=lambda d: d["similarity"], reverse=True)
    for rank, d in enumerate(pairs, start=1):
        d["rank"] = rank
    return pairs


def save_ranking(path, pairs):
    fields = ["rank", "pair", "similarity", "even_i_odd_j", "even_j_odd_i", "within_i", "within_j"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for d in pairs:
            writer.writerow([d["rank"], d["pair"]] + [f"{d[k]:.5f}" for k in fields[2:]])


def print_pairs(title, pairs):
    print(f"\n{title}")
    print(f"  {'rank':>4}  {'pair':<24} {'similarity':>10}   within-letter r (a, b)")
    for d in pairs:
        print(f"  {d['rank']:>4}  {d['pair']:<24} {d['similarity']:>10.3f}   "
              f"{d['within_i']:.2f}, {d['within_j']:.2f}")


def plot_top(pairs, path):
    labels = [d["pair"] for d in pairs][::-1]
    values = [d["similarity"] for d in pairs][::-1]
    fig, ax = plt.subplots(figsize=(6, 0.32 * len(pairs) + 1.2))
    ax.barh(labels, values, color="skyblue")
    ax.set_xlabel("similarity (mean cross-half correlation)")
    ax.set_title(f"Top {len(pairs)} most similar letter pairs")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.figdir.mkdir(parents=True, exist_ok=True)

    letters, M = load_matrix(args.matrix)
    pairs = rank_pairs(letters, M)
    total = len(pairs)

    print(f"{len(letters)} letters, {total} pairs, ranked by similarity (1 = most similar)")
    if (args.top):
        print_pairs(f"Top {args.top} most similar pairs:", pairs[:args.top])
        plot_top(pairs[:args.top], args.figdir / "top_pairs.png")
    if (args.bottom):
        print_pairs(f"Bottom {args.bottom} least similar pairs:", pairs[-args.bottom:])
    by_name = {d["pair"]: d for d in pairs}
    for a, b in args.pair:
        d = by_name.get(f"{a}-{b}") or by_name.get(f"{b}-{a}")
        if d is None:
            print(f"\n{a}-{b}: not found, letters are {letters}")
        else:
            print(f"\n{d['pair']}: rank {d['rank']} of {total}, similarity {d['similarity']:.3f}, "
                  f"within-letter r {d['within_i']:.2f} and {d['within_j']:.2f}")

    save_ranking(args.outdir / "pair_ranking.csv", pairs)
    print(f"\nfull ranking in {args.outdir}/pair_ranking.csv, figure in {args.figdir}/top_pairs.png")


if __name__ == "__main__":
    main()
