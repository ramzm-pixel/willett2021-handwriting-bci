# Handwriting BCI: exploring the Willett et al. 2021 data

Working through the dataset of [Willett et al. 2021, *High-performance brain-to-text communication via handwriting*](https://www.nature.com/articles/s41586-021-03506-2) (Nature) as a feasibility study before attempting a reproduction. The data are intracortical (192-electrode) recordings from participant T5 imagining handwriting single characters.

This repository is a learning log: each step is a separate commit.

## Step 1: is the neural activity for a character reproducible?

`letter_split_half_test.py` asks whether the neural activity pattern for each character is consistent enough to be told apart from the other characters.

1. For each of the 31 characters, average the trials separately over the even- and odd-numbered trials and subtract the grand mean across characters.
2. Correlate every even-half pattern with every odd-half pattern (31 x 31 matrix, diagonal = same character).
3. Statistic = mean(diagonal) - mean(off-diagonal).
4. Shuffle the character labels of the trials, repeat, and compare (permutation test).

### Results (session `t5.2019.05.08`, 1000 shuffles)

| | |
|---|---|
| Within-letter (diagonal) mean r | 0.209 |
| Between-letter (off-diagonal) mean r | -0.022 |
| Real statistic (diagonal - off-diagonal) | 0.230 |
| Shuffled labels: mean / 95th percentile / max | 0.006 / 0.013 / 0.019 |
| p | 0.001 (the smallest possible with 1000 shuffles) |

All 201 time bins (delay period plus writing) and all 192 electrodes were used. 837 trials, 31 characters.

![split-half correlation](figures/split_half_correlation.png)
![permutation null](figures/permutation_null.png)

In the matrix, the letters x and y (and a/q/g) correlate with each other almost as strongly as with themselves.

Reading the result: the same character produces a measurably more similar pattern across independent halves of the trials than different characters do. The data carry character-specific structure.

Limits:
- One session, one participant.
- The test says the patterns are reproducible, not how accurately single trials could be decoded.
- The shuffled null is centered slightly above zero. This is expected: subtracting the grand mean pushes the off-diagonal correlations slightly negative. The null goes through the same pipeline, so the p-value is not affected.

## Step 2: which letters have the most similar neural patterns?

`rank_letter_pairs.py` reads the matrix saved by Step 1 (`results/split_half_correlations.csv`, no neural data needed) and ranks all 465 pairs of different letters. The similarity of a pair is the average of its two cross-half correlations, (even i vs odd j + even j vs odd i) / 2.

### Results

Most similar pairs:

| Rank | Pair | Similarity |
|---:|---|---:|
| 1 | x-y | 0.218 |
| 2 | g-q | 0.156 |
| 3 | a-q | 0.156 |
| 4 | d-o | 0.128 |
| 5 | greaterThan-y | 0.127 |
| 6 | n-r | 0.125 |
| 7 | c-s | 0.120 |
| 8 | b-p | 0.118 |
| 9 | tilde-u | 0.113 |
| 10 | greaterThan-x | 0.109 |

Least similar pairs (rank 465 is the least similar):

| Rank | Pair | Similarity |
|---:|---|---:|
| 461 | p-q | -0.141 |
| 462 | c-x | -0.143 |
| 463 | a-y | -0.145 |
| 464 | a-x | -0.147 |
| 465 | c-y | -0.150 |

![top pairs](figures/top_pairs.png)

The full ranking, with the two directional values and each letter's within-letter r, is in `results/pair_ranking.csv`.

Observations (by eye, not tested here):
- Many of the most similar pairs look alike when written and share strokes: x/y (y is like x with a shorter first stroke), a/q/g, n/r, b/p.
- p and q are near the bottom even though both are reliable letters (within-letter r about 0.20 and 0.22). A negative value means the two letters deviate from the average pattern in opposite directions. One hypothesis is that they are mirror images, so their horizontal pen movement is reversed. This has not been checked against the pen trajectories.

Limits:
- This comes from a single even/odd split of one session, so the order within the top ten is noisy. Do not read much into 3rd vs 5th place.
- There is no significance test per pair.
- Pairs that involve a letter with a low within-letter r (h, l) are less reliable.
- The off-diagonal values are shifted slightly negative by the grand-mean subtraction (see Step 1), so small negative values are not meaningful.

## Run

Download the dataset from Dryad (see the paper's data availability statement; about 1.4 GB), unzip it into `data/` so that this file exists:

```
data/handwritingBCIData/Datasets/t5.2019.05.08/singleLetters.mat
```

```bash
pip install -r requirements.txt

# Step 1
python letter_split_half_test.py                 # defaults: 1000 shuffles, seed 47
python letter_split_half_test.py --n-perm 5000   # smaller p-value floor

# Step 2 (needs the CSV from Step 1)
python rank_letter_pairs.py                      # top 15 and bottom 5
python rank_letter_pairs.py --top 20 --bottom 10 --pair x y
```

Outputs: `figures/split_half_correlation.png`, `figures/permutation_null.png`, `results/split_half_correlations.csv` (the 31 x 31 matrix with letter labels), `results/pair_ranking.csv`, `figures/top_pairs.png`. The data folder is git-ignored.

## Next steps

- Test whether neural similarity follows how the letters are written, using the pen-trajectory templates from the dataset, and check the p/q mirror-image idea.
- Check whether the effect holds across session blocks, and across other sessions (analyzed separately, not pooled).
- Try to get the official decoding pipeline running (TensorFlow 1.15, HMM labeling, language model).