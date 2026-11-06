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

## Step 3: does neural similarity follow how the letters are written?

`trajectory_vs_neural.py` compares two letter-by-letter tables over the same 465 pairs:

- **S**, neural similarity, read from the Step 1 matrix (higher = more similar).
- **D**, trajectory distance, computed from `computerMouseTemplates.mat`, which holds a 2-D pen-velocity template for each character (lower = more similar).

Method:
1. Resample every template to 100 points and scale it to unit RMS speed, so duration and arbitrary units do not drive the distance.
2. D = dynamic time warping (DTW) distance between every pair of templates.
3. Spearman correlation between the 465 values of S and the 465 values of D. S is a similarity and D is a distance, so a relationship shows up as a **negative** correlation.
4. Mantel permutation test: shuffle the letter labels of D (10,000 times, rows and columns together) and see how often a correlation this negative appears. The letters are shuffled, not the pairs, because pairs that share a letter are not independent.

### Results (session `t5.2019.05.08`, neural side uses all 201 bins)

| Distance measure | Spearman rho | p (10,000 shuffles) | Length-bias check |
|---|---:|---|---:|
| DTW, unit RMS (default) | -0.49 | < 0.0001 | -0.04 |
| DTW, no RMS scaling | -0.38 | < 0.0001 | -0.25 |
| Frechet, unit RMS | -0.23 | < 0.0001 | -0.05 |

The length-bias check is the Spearman correlation between a pair's distance and the sum of the two original template lengths. It should be near 0. Without RMS scaling it is -0.25, so that variant is partly measuring template size or speed and is shown only as a check.

![neural and trajectory matrices](figures/neural_vs_trajectory_matrices.png)
![scatter](figures/neural_vs_trajectory_scatter.png)

Pairs at the top of one ranking tend to rank high in the other. With the default measure, the 10 most neurally similar pairs have an average trajectory rank of about 51 of 465, and the 10 closest-trajectory pairs have an average neural rank of about 43. With no relationship both would be near 233.

Reading the result: letters whose pen-velocity profiles are alike tend to have more similar neural patterns. The direction holds for all three distance measures and is significant in each. The size depends on the measure, from -0.23 to -0.49.

Limits:
- One session, one participant.
- The neural similarity is noisy (within-letter r is about 0.21), so the correlation understates any true relationship.
- Letters that share strokes also share starting direction, speed and stroke count. This analysis cannot say whether the neural similarity follows shape specifically or those other properties.
- The templates are mouse-drawn approximations of how letters are written, not T5's own imagined movements.
- Velocity-based distances treat mirror-image letters as far apart, because the sign of the horizontal velocity flips.
- The pair x-y has the highest neural similarity but a middling DTW rank (84) and a poor Frechet rank (434), so it is a clear exception.
- The neural similarity covers the whole trial, including the delay period before the go cue.

## Run

Download the dataset from Dryad (see the paper's data availability statement; about 1.4 GB), unzip it into `data/` so that these files exist:

```
data/handwritingBCIData/Datasets/t5.2019.05.08/singleLetters.mat
data/handwritingBCIData/Datasets/computerMouseTemplates.mat
```

```bash
pip install -r requirements.txt

# Step 1
python letter_split_half_test.py                 # defaults: 1000 shuffles, seed 47
python letter_split_half_test.py --n-perm 5000   # smaller p-value floor

# Step 2 (needs the CSV from Step 1)
python rank_letter_pairs.py                      # top 15 and bottom 5
python rank_letter_pairs.py --top 20 --bottom 10 --pair x y

# Step 3 (needs the CSV from Step 1)
python trajectory_vs_neural.py                   # DTW, unit RMS, 10,000 shuffles
python trajectory_vs_neural.py --measure frechet
python trajectory_vs_neural.py --no-rms
```

Outputs: `figures/split_half_correlation.png`, `figures/permutation_null.png`, `results/split_half_correlations.csv` (the 31 x 31 matrix with letter labels), `results/pair_ranking.csv`, `figures/top_pairs.png`, `figures/neural_vs_trajectory_matrices.png`, `figures/neural_vs_trajectory_scatter.png`. The data folder is git-ignored. Each Step 3 run overwrites the two Step 3 figures, so the committed figures come from the default run.

## Next steps

- Check whether the effects hold across session blocks, and across other sessions (analyzed separately, not pooled).
- Check the p/q mirror-image idea against the pen trajectories.
- Try to get the official decoding pipeline running (TensorFlow 1.15, HMM labeling, language model).