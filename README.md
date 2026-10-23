# Reproducing "Fuzzy natural neighbors for outlier detection" (Task 2)

Python reproduction of the main experiment (Section 4) of
Saltos, Weber & Pedrycz, *Applied Soft Computing* 186 (2026) 114114.

The authors released MATLAB code and the datasets at
https://github.com/rjsaltos1989/FuNaNOutlier (MIT licence). This folder contains an
independent Python port of their four methods (FuNaN I, II, III, FuDSR) plus the crisp
DSR baseline. It also includes the 18 datasets copied from that repository.

## Folder contents

| File | What it is |
| --- | --- |
| `funan.py` | The algorithms: neighbour ranks, fuzzy memberships, the four detectors, DSR, and the AUC/Precision metrics |
| `reproduce.py` | Runs all 18 datasets, prints our results next to the paper's, saves CSV files |
| `data/` | The 18 datasets (`Data` = feature matrix, `y` = 1 for outliers, 0 for inliers) |
| `results_reference/` | The output I obtained, so you can check yours against it |
| `requirements.txt` | Python packages needed |

## Step-by-step

1. **Install Python 3.10 or newer** (python.org or Anaconda).
2. **Open a terminal in this folder** and install the packages:
   ```
   pip install -r requirements.txt
   ```
3. **Run the main experiment (Tables 3 and 4, standardized data):**
   ```
   python reproduce.py
   ```
   It takes about 1 minute and needs about 1.5 GB of RAM. Each dataset prints one line as it
   finishes. At the end you get two comparison tables (AUC and Precision), each showing *ours*, *paper*
   and *ours − paper*.
4. **Run the second comparison (Tables 7 and 8, min–max data):**
   ```
   python reproduce.py --normalization minmax
   ```
5. **Collect the outputs** from the new `results/` folder:
   - `auc_<norm>.csv` and `precision_<norm>.csv` hold the side-by-side comparison with the paper.
   - `funan3_auc_per_gamma_<norm>.csv` holds FuNaN III's AUC for each γ (compare with Table 11).
6. **Optional:**
   - To test a few datasets only: `python reproduce.py --datasets Motivation Glass Lympho`
   - If you run out of memory, lower the block size: `--block 250`

## Parameters

| Parameter | Value used | Where it comes from |
| --- | --- | --- |
| Normalization | z-score (std with n−1) for Tables 3–4; min–max to [0, 1] for Tables 7–8 | Paper §4.2 and §4.3; authors' code |
| Distance | Euclidean | Authors' code (`pdist2`, `knnsearch`) |
| Membership | μ_ij = 1/r_ij; υ_ij = min(μ_ij, μ_ji); self excluded | Eqs. 10–12 |
| Fuzzifier m | 2 | Footnote 3; authors' code |
| γ (FuNaN III) | 0.1, 0.2, …, 0.9; AUC and Precision averaged over the nine runs | §4.2; Table 11 |
| Bounds A, B | A = n/2, B = n(H_n − 1) | Eq. 16 |
| FuNaN I / II score | 1 / fuzzy cardinality, 1 / ψ (higher = more outlying) | Eqs. 14–15; authors' code |
| DSR | λ from NaN Search; points with no natural neighbour get NaN and are dropped | Eq. 4; authors' code |
| Detection threshold α / top-η | Not needed: AUC and Precision are threshold-free | — |
| Precision | Highest precision over all score thresholds (`max(Prec(2:end))` from MATLAB `perfcurve`) | Authors' code |
| Datasets | 18 datasets; SMTP and Shuttle use the authors' 10,000-point subsamples | Table 2; repository |
| Block size | 1000 rows (memory only; does not change results) | Implementation detail |

## Results

**Min–max (Tables 7 and 8):** exact reproduction. All 72 AUC values match to 4 decimals,
apart from a difference of 0.0001 on Shuttle. All Precision values match, apart from FuDSR on
Shuttle (0.9439 vs 0.9444). The means are identical: AUC 0.8189 / 0.8091 / 0.8189 / 0.8329 and
Precision 0.7624 / 0.7333 / 0.8269 / 0.7708 for FuNaN I / II / III / FuDSR.

**Z-score (Tables 3 and 4, the main experiment):** near-exact reproduction.

| Mean AUC | DSR | FuNaN I | FuNaN II | FuNaN III | FuDSR |
| --- | --- | --- | --- | --- | --- |
| Ours | 0.8196 | 0.8151 | 0.8048 | 0.8153 | 0.8389 |
| Paper | 0.8202 | 0.8152 | 0.8048 | 0.8152 | 0.8389 |

AUC matches exactly on 9 of the 18 datasets and within 0.002 on six more. The largest
differences are on SMTP (0.011), Shuttle (0.008) and Mammography (0.008). Per-γ AUCs for FuNaN III match Table 11. Precision matches except on a
few datasets, mainly Shuttle, where we get 0.75 instead of 1.00 for FuNaN I–III. Mean precision
is about 0.014 lower for FuNaN I–III and the same for FuDSR. The paper's conclusions still hold:
FuDSR has the best mean AUC and the fuzzy methods beat DSR on mean precision.

## Why small differences remain (z-score run only)

1. **Different preprocessing for Table 3.** The public MATLAB code only contains the min–max
   version, and that version matches Tables 7–8 exactly. The standardized pipeline behind
   Table 3 is therefore not available. For Shuttle, standardizing with the mean and standard deviation of the
   *full* 49,097-point dataset before subsampling gives 0.8352 / 0.7936 / 0.8568 / 0.9943,
   almost exactly the paper's 0.8347 / 0.7930 / 0.8563 / 0.9943. This suggests Table 3 was
   computed with a slightly different standardization or subsample than the files released. The
   same may apply to SMTP.
2. **Tied distances.** Shuttle has about 180 tied distances per point, so the neighbour ranks
   depend on how ties are broken (MATLAB's k-d tree versus our stable sort). A jitter of 1e-9
   moves FuNaN I's AUC on Shuttle by about ±0.0005.
3. **Precision is fragile.** "Precision" here is the highest precision at any threshold, so it is
   often decided by whether the single top-scoring point is an outlier. One swapped point
   changes it from 1.0 to 0.75, as on Shuttle. AUC uses every point and is far more stable.
