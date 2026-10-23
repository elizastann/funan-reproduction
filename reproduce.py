"""
Reproduce the main experiment of Saltos, Weber & Pedrycz (2026), Section 4:
AUC and Precision of FuNaN I, FuNaN II, FuNaN III, FuDSR (and the crisp DSR baseline)
on the 18 benchmark datasets, compared with the values printed in the paper.

Usage examples
    python reproduce.py                                   # Tables 3-4 (standardized data)
    python reproduce.py --normalization minmax            # Tables 7-8 (min-max data)
    python reproduce.py --datasets Motivation Glass Lympho
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import scipy.io as sio

from funan import run_all

# paper name -> file name in the authors' DataOut folder (same order as Table 2)
DATASETS = {
    "Cluto": "Cluto", "Motivation": "Motivation", "Zelnik2": "Zelnik2", "Zelnik4": "Zelnik4",
    "Zelnik6": "Zelnik6", "SMTP": "SmtpS", "BankNote": "BankNote", "Annthyroid": "Annthyroid",
    "Mammography": "Mammo", "Thyroid": "Thyroid", "Vertebral": "Vertebral", "Rice": "Rice",
    "Glass": "Glass", "Shuttle": "ShuttleS", "Vowels": "Vowels", "Pendigits": "PenDigits",
    "Lympho": "Lympho", "WBC": "WBC",
}
METHODS = ["DSR", "FuNaN I", "FuNaN II", "FuNaN III", "FuDSR"]

# ---- values reported in the paper -------------------------------------------------
# Tables 3 (AUC) and 4 (Precision): comparison with NaN-based methods (standardized data)
# columns: DSR, FuNaN I, FuNaN II, FuNaN III, FuDSR
T3 = {
    "Cluto": [.9540, .9819, .9848, .9733, .8811], "Motivation": [1, 1, 1, 1, 1],
    "Zelnik2": [1, .9582, .9347, .9768, .9960], "Zelnik4": [.9852, .9813, .9762, .9849, .9816],
    "Zelnik6": [.9661, .9559, .9223, .9719, 1], "SMTP": [.6923, .7984, .8013, .7993, .7763],
    "BankNote": [.5148, .4782, .4655, .4762, .5119], "Annthyroid": [.7796, .8040, .8094, .8004, .7707],
    "Mammography": [.7903, .7911, .7790, .7928, .8346], "Thyroid": [.9645, .8875, .8749, .8896, .9444],
    "Vertebral": [.3166, .4492, .4616, .4551, .4017], "Rice": [.5867, .5351, .5245, .5301, .5995],
    "Glass": [.7884, .7576, .7631, .7661, .7146], "Shuttle": [.7390, .8347, .7930, .8563, .9943],
    "Vowels": [.9806, .9536, .9574, .9572, .9264], "Pendigits": [.7925, .7110, .6754, .6776, .8588],
    "Lympho": [.9481, .9401, .9284, .9292, .9683], "WBC": [.9653, .8552, .8357, .8372, .9403],
}
T4 = {
    "Cluto": [1, 1, 1, 1, 1], "Motivation": [1, 1, 1, 1, 1], "Zelnik2": [1, 1, 1, 1, 1],
    "Zelnik4": [1, 1, 1, 1, 1], "Zelnik6": [1, 1, 1, 1, 1], "SMTP": [.0003, 1, 1, 1, 1],
    "BankNote": [.4897, .4628, .4584, .4833, 1], "Annthyroid": [.4211, .2440, .2452, .2396, .5],
    "Mammography": [.5854, .6667, .6667, .6667, .5], "Thyroid": [.4412, .1504, .1429, .1617, .5],
    "Vertebral": [.1255, .1298, .1358, .1338, .1266], "Rice": [.5280, .6, .6, .5901, .5854],
    "Glass": [.1429, 1, 1, .6111, .5], "Shuttle": [.75, 1, 1, 1, .9092],
    "Vowels": [1, 1, 1, 1, 1], "Pendigits": [.3333, 1, 1, 1, 1],
    "Lympho": [1, .3571, .3, .3, 1], "WBC": [.7273, 1, 1, 1, .5],
}
# Tables 7 (AUC) and 8 (Precision): comparison with fuzzy/rough methods (min-max data)
# columns: FuNaN I, FuNaN II, FuNaN III, FuDSR   (DSR not reported -> NaN)
T7 = {
    "Cluto": [.9819, .9848, .9105, .8813], "Motivation": [1, 1, .9955, 1],
    "Zelnik2": [.9568, .9324, .9678, .9954], "Zelnik4": [.9815, .9766, .9614, .9810],
    "Zelnik6": [.9372, .9169, .9749, 1], "SMTP": [.8375, .8386, .8497, .8315],
    "BankNote": [.4788, .4669, .4999, .5056], "Annthyroid": [.7488, .7535, .6940, .6981],
    "Mammography": [.8074, .8058, .7551, .7608], "Thyroid": [.9097, .9037, .9005, .9159],
    "Vertebral": [.4251, .4381, .4050, .3879], "Rice": [.5369, .5233, .5736, .6135],
    "Glass": [.7484, .7413, .7291, .7168], "Shuttle": [.8771, .8386, .9725, .9909],
    "Vowels": [.9548, .9578, .9140, .9244], "Pendigits": [.7166, .6784, .7369, .8576],
    "Lympho": [.9695, .9542, .9675, .9836], "WBC": [.8713, .8535, .9316, .9471],
}
T8 = {
    "Cluto": [1, 1, 1, 1], "Motivation": [1, 1, 1, 1], "Zelnik2": [1, 1, 1, 1],
    "Zelnik4": [1, 1, 1, 1], "Zelnik6": [1, 1, 1, 1], "SMTP": [1, 1, 1, 1],
    "BankNote": [.4615, .4566, .8471, 1], "Annthyroid": [.2438, .2438, .6465, .3455],
    "Mammography": [1, 1, 1, 1], "Thyroid": [.1733, .1652, .7704, .2692],
    "Vertebral": [.1282, .1290, .1279, .1261], "Rice": [.5714, .5714, 1, .6061],
    "Glass": [.5, .5, .2963, .5], "Shuttle": [.6447, .6327, .9511, .9444],
    "Vowels": [1, 1, .6122, .5833], "Pendigits": [1, 1, 1, 1],
    "Lympho": [1, 1, 1, 1], "WBC": [1, .5, .6320, .5],
}


def paper_table(normalization, metric):
    """Return the paper's values as a DataFrame (rows = datasets, cols = METHODS)."""
    if normalization == "zscore":
        src = T3 if metric == "AUC" else T4
        return pd.DataFrame(src, index=METHODS).T
    src = T7 if metric == "AUC" else T8
    df = pd.DataFrame(src, index=METHODS[1:]).T
    df.insert(0, "DSR", np.nan)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", help="folder with the .mat datasets (default: data)")
    ap.add_argument("--normalization", default="zscore", choices=["zscore", "minmax"],
                    help="zscore -> compare with Tables 3/4, minmax -> compare with Tables 7/8")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS), help="subset of datasets to run")
    ap.add_argument("--gammas", nargs="*", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--m", type=float, default=2.0, help="fuzzifier m")
    ap.add_argument("--block", type=int, default=1000, help="rows per block (lower = less memory)")
    ap.add_argument("--out", default="results", help="output folder for CSV files")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows, gamma_rows = [], []
    for name in args.datasets:
        mat = sio.loadmat(os.path.join(args.data_dir, DATASETS[name] + ".mat"))
        X, y = mat["Data"].astype(float), mat["y"].ravel().astype(int)  # y = 1 marks outliers
        t0 = time.time()
        res, _ = run_all(X, y, args.normalization, with_dsr=True, block=args.block,
                         gammas=tuple(args.gammas), m=args.m)
        dt = time.time() - t0
        print(f"{name:12s} n={len(y):6d} d={X.shape[1]:3d} lambda={res['_lambda']:3d} ({dt:6.1f}s)  "
              + "  ".join(f"{k}: {res[k]['AUC']:.4f}" for k in METHODS))
        for k in METHODS:
            rows.append(dict(Dataset=name, Method=k, AUC=res[k]["AUC"], Precision=res[k]["Precision"]))
        for g, a in res["FuNaN III"]["AUC_per_gamma"].items():
            gamma_rows.append(dict(Dataset=name, gamma=g, AUC=a,
                                   Precision=res["FuNaN III"]["Precision_per_gamma"][g]))

    ours = pd.DataFrame(rows)
    gam = pd.DataFrame(gamma_rows)
    tag = args.normalization
    for metric in ["AUC", "Precision"]:
        mine = ours.pivot(index="Dataset", columns="Method", values=metric).loc[args.datasets, METHODS]
        paper = paper_table(tag, metric).loc[args.datasets, METHODS]
        diff = mine - paper
        mine.loc["Mean"], paper.loc["Mean"], diff.loc["Mean"] = mine.mean(), paper.mean(), diff.mean()
        side = pd.concat({"ours": mine, "paper": paper, "diff": diff}, axis=1)
        side.round(4).to_csv(os.path.join(args.out, f"{metric.lower()}_{tag}.csv"))
        print(f"\n===== {metric} ({tag}) : ours / paper / ours - paper =====")
        print(side.round(4).to_string())
    gam.pivot(index="Dataset", columns="gamma", values="AUC").loc[args.datasets].round(4) \
        .to_csv(os.path.join(args.out, f"funan3_auc_per_gamma_{tag}.csv"))
    print(f"\nCSV files written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
