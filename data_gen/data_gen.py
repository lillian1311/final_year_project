"""
SA-6-style synthetic ordinal risk dataset generator (merged, line-by-line commented).

THE IDEA IN FIVE LINES
    1. Every person has a hidden security attitude, theta.
    2. The six survey items are noisy readings of theta (what a model gets to see).
    3. The person's TRUE risk is a separate noisy function of theta (the label).
    4. The label is NOT computed from the items, so predicting it is a genuinely
       noisy problem rather than "add up the items and threshold".
    5. Higher SA-6 attitude = more protective = LOWER risk, so the sign is flipped.

SOURCES FOR THE NUMBERS
    Loadings (0.81 ... 0.71) and alpha (~0.84-0.89): Faklaris, Dabbish & Hong (2019).
    Total score mean/SD (21.53 / 4.67): a secondary study using SA-6 (illustrative).
"""

import json                              # writes the metadata file (human-readable settings record)
from pathlib import Path                 # builds folder/file paths in a way that works on any OS

import numpy as np                       # fast array maths (the workhorse for the data)
import pandas as pd                      # tables (DataFrames) and CSV reading/writing
import tensorflow as tf                  # TensorFlow; TFP below is built on top of it
import tensorflow_probability as tfp     # probability distributions we sample random numbers from

tfd = tfp.distributions                  # short alias so we can write tfd.Normal instead of tfp.distributions.Normal

# --- Fixed settings taken from the SA-6 literature ---------------------------
LOADINGS = np.array([0.81, 0.78, 0.77, 0.72, 0.71, 0.71], dtype=np.float32)
# ^ how strongly each of the 6 items tracks the hidden attitude (from the paper's factor analysis)

TARGET_ALPHA = 0.86
# ^ the Cronbach's alpha (scale reliability) we want our fake data to reach; real SA-6 studies report ~0.84-0.89

CATEGORY_PROPORTIONS = np.array([0.06, 0.12, 0.22, 0.35, 0.25])
# ^ share of people answering 1,2,3,4,5 on each item; leans toward "agree" like real attitude data.
#   These sum to 1.0 and give a total score close to the reference mean of 21.5

REFERENCE_TOTAL_MEAN = 21.53             # real-world mean of the 6-30 total score (used only for a realism check)
REFERENCE_TOTAL_SD = 4.67                # real-world SD of the total score (also only for the check)

CUTPOINTS = tfd.Normal(0.0, 1.0).quantile(np.cumsum(CATEGORY_PROPORTIONS)[:-1]).numpy()
# ^ turns the proportions above into 4 cut points on a standard-normal scale.
#   cumsum -> [0.06, 0.18, 0.40, 0.75, 1.0]; [:-1] drops the final 1.0; quantile() finds the
#   normal-curve position below which that share of people fall. A continuous score is later
#   sliced at these 4 points into 5 ordered categories.


# =============================================================================
# STEP 1: generate the six Likert items from a hidden trait
# =============================================================================
def _draw_items(n_people, loadings, noise_scale, seed):
    """Return (theta, likert) where likert is an (n_people, 6) array of answers 1-5."""
    std_normal = tfd.Normal(0.0, 1.0)                                  # a standard bell curve to draw random numbers from

    theta = std_normal.sample(n_people, seed=seed).numpy()             # one hidden attitude per person, mean 0 SD 1
    eps = std_normal.sample((n_people, len(loadings)), seed=seed + 1).numpy()
    # ^ one independent random "item-specific noise" number per person per item (a different seed
    #   from theta so the two streams of randomness are independent)

    error_sd = np.sqrt(1.0 - loadings ** 2) * noise_scale
    # ^ how big each item's private noise is. With loading L the leftover variance is 1 - L^2, so
    #   sqrt(1 - L^2) is its SD. noise_scale is a global dial: >1 = noisier items, <1 = cleaner items

    raw = theta[:, None] * loadings[None, :] + eps * error_sd[None, :]
    # ^ THE CORE EQUATION: item = loading * hidden attitude + private noise.
    #   theta[:, None] makes theta a column and loadings[None, :] a row so numpy multiplies every
    #   person by every item (broadcasting).

    spread = np.sqrt(loadings ** 2 + error_sd ** 2)                    # the total SD of each raw item (signal + noise combined)
    z = raw / spread[None, :]
    # ^ rescale each item back to SD = 1. This keeps the share of people choosing each answer fixed
    #   no matter how noisy we make the items, so the noise dial changes reliability ONLY

    likert = np.digitize(z, CUTPOINTS) + 1
    # ^ digitize counts how many cut points each value exceeds (0-4); +1 turns that into the answers 1-5
    return theta, likert


def _cronbachs_alpha(items):
    """Scale reliability: near 1 means the items agree strongly with each other."""
    items = items.astype(float)                                        # make sure maths is done on decimals, not integers
    k = items.shape[1]                                                 # number of items (6)
    item_vars = items.var(axis=0, ddof=1).sum()                        # sum of each item's own variance
    total_var = items.sum(axis=1).var(ddof=1)                          # variance of the summed total score
    return (k / (k - 1)) * (1.0 - item_vars / total_var)               # the standard alpha formula


def _calibrate_noise_scale(n_people, loadings, target_alpha, seed, tol=0.004, max_iter=25):
    """Binary-search the noise dial until the generated data hits the target alpha."""
    lo, hi = 0.05, 4.0                                                 # smallest and largest noise dial values we will try
    scale, alpha = 1.0, float("nan")                                   # placeholders so the names exist even if the loop breaks early
    for _ in range(max_iter):                                          # up to 25 attempts, halving the search range each time
        scale = (lo + hi) / 2.0                                        # try the middle of the current range
        _, likert = _draw_items(n_people, loadings, scale, seed)       # generate data at this noise level (same seed every time = fair comparison)
        alpha = _cronbachs_alpha(likert)                               # measure the reliability we got
        if abs(alpha - target_alpha) < tol:                            # close enough to the target -> stop
            break
        if alpha < target_alpha:                                       # too unreliable = too noisy
            hi = scale                                                 # so search lower noise values
        else:                                                          # too reliable = too clean
            lo = scale                                                 # so search higher noise values
    return scale, alpha


# =============================================================================
# STEP 2: generate the TRUE risk label, independent of the observed items
# =============================================================================
def _draw_risk_latent(theta, label_corr, seed):
    """True underlying risk: correlated with theta (negatively) but with its own noise."""
    eta = tfd.Normal(0.0, 1.0).sample(len(theta), seed=seed).numpy()   # noise that belongs only to the label, not to any item
    mixed = label_corr * theta + np.sqrt(1.0 - label_corr ** 2) * eta
    # ^ blend theta with fresh noise. label_corr = 1.0 would make risk a perfect copy of theta;
    #   0.85 means attitude explains roughly 72% (0.85^2) of the variation in true risk.
    #   The sqrt(...) factor keeps the result's variance at exactly 1.
    return -mixed                                                      # flip sign: HIGH protective attitude -> LOW risk


def assign_risk_class(risk_latent, method="tertile"):
    """Cut the continuous true risk into 3 ordered classes (0 = low, 1 = medium, 2 = high)."""
    if method == "tertile":                                            # option A: equal thirds -> BALANCED classes
        q_low, q_high = np.percentile(risk_latent, [33.33, 66.67])     # values below which 1/3 and 2/3 of people fall
    elif method == "mean_sd":                                          # option B: mean +/- 1 SD -> IMBALANCED classes (~16/68/16)
        q_low = risk_latent.mean() - risk_latent.std(ddof=1)           # one SD below the average
        q_high = risk_latent.mean() + risk_latent.std(ddof=1)          # one SD above the average
    else:
        raise ValueError("method must be 'tertile' or 'mean_sd'")      # stop with a clear message on a typo
    risk_class = np.digitize(risk_latent, [q_low, q_high])             # 0 if below q_low, 1 if between, 2 if above q_high
    labels = np.array(["low_risk", "medium_risk", "high_risk"])[risk_class]   # human-readable name for each class
    return risk_class, labels, (float(q_low), float(q_high))


# =============================================================================
# STEP 3: stratified train / validation / test split
# =============================================================================
def _stratified_split(labels, train_frac, val_frac, rng):
    """Split row indices so each split keeps the same class proportions."""
    train_idx, val_idx, test_idx = [], [], []                          # three empty lists to fill
    for cls in np.unique(labels):                                      # handle each risk class separately
        idx = np.where(labels == cls)[0]                               # row numbers belonging to this class
        rng.shuffle(idx)                                               # shuffle them so the split is random
        n_tr = int(train_frac * len(idx))                              # how many go to training
        n_va = int(val_frac * len(idx))                                # how many go to validation
        train_idx.append(idx[:n_tr])                                   # first chunk -> train
        val_idx.append(idx[n_tr:n_tr + n_va])                          # next chunk -> validation
        test_idx.append(idx[n_tr + n_va:])                             # remainder -> test
    out = [np.concatenate(p) for p in (train_idx, val_idx, test_idx)]  # join the per-class pieces
    for arr in out:                                                    # mix the classes back together
        rng.shuffle(arr)                                               # so rows are not sorted by class
    return out


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def generate_sa6_dataset(seed=42, n_samples=3000, class_method="tertile",
                         label_corr=0.85, target_alpha=TARGET_ALPHA,
                         loadings=None, train_frac=0.6, val_frac=0.2):
    """
    Returns train_df, val_df, test_df, truth_df, metadata.

    seed          : master seed; the same seed always gives the same dataset
    n_samples     : number of fake people
    class_method  : 'tertile' (balanced) or 'mean_sd' (imbalanced)
    label_corr    : how strongly true risk follows the hidden attitude (lower = noisier, harder problem)
    target_alpha  : reliability the items should reach
    """
    loadings = LOADINGS if loadings is None else np.asarray(loadings, dtype=np.float32)   # use paper values unless overridden
    rng_split = np.random.default_rng(seed + 4)                        # its own random stream for splitting, independent of the data

    noise_scale, achieved_alpha = _calibrate_noise_scale(              # find the noise dial that hits the target alpha
        n_samples, loadings, target_alpha, seed)
    theta, likert = _draw_items(n_samples, loadings, noise_scale, seed)   # generate the final items with that dial setting

    risk_latent = _draw_risk_latent(theta, label_corr, seed + 2)       # true risk, with its own noise (seed+2 keeps it independent)
    risk_class, risk_label, class_cuts = assign_risk_class(risk_latent, class_method)   # cut it into 3 ordered classes

    df = pd.DataFrame(likert, columns=[f"SA6_{i + 1}" for i in range(likert.shape[1])])   # table of answers, columns SA6_1..SA6_6
    df.insert(0, "participant_id", np.arange(1, n_samples + 1))        # ID column at the front, starting from 1
    df["total_score"] = likert.sum(axis=1)                             # convenience column: sum of items (range 6-30, like real SA-6)
    df["risk_class"] = risk_class                                      # numeric label 0/1/2 (what models predict)
    df["risk_label"] = risk_label                                      # same label as text

    truth_df = pd.DataFrame({                                          # hidden ground truth, kept in a SEPARATE table so it cannot leak
        "participant_id": df["participant_id"],                        # key to join back to the main table if needed
        "theta_true": theta,                                           # the real hidden attitude
        "risk_latent_true": risk_latent,                               # the real continuous risk before cutting into classes
    })

    tr, va, te = _stratified_split(risk_class, train_frac, val_frac, rng_split)   # row numbers for each split
    train_df = df.iloc[tr].reset_index(drop=True)                      # pick those rows, renumber from 0
    val_df = df.iloc[va].reset_index(drop=True)
    test_df = df.iloc[te].reset_index(drop=True)

    metadata = {                                                       # a full record so anyone can reproduce or audit the dataset
        "seed": seed, "n_samples": n_samples, "class_method": class_method,
        "label_corr": label_corr, "loadings": loadings.tolist(),
        "target_alpha": target_alpha, "achieved_alpha": float(achieved_alpha),
        "noise_scale": float(noise_scale),
        "category_proportions_target": CATEGORY_PROPORTIONS.tolist(),
        "item_cutpoints": CUTPOINTS.tolist(), "class_cutpoints": class_cuts,
        "class_balance": {k: float(v) for k, v in
                          df["risk_label"].value_counts(normalize=True).items()},
        "total_score_mean": float(df["total_score"].mean()),
        "total_score_sd": float(df["total_score"].std()),
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
    }
    return train_df, val_df, test_df, truth_df, metadata


# =============================================================================
# VALIDATION: does the dataset behave the way we claim?
# =============================================================================
def validate_dataset(train_df, val_df, test_df, truth_df, metadata):
    """Print checks you can quote in your methodology chapter."""
    full = pd.concat([train_df, val_df, test_df])                      # stack all splits to check the whole dataset at once
    item_cols = [c for c in full.columns if c.startswith("SA6_")]      # names of the six item columns

    print(f"[1] Reliability:    alpha = {metadata['achieved_alpha']:.3f} (target {metadata['target_alpha']})")
    print(f"[2] Total score:    mean {metadata['total_score_mean']:.2f}, SD {metadata['total_score_sd']:.2f}"
          f"   (reference {REFERENCE_TOTAL_MEAN} / {REFERENCE_TOTAL_SD})")
    props = np.round([(full[c] == k).mean() for c in item_cols[:1] for k in range(1, 6)], 2)
    print(f"[3] Item 1 answer shares 1..5: {props.tolist()}   (target {CATEGORY_PROPORTIONS.tolist()})")
    print(f"[4] Class balance:  { {k: round(v, 3) for k, v in metadata['class_balance'].items()} }")

    ceiling = (full.groupby("total_score")["risk_class"]
                   .agg(lambda s: s.value_counts().iloc[0]).sum()) / len(full)
    # ^ best possible accuracy if you knew ONLY the total score and always guessed its most common class.
    #   If this were ~1.0 the problem would be trivial; a clearly lower number means real irreducible noise.
    print(f"[5] Accuracy ceiling from total score alone: {ceiling:.3f}   (should be clearly below 1.0)")

    merged = full.merge(truth_df, on="participant_id")                 # attach the hidden truth to check relationships
    print(f"[6] corr(theta_true, total_score)  = {merged['theta_true'].corr(merged['total_score']):.3f}"
          "   (positive: attitude drives the answers)")
    print(f"[7] corr(theta_true, risk_class)   = {merged['theta_true'].corr(merged['risk_class']):.3f}"
          "   (negative: higher attitude -> lower risk)")
    return ceiling


# =============================================================================
# SAVING
# =============================================================================
def save_sa6_dataset(train_df, val_df, test_df, truth_df, metadata, out_dir="data"):
    out = Path(out_dir)                                                # folder to write into
    out.mkdir(parents=True, exist_ok=True)                             # create it if it does not exist yet
    train_df.to_csv(out / "train.csv", index=False)                    # training split (index=False = no extra row-number column)
    val_df.to_csv(out / "val.csv", index=False)                        # validation split (for tuning)
    test_df.to_csv(out / "test.csv", index=False)                      # test split (touch once, at the end)
    truth_df.to_csv(out / "truth_DO_NOT_TRAIN_ON.csv", index=False)    # hidden ground truth, only for validating results
    with open(out / "metadata.json", "w") as f:                        # open a text file for writing
        json.dump(metadata, f, indent=2)                               # write all settings in readable, indented form


# =============================================================================
# RUN: build one balanced and one imbalanced version
# =============================================================================
if __name__ == "__main__":                                             # only runs when you execute this file directly
    for method, folder in [("tertile", "data/balanced"), ("mean_sd", "data/imbalanced")]:
        print(f"\n===== {method} =====")
        result = generate_sa6_dataset(seed=42, n_samples=3000, class_method=method)
        validate_dataset(*result)                                      # print the quality checks
        save_sa6_dataset(*result, out_dir=folder)                      # write the files