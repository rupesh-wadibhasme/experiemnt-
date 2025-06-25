import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────
# Elbow detector with *relative-slope* threshold
#   • slope_frac    – how much steeper (e.g. 0.20 ⇒ +20 %) than the “flat”
#                     section a Δ-error must be to mark the elbow
#   • baseline_perc – first fraction of points used to define the flat slope
# ---------------------------------------------------------------------------
def elbow_anomaly_bundle(row_score,
                         blocks,
                         show_gaps=True,
                         loss_col="reconstruction_loss",
                         slope_frac=0.20,
                         baseline_perc=0.10):
    """
    Parameters
    ----------
    row_score     : 1-D array of reconstruction errors.
    blocks        : list[pd.DataFrame] – original blocks, still row-aligned.
    show_gaps     : bool – draw the Δ-error curve if True.
    loss_col      : str  – name of the error column in df_anom.
    slope_frac    : float (0–1) – elbow = first Δ exceeding
                    (1+slope_frac)*baseline_slope.
    baseline_perc : float (0–1) – left-hand share of the curve used
                    to estimate the baseline slope.

    Returns
    -------
    [df_anom, df_all] , elbow_score (float) , elbow_idx (int)
    """
    # 1️⃣  sort errors and compute consecutive differences
    sorted_scores = np.sort(row_score)
    deltas        = np.diff(sorted_scores)

    # 2️⃣  baseline slope from the flat left segment
    baseline_n    = max(1, int(len(deltas) * baseline_perc))
    baseline_slope= deltas[:baseline_n].mean()
    threshold     = (1.0 + slope_frac) * baseline_slope

    # 3️⃣  find the first Δ beyond threshold **after** the baseline zone
    search_idx    = np.arange(baseline_n, len(deltas))          # skip baseline part
    cand_idx      = search_idx[deltas[baseline_n:] > threshold]

    if cand_idx.size:                     # normal case
        elbow_idx = int(cand_idx[0])      # scalar
    else:                                 # fallback: largest overall jump
        elbow_idx = int(np.argmax(deltas))

    elbow_score = float(sorted_scores[elbow_idx])               # scalar

    # 4️⃣  diagnostic plots --------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(sorted_scores, lw=2)
    plt.axvline(elbow_idx, ls="--", color="tab:red",
                label=f"elbow (>{slope_frac:.0%} jump)")
    plt.axhline(elbow_score, ls="--", color="tab:red")
    plt.title("Sorted Reconstruction Errors")
    plt.xlabel("Row index"); plt.ylabel("Error")
    plt.legend(); plt.tight_layout(); plt.show()

    if show_gaps:
        plt.figure(figsize=(7, 3))
        plt.plot(deltas, lw=2)
        plt.axhline(threshold, ls="--", color="tab:green",
                    label=f"threshold = {threshold:.3g}")
        plt.axvline(elbow_idx, ls="--", color="tab:red")
        plt.title("Δ error between consecutive rows")
        plt.xlabel("Index (i→i+1)"); plt.ylabel("Δ error")
        plt.legend(); plt.tight_layout(); plt.show()

    # 5️⃣  build anomaly and full DataFrames --------------------------------
    df_all  = pd.concat(blocks, axis=1).reset_index(drop=True)
    mask    = row_score > elbow_score
    df_anom = df_all.loc[mask].copy()
    df_anom[loss_col] = row_score[mask]

    print(f"Elbow @ {elbow_score:.6f} (idx {elbow_idx}) ⇒ "
          f"{mask.sum():,} anomalies ({100*mask.mean():.2f} % of "
          f"{len(row_score):,})")

    return [df_anom, df_all], elbow_score, elbow_idx
