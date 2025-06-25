def elbow_anomaly_bundle(row_score,
                         blocks,
                         show_gaps=True,
                         loss_col="reconstruction_loss",
                         slope_frac=0.20,          # ← NEW tunable parameter
                         baseline_perc=0.10):      # how much of the left tail defines “flat”
    """
    slope_frac     – set to 0.20 for “20 % departure”, 0.30 for 30 %, etc.
                     The elbow is the *first* index where Δ > (1+slope_frac)*baseline.
    baseline_perc  – fraction of the curve (from the left) used to estimate the
                     baseline slope. 0.10 ⇒ use the first 10 % of points.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # 1️⃣  sort and compute slope
    sorted_scores = np.sort(row_score)
    deltas        = np.diff(sorted_scores)

    # 2️⃣  estimate baseline slope from the flat left-hand section
    baseline_n    = max(1, int(len(deltas) * baseline_perc))
    baseline_slope= deltas[:baseline_n].mean()

    # 3️⃣  find first index that exceeds the baseline by slope_frac
    threshold     = (1 + slope_frac) * baseline_slope
    cand          = np.where(deltas > threshold)[0]
    elbow_idx     = cand[0] if len(cand) else np.argmax(deltas)   # fallback to max gap
    elbow_score   = sorted_scores[elbow_idx]

    # 4️⃣  plotting (unchanged except the label)
    plt.figure(figsize=(7,4))
    plt.plot(sorted_scores, lw=2)
    plt.axvline(elbow_idx,  ls="--", color="tab:red",
                label=f"elbow (>{slope_frac:.0%} jump)")
    plt.axhline(elbow_score, ls="--", color="tab:red")
    plt.title("Sorted Reconstruction Errors")
    plt.xlabel("Row index"); plt.ylabel("Error")
    plt.legend(); plt.tight_layout(); plt.show()

    if show_gaps:
        plt.figure(figsize=(7,3))
        plt.plot(deltas, lw=2)
        plt.axhline(threshold, ls="--", color="tab:green",
                    label=f"threshold = {threshold:.3g}")
        plt.axvline(elbow_idx, ls="--", color="tab:red")
        plt.title("Δ error between consecutive rows")
        plt.xlabel("Index (i→i+1)"); plt.ylabel("Δ error")
        plt.legend(); plt.tight_layout(); plt.show()

    # 5️⃣  build anomaly / full DataFrames
    df_all  = pd.concat(blocks, axis=1).reset_index(drop=True)
    mask    = row_score > elbow_score
    df_anom = df_all.loc[mask].copy()
    df_anom[loss_col] = row_score[mask]

    print(f"Elbow @ {elbow_score:.6f} (idx {elbow_idx}) "
          f"→ {mask.sum():,} anomalies ({100*mask.mean():.2f} %)")

    return [df_anom, df_all], elbow_score, elbow_idx
