import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def elbow_anomaly_bundle(row_score,
                         blocks,
                         show_gaps=True,
                         loss_col="reconstruction_loss"):
    """
    Parameters
    ----------
    row_score : 1-D np.ndarray
    blocks    : list[pd.DataFrame] – original blocks (same row order)
    show_gaps : bool               – also plot Δ-error curve if True
    loss_col  : str                – name for loss column in df_anom

    Returns
    -------
    bundle       – [df_anom, df_all]
    elbow_score  – float
    elbow_idx    – int
    """
    # 1️⃣  elbow detection
    sorted_scores = np.sort(row_score)
    deltas        = np.diff(sorted_scores)
    elbow_idx     = np.argmax(deltas)
    elbow_score   = sorted_scores[elbow_idx]

    # 2️⃣  plots
    plt.figure(figsize=(7,4))
    plt.plot(sorted_scores, lw=2)
    plt.axvline(elbow_idx,  ls="--", color="tab:red",  label="elbow")
    plt.axhline(elbow_score,ls="--", color="tab:red")
    plt.title("Sorted Reconstruction Errors")
    plt.xlabel("Row index"); plt.ylabel("Error")
    plt.legend(); plt.tight_layout(); plt.show()

    if show_gaps:
        plt.figure(figsize=(7,3))
        plt.plot(deltas, lw=2)
        plt.axvline(elbow_idx, ls="--", color="tab:red")
        plt.title("Gap between consecutive sorted errors")
        plt.xlabel("Index (i → i+1)"); plt.ylabel("Δ error")
        plt.tight_layout(); plt.show()

    # 3️⃣  build DataFrames
    df_all  = pd.concat(blocks, axis=1).reset_index(drop=True)
    mask    = row_score > elbow_score
    df_anom = df_all.loc[mask].copy()
    df_anom[loss_col] = row_score[mask]

    print(f"Elbow @ {elbow_score:.6f} → {mask.sum():,} anomalies "
          f"({100*mask.mean():.2f} % of {len(row_score):,})")

    return [df_anom, df_all], elbow_score, elbow_idx



# after you have `row_score` and `blocks`
bundle, elbow_score, elbow_idx = elbow_anomaly_bundle(
        row_score=row_score,
        blocks=blocks,
        show_gaps=True)

df_anom, df_all = bundle   # unpack as needed

# --- saving happens OUTSIDE the function ---
df_anom.to_csv("elbow_anomalies.csv", index=False)
print("Anomaly rows written to elbow_anomalies.csv")
