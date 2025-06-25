import numpy as np
import matplotlib.pyplot as plt

def plot_error_with_quantile_lines(row_score,
                                   quantiles=(0.01, 0.05, 0.10),
                                   line_kw=None,
                                   figsize=(7, 4)):
    """
    Draw the sorted-error curve and add vertical dotted lines that mark
    the top-q fraction of highest-error rows.  Prints how many rows fall
    in each tail.

    Parameters
    ----------
    row_score : 1-D array-like
    quantiles : tuple of floats (0-1)   – fractions to flag (default 1 %, 5 %, 10 %)
    line_kw   : dict  – extra kwargs forwarded to plt.axvline
    figsize   : tuple – size of the figure
    """
    if line_kw is None:
        line_kw = dict(ls=":", lw=1.5, color="tab:red")

    # sort ascending so largest errors are on the right
    sorted_scores = np.sort(row_score)
    N             = len(sorted_scores)

    plt.figure(figsize=figsize)
    plt.plot(sorted_scores, lw=2)
    plt.title("Sorted Reconstruction Errors")
    plt.xlabel("Row index (after sorting ↑)")
    plt.ylabel("Error")

    # print header
    print("Tail summary:")
    print(f"{'Quantile':>9} | {'Rows':>6}")

    # add one dotted line per quantile
    for q in sorted(quantiles):
        idx   = int(np.ceil((1 - q) * N)) - 1          # right-hand start index
        nrows = N - idx                                # how many rows in the tail

        # vertical line + text label on plot
        plt.axvline(idx, **line_kw)
        plt.text(idx, sorted_scores[0],
                 f" {int(q*100)}% ({nrows})", rotation=90,
                 va="bottom", ha="right", fontsize=8)

        # console print-out
        print(f"{q:>7.0%} | {nrows:6}")

    plt.tight_layout()
    plt.show()
