
import warnings
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = UserWarning)

# Score bands summary
def bin_summary(
    scores: pd.Series,
    bin_labels: pd.Series,
    y_true: pd.Series
) -> pd.DataFrame:
    
    """
    Summary table for score bands.

    Description:
        Summary table for assigned score bands.

    Args:
        scores (pd.Series)      : The input of calculated score.
        bin_labels (pd.Series)  : Output from assign_score_bands().
        y_true (pd.Series)      : The actual target.

    Returns:
        pd.DataFrame: The summary table for statistical tests.

    Notes:
        - N/A.
    """

    df = pd.DataFrame(
        {
            "score": scores,
            "bin": bin_labels,
            'default': y_true.values
        }
    )
    agg = {
        "n": ("score", "count"),
        "pct": ("score", lambda x: round(len(x) / len(scores), 2)),
        "min": ("score", "min"),
        "max": ("score", "max"),
        "mean": ("score", lambda x: round(x.mean(), 2)),
        "bad": ("default", "sum"),
        "odr": ("default", 'mean')
    }
    summary = (
        df.groupby("bin", observed = True)
        .agg(**agg)
        .reset_index()
        .sort_values("odr", ascending = False)
    )
    summary['good'] = summary['n'] - summary['bad']
    summary['cum_bad'] = summary['bad'].cumsum() / summary['bad'].sum()
    summary['cum_good'] = summary['good'].cumsum() / summary['good'].sum()
    summary['roc'] = (summary['cum_good'] - summary['cum_good'].shift(1, fill_value = 0)) * \
                    (summary['cum_bad'] + summary['cum_bad'].shift(1, fill_value = 0)) * 0.5
    summary['ks'] = abs(summary['cum_good'] - summary['cum_bad'])

    return summary

# Plot ROC Curve
def plot_roc(
    cum_good: pd.Series,
    cum_bad: pd.Series,
) -> None:
    
    """
    Plot ROC Curve.

    Description:
        Plot Receiver Operating Characteristic.
        ROC is a graphical plot illustrating the performance of a binary classification model at threshold.

    Args:
        cum_good (pd.Series)    : Cumulative of good distribution.
        cum_bad (pd.Series)     : Cumulative of bad distribution.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    roc = (cum_good - cum_good.shift(1, fill_value = 0)) * (cum_bad + cum_bad.shift(1, fill_value = 0)) * 0.5
    auc = roc.sum()

    plt.figure(figsize = (10, 6))
    colorY = '#ffd500' #Set color theme --> Yellow
    colorG = '#808080' #Set color theme --> Gray    
    plt.plot(
        np.hstack((0, cum_good)),
        np.hstack((0, cum_bad)),
        color = colorY,
        linewidth = 2
    )
    plt.plot([0, 1], [0, 1], c = colorG, linestyle = '--', linewidth = 2)
    plt.plot([], [], ' ', label = f"AUC: {auc * 100:.2f}%")
    plt.plot([], [], ' ', label = f"GINI: {(2 * auc - 1) * 100:.2f}%")
    plt.gca().set_yticklabels([f'{y * 100:.2f}%' for y in plt.gca().get_yticks()])
    plt.gca().set_xticklabels([f'{y * 100:.2f}%' for y in plt.gca().get_xticks()])
    plt.title('ROC Curve')
    plt.xlabel('Percentage of non-defaults')
    plt.ylabel('Percentage of defaults')
    plt.legend(frameon = True, facecolor = 'white')
    plt.tight_layout()

    return plt.show()

# Plot KS
def plot_ks(
    cum_good: pd.Series,
    cum_bad: pd.Series,
) -> None:
    
    """
    Plot KS Curve.

    Description:
        Plot Kolmogorov-Smirnov.
        KS measures the maximum separation between the cumulative distribution functions (CDFs) of two samples.

    Args:
        cum_good (pd.Series)    : Cumulative of good distribution.
        cum_bad (pd.Series)     : Cumulative of bad distribution.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    diff = (cum_good - cum_bad).abs()
    ks = diff.max()
    ks_idx = diff.idxmax()
    cg = cum_good.loc[ks_idx]
    cb = cum_bad.loc[ks_idx]

    plt.figure(figsize = (10, 6))
    colorY = '#ffd500' #Set color theme --> Yellow
    colorG = '#808080' #Set color theme --> Gray
    plt.plot(cum_good, label = 'Cumulative good', color = colorY, linewidth = 2)
    plt.plot(cum_bad, label = 'Cumulative bad', color = colorG, linewidth = 2)
    plt.vlines(
        ks_idx,
        ymin = min(cg, cb),
        ymax = max(cg, cb),
        colors = "red",
        linestyles = "--",
        linewidth = 2,
    )
    plt.plot([], [], ' ', label = f"KS: {ks * 100:.2f}%")
    plt.gca().set_yticklabels([f'{y * 100:.2f}%' for y in plt.gca().get_yticks()])
    plt.gca().set_xticklabels([f'B{int(i + 1)}' for i in plt.gca().get_xticks()])
    plt.title('KS Curve')
    plt.xlabel('Score bands')
    plt.ylabel('Cumulative distribution')
    plt.legend(frameon = True, facecolor = 'white')
    plt.tight_layout()

    return plt.show()

# Plot monthly classification back-testing
def plot_classification_monthly(
    month: pd.Series,
    bin_labels: pd.Series,
    y_true: pd.Series,
) -> None:
    
    """
    Plot monthly classification back-testing.

    Description:
        Testing classification ability on historical monthly basis.

    Args:
        month (pd.Series)       : The input of month date as a key for calculation.
        bin_labels (pd.Series)  : Output from assign_score_bands().
        y_true (pd.Series)      : The actual target.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    df = pd.DataFrame(
        {
            "month": month,
            "bin": bin_labels,
            'default': y_true.values
        }
    )
    agg = {
        "n": ("default", "size"),
        "bad": ("default", "sum"),
        "odr": ("default", 'mean')
    }
    summary = (
        df.groupby(["month", "bin"], observed = True)
        .agg(**agg)
        .reset_index()
        .sort_values(["month", "odr"], ascending = [True , False])
    )
    summary['good'] = summary['n'] - summary['bad']
    summary["cum_bad"] = (
        summary.groupby("month")["bad"].cumsum() / summary.groupby("month")["bad"].transform("sum")
    )
    summary["cum_good"] = (
        summary.groupby("month")["good"].cumsum() / summary.groupby("month")["good"].transform("sum")
    )
    summary["roc"] = (
        (
            summary["cum_good"] - summary.groupby("month")["cum_good"].shift(1, fill_value = 0)

        ) * \
        (
            summary["cum_bad"] + summary.groupby("month")["cum_bad"].shift(1, fill_value = 0)
        ) * 0.5
    )
    summary["ks"] = abs(summary["cum_good"] - summary["cum_bad"])

    # Back-testing
    auc = summary.groupby("month")["roc"].sum()
    gini = 2 * auc - 1
    ks = summary.groupby("month")["ks"].max()

    # Plot
    fig, axs = plt.subplots(1, 3, figsize = (21, 4), sharex = True)
    fig.subplots_adjust(wspace = 0.2)
    plt.suptitle("Back-testing: Monthly classification", y = 1)
    axs = axs.ravel()

    axs[0].set_title('AUC')
    axs[0].plot(auc, color = 'royalblue', linewidth = 2)
    axs[0].margins(0) #Remove default margins
    axs[0].axhspan(0, 0.6, facecolor = '#C00000', alpha = 0.5)
    axs[0].axhspan(0.6, 0.7, facecolor = '#FFC000', alpha = 0.5)
    axs[0].axhspan(0.7, 1.0, facecolor = '#00B050', alpha = 0.5)
    axs[0].set_yticklabels([f"{y * 100:.2f}%" for y in axs[0].get_yticks()])
    axs[0].xaxis.set_major_locator(mdates.YearLocator())
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    axs[1].set_title('GINI')
    axs[1].plot(gini, color = 'royalblue', linewidth = 2)
    axs[1].margins(0) #Remove default margins
    axs[1].axhspan(0, 0.2, facecolor = '#C00000', alpha = 0.5)
    axs[1].axhspan(0.2, 0.4, facecolor = '#FFC000', alpha = 0.5)
    axs[1].axhspan(0.4, 1.0, facecolor = '#00B050', alpha = 0.5)
    axs[1].set_yticklabels([f"{y * 100:.2f}%" for y in axs[1].get_yticks()])
    axs[1].xaxis.set_major_locator(mdates.YearLocator())
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    axs[2].set_title('KS')
    axs[2].plot(ks, color = 'royalblue', linewidth = 2)
    axs[2].margins(0) #Remove default margins
    axs[2].axhspan(0, 0.2, facecolor = '#C00000', alpha = 0.5)
    axs[2].axhspan(0.2, 0.4, facecolor = '#FFC000', alpha = 0.5)
    axs[2].axhspan(0.4, 1.0, facecolor = '#00B050', alpha = 0.5)
    axs[2].set_yticklabels([f"{y * 100:.2f}%" for y in axs[2].get_yticks()])
    axs[2].xaxis.set_major_locator(mdates.YearLocator())
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    return plt.show()


# Plot monthly stability back-testing
def plot_stability_monthly(
    month: pd.Series,
    bin_labels: pd.Series,
    y_true: pd.Series,
) -> None:

    """
    Plot monthly stability back-testing.

    Description:
        Testing model stability on historical monthly basis.

    Args:
        month (pd.Series)       : The input of month date as a key for calculation.
        bin_labels (pd.Series)  : Output from assign_score_bands().
        y_true (pd.Series)      : The actual target.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """   

    df = pd.DataFrame(
        {
            "month": month,
            "bin": bin_labels,
            'default': y_true.values
        }
    )
    agg = {
        "n": ("default", "size")
    }
    dist = (
        df.groupby(["month", "bin"], observed = True)
        .agg(**agg)
        .reset_index()
        .assign(total = lambda x: x.groupby("month")["n"].transform("sum"))
        .assign(p = lambda x: x["n"] / x["total"])
        .assign(month_next = lambda x: x["month"] + pd.DateOffset(months=1))
        .sort_values(["month", "bin"], ascending = [True , False])
    )
    dist = pd.merge(
        dist,
        dist[["month", "bin", "p"]],
        how = "left",
        left_on = ["month_next", "bin"],
        right_on = ["month", "bin"],
        suffixes=("_t0", "_t1")
    )
    p0 = np.clip(dist["p_t0"], 1e-6, dist["p_t0"])
    p1 = np.clip(dist["p_t1"], 1e-6, dist["p_t1"])
    dist["psi"] = (p0 - p1) * np.log(p0 / p1)

    # Back-testing
    psi = dist.groupby("month_t0")["psi"].sum()
    portion = pd.pivot_table(
        data = dist,
        index = "month_t0",
        columns = "bin",
        values = "n",
        fill_value = 0
    )
    portion = portion[bin_labels.unique()].div(portion[bin_labels.unique()].sum(axis = 1), axis = 0)

    # Plot
    fig, axs = plt.subplots(1, 2, figsize = (14, 4), sharex = True)
    fig.subplots_adjust(wspace = 0.2)
    plt.suptitle("Back-testing: Monthly stability", y = 1)
    axs = axs.ravel()

    axs[0].set_title('PSI')
    axs[0].plot(psi, color = 'royalblue', linewidth = 2)
    axs[0].margins(0) #Remove default margins
    axs[0].axhspan(0, 0.1, facecolor = '#00B050', alpha = 0.5)
    axs[0].axhspan(0.1, 0.25, facecolor = '#FFC000', alpha = 0.5)
    axs[0].axhspan(0.25, 1.0, facecolor = '#C00000', alpha = 0.5)
    axs[0].set_yticklabels([f"{y * 100:.2f}%" for y in axs[0].get_yticks()])
    axs[0].xaxis.set_major_locator(mdates.YearLocator())
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    axs[1].set_title('Proportion')
    axs[1].stackplot(portion.index, portion.T, alpha = 0.7)
    axs[1].margins(0) #Remove default margins
    axs[1].set_yticklabels([f"{y * 100:.2f}%" for y in axs[1].get_yticks()])
    axs[1].xaxis.set_major_locator(mdates.YearLocator())
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    return plt.show()
