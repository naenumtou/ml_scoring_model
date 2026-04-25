
import warnings
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from scipy import stats

warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = UserWarning)

# Helper functions
def _prob_to_score(
    prob: np.ndarray,
    pdo: int,
    base_score: int,
    base_odds: int
) -> np.ndarray:
    
    """
    To convert calibrated probability to score.   

    Description:
        Coverting calibrated probability of good to scorecard by PDO Formula.
        The key interest is logit function of log(odds of good) --> log(P(Good) / P(Bad)).
        Therefore, the PDO Formula is used (+) offset, meaning high score --> low risk.

    Args:
        prob (np.array)     : The trained model from best parameters.
        pdo (int)           : PDO is the number of scorecard points required to double the odds of being a Good customer.
                            25 means every 25-point increase in the score means the Good-to-Bad odds are doubled.
        base_score (int)    : The base score is the score assigned when the odds equal the base odds.
                            E.g., a customer with BASE_ODDS receives a score of 800.
        base_odds (int)     : Base odds represent the reference odds of being Good vs Bad at the base score.

    Returns:
        np.array: Calculated score from probability.

    Notes:
        - If the key interest is logit function to log(odds of bad) --> log(P(Bad) / P(Good)).
        - The PDO Formula will be adjusted to (-) as offset - factor * np.log(odds_good).
        - This is to remain the defintion of high score --> low risk.
    """

    eps = 1e-6 #To aviod zero divided
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    odds_good = (1 - prob + eps) / (prob + eps)
    score = offset + factor * np.log(odds_good)

    return score

# Cutpoint calculation
def _compute_cutpoints(
    scores: pd.Series,
    n_bins: int,
    method: str = 'normal',
) -> list:

    """
    Computation of scoring cut-points.

    Description:
        Ramdomly report credit score from given input.

    Args:
        scores (pd.Series)   : The input of calculated score.
        n_bins (int)         : The number of bins for score bands.
        method (str)         : The cut-points method. Default = 'normal' using normal distribution to define cut-points.

    Returns:
        list: List of cutting points.

    Notes:
        - The method can be defined as 'equal' that will be given equal range of scores.
        - The method can be defined as 'quantile' that will be considered by equal population of score bands.
    """

    lo = scores.min()
    hi = scores.max()

    if method == "equal":
        cutpoints = np.linspace(lo, hi, n_bins + 1)
    elif method == "quantile":
        quantiles = np.linspace(0, 100, n_bins + 1)
        cutpoints = np.percentile(scores, quantiles)
    else:
        # For normal distribution cut-points
        mean = scores.mean()
        std = scores.std()
        prob_cuts  = np.linspace(0, 1, n_bins + 1) #Probability cutpoints around 0.5
        cutpoints  = stats.norm.ppf(
            np.clip(prob_cuts, 0.001, 0.999),
            loc = mean,
            scale = std,
        )
        cutpoints[0] = lo
        cutpoints[-1] = hi

    cutpoints = np.unique(cutpoints.round(1)) #Avoid duplicate cutpoints
    
    return cutpoints.tolist()

# Find base odds
def find_base_odds(
    y: pd.Series
) -> int:
    
    """
    To find the base odds.   

    Description:
        Computing the actual odds from the actual data.

    Args:
        y (pd.Series): The input of target for scoring.

    Returns:
        int: Actual base odds.

    Notes:
        - N/A.
    """

    actual_default = y.mean()
    
    return int((1 - actual_default) / actual_default)

# Find PDO
def find_best_pdo(
    base_model: CatBoostClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    base_score: int = 600,
    base_odds: int,
    eps: float = 1e-6,
) -> int:

    """
    To find the best PDO.   

    Description:
        Computing the best PDO from actual data not assumed.

    Args:
        base_model (catboost.CatBoostClassifier)    : The trained model from best parameters.
        X (pd.DataFrame)                            : The input of features for scoring.
        y (pd.Series)                               : The input of target for scoring.
        base_score (int)                            : The base score is the score assigned when the odds equal the base odds.
        base_odds (int)                             : Base odds represent the reference odds of being Good vs Bad at the base score.
        eps (float)                                 : Minimal number to avoid divide by zero.

    Returns:
        int: The best PDO.

    Notes:
        - N/A.
    """

    prob_cal = base_model.predict_proba(X)[:, 1]
    pdo_range = np.linspace(10, 100, num = 10, dtype = int)

    results = []
    for pdo in pdo_range:
        factor = pdo / np.log(2)
        offset = base_score - factor * np.log(base_odds)
        odds_good  = (1 - prob_cal + eps) / (prob_cal + eps)
        scores_raw = offset + factor * np.log(odds_good)
        scores = np.clip(scores_raw, 300, 850)

        # Metrics
        pct_clipped = ((scores == 300) | (scores == 850)).mean() * 100
        ks_stat = None
        good_scores = scores[y == 0]
        bad_scores = scores[y == 1]
        ks_stat = stats.ks_2samp(good_scores, bad_scores).statistic
        results.append(
            {
                "pdo": pdo,
                "factor": round(factor, 2),
                "mean": round(scores.mean(), 1),
                "std": round(scores.std(), 1), #Higher more seperated
                "min_raw": round(scores_raw.min(), 1),
                "max_raw": round(scores_raw.max(), 1),
                "pct_clipped": round(pct_clipped, 1),
                "ks_statistic": round(ks_stat, 4) if ks_stat else None
            }
        )
    result_df = pd.DataFrame(results)

    # Best PDO
    # High std and low pct_clipped
    result_df["score_quality"] = (
        result_df["std"] / result_df["std"].max()
        - result_df["pct_clipped"] / 100
    )
    best_idx = result_df["score_quality"].idxmax() #Using maximum
    best_pdo = result_df.loc[best_idx, "pdo"]

    print(f"=== Result ===\nPDO: {best_pdo}")
    print(f"Standard Deviation: {result_df.loc[best_idx, 'std']:.2f}")
    print(f"Outbound: {result_df.loc[best_idx, 'pct_clipped']:.2f}%")
    
    return best_pdo

# Features score
def compute_feature_scores(
    base_model: CatBoostClassifier,
    X: pd.DataFrame,
    pdo: int,
    base_score: int,
    base_odds: int
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:

    """
    Function to compute any input datasets and compare output from model and features.   

    Description:
        Computing the scorecard from input features.
        Comparing the computed scorecard from features and model.

    Args:
        base_model (catboost.CatBoostClassifier)    : The trained model from best parameters.
        X (pd.DataFrame)                            : The input of features for scoring.
        y (pd.Series)                               : The input of target for scoring.
        pdo (int)                                   : PDO is the number of scorecard points required to double the odds.
        base_score (int)                            : The base score is the score assigned when the odds equal the base odds.
        base_odds (int)                             : Base odds represent the reference odds of being Good vs Bad at the base score.

    Returns:
        pd.DataFrame    : Calculated score of each feature with adjustment scalar with shape (n, n_features).
        pd.Series       : Calculated total score from features.
        pd.Series       : Calculated total score from model.

    Notes:
        - The outputs are for validating the results that two sources of calculation gives the same results.
        - It can be very small difference between those two scores.
    """

    print("=== Processing ===\n[Score construction]")
    print(f"Total input: {X.shape[0]}")

    explainer = shap.TreeExplainer(base_model)
    shap_vals = explainer.shap_values(X)
    base_value = explainer.expected_value
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    # Prediction probability from base model and calibrated model
    prob_raw = base_model.predict_proba(X)[:, 1]

    # Feature points from SHAP
    # Negative because CatBoost predict default --> SHAP (+) means high risk --> Score (-)
    feature_points = pd.DataFrame(
        -shap_vals * factor, #Negative SHAP --> high score = low risk
        columns = X.columns,
        index = X.index,
    )

    # Base points
    base_points = offset + factor * (-base_value)

    # Total score
    total_score_shap = (
        base_points + feature_points.sum(axis = 1)
    ).round()
    total_score_shap = np.clip(total_score_shap, 0, 850)

    total_score_model = pd.Series(
        _prob_to_score(prob_raw, pdo, base_score, base_odds).round(),
        index = X.index,
    )
    total_score_model = np.clip(total_score_model, 0, 850)

    feature_points.columns = ['score_' + str(c) for c in feature_points.columns]
    feature_points['score_base'] = base_points

    feature_points['score_from_shap'] = total_score_shap
    feature_points['score_from_model'] = total_score_model

    return feature_points

# Plot distribution
def plot_score_distribution(
    scores: pd.Series,
    target: pd.Series,
    good_label: int = 0,
    bad_label: int = 1,
    bins: int = 30,
    title: str = "Score Distribution",
) -> None:

    """
    Plot score distribution for good and bad using histogram overlay.

    Description:
        Plot dendrogram of hierarchical clustering to show features containing in each cluster.

    Args:
        scores (pd.Series)  : The computed score from the model.
        target (pd.Series)  : The actual target.
        good_label (int)    : The label to define good (0).
        bad_label (int)     : The label to define bad (1).
        bins (int)          : Number of bins for the plot
        title (str)         : The title for the plot

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    good_scores = scores[target == good_label]
    bad_scores  = scores[target == bad_label]

    fig, ax = plt.subplots(figsize = (10,6))

    ax.hist(
        good_scores,
        bins = bins,
        density = True,
        alpha = 0.7,
        color = "#2ca02c",
        label = "Good",
    )
    ax.hist(
        bad_scores,
        bins = bins,
        density = True,
        alpha = 0.7,
        color = "#d62728",
        label = "Bad",
    )

    # Mean lines
    ax.axvline(good_scores.mean(), color = "#2ca02c", linestyle = "--", linewidth = 1.5)
    ax.axvline(bad_scores.mean(),  color = "#d62728", linestyle = "--", linewidth = 1.5)
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    plt.gca().set_yticklabels([f'{y * 100:.2f}%' for y in plt.gca().get_yticks()])
    ax.legend()
    plt.tight_layout()
    
    return plt.show()

# Reporting
def sample_scorecard_report(
    X: pd.DataFrame,
    base_model: CatBoostClassifier,
    pdo: int,
    base_score: int,
    base_odds: int
) -> None:

    """
    Report credit score.

    Description:
        Ramdomly report credit score from given input.

    Args:
        X (pd.DataFrame)                            : The input of features for reporting scoring.
        base_model (catboost.CatBoostClassifier)    : The trained model from best parameters.
        pdo (int)                                   : PDO is the number of scorecard points required to double the odds.
        base_score (int)                            : The base score is the score assigned when the odds equal the base odds.
        base_odds (int)                             : Base odds represent the reference odds of being Good vs Bad at the base score.

    Returns:
        Print: Showing scoring report.

    Notes:
        - N/A.
    """
    
    X_sample = X.sample(n = 1)

    explainer  = shap.TreeExplainer(base_model)
    shap_vals = explainer.shap_values(X_sample)
    base_value = explainer.expected_value
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    # Score calculation
    base_points = offset + factor * (-base_value)
    feature_points = -shap_vals * factor # Negative because CatBoost predict default --> SHAP (+) means high risk --> Score (-)
    total_score_shap = base_points + np.sum(feature_points)

    # Reporting
    report = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "raw_value": X_sample.values.ravel().round(2),
            "feature_score": feature_points.ravel().round(2),
        }
    )

    # Print
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  Credit Score                  : {total_score_shap:.2f}")
    print(f"  Base Score                    : {base_points:.2f}")
    print(sep)
    print(report.to_string(index=False))
    print(sep)
    print(f"  Base + ΣFeature Scores       : {base_points:.2f} + {np.sum(feature_points):.2f}")
    print(sep)

# Score bands
def assign_score_bands(
    scores: pd.Series,
    bins: list = None,
    n_bins: int = 10,
    method: str = 'normal'
) -> pd.Series:
    
    """
    Assign score bands.

    Description:
        Assign score bands B1 is the best --> B(n) is the worst.

    Args:
        scores (pd.Series)   : The input of calculated score.
        bins (list)          : List of own designed cut-points. If None, using methodologies for calcualtion.
        n_bins (int)         : The number of bins for score bands.

    Returns:
        pd.Series: The series of bin labels (B1, B2, ...).

    Notes:
        - N/A.
    """

    if bins is not None:
        cutpoints = sorted(bins)
    else:
        cutpoints = _compute_cutpoints(scores, n_bins, method = method)
    
    # Create labels B1 (Best) → Bn (Worst)
    n = len(cutpoints) - 1
    labels = [f"B{i}" for i in range(1, n + 1)]
    result = pd.cut(
        scores,
        bins = cutpoints,
        labels = labels[::-1], #Inverse high score B1
        include_lowest = True,
    )

    return result
