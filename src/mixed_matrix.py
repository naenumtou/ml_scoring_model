
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr, chi2_contingency

warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = UserWarning)

# Cramer's V for categorical vs categorical
def cramers_v(
    x: pd.Series,
    y: pd.Series
) -> float:
 
    """
    Cramer's V for categorical vs categorical

    Description:
        Cramer's V is a measure of association between two categorical variables.
        It is based on Pearson's chi-squared statistic and was published by Harald Cramér in 1946.

    Args:
        x (pd.Series)   : Categorical features series.
        y (pd.Series)   : Target series.

    Returns:
        float: Cramér's V Statistic is computed by taking the square root of the chi-squared statistic
             divided by the sample size and the minimum dimension minus 1.

    Notes:
        - N/A.
    """

    confusion = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion, correction = False)[0]
    n = confusion.sum().sum()
    min_dim = min(confusion.shape) - 1

    if min_dim == 0 or n == 0:
        return 0.0
    
    return float(np.sqrt(chi2 / (n * min_dim)))

# Correlation ratio for categorical vs numerical
def correlation_ratio(
    cat: pd.Series,
    num: pd.Series
) -> float:
    
    """
    Correlation ratio for categorical vs numerical

    Description:
        The correlation ratio (eta), measures the strength of non-linear relationships between variables.
        The measure is defined as the ratio of two standard deviations representing these types of variation.

    Args:
        cat (pd.Series)   : Categorical features series.
        num (pd.Series)   : Numerical features series.

    Returns:
        float: The correlation ratio statistic is computed by taking the square root of the difference between values are 
             the weighted sum of the squares of the differences between the subject averages and the overall average
             divided by the overall sum of squares of the differences from the overall average.

    Notes:
        - N/A.
    """

    grand_mean = num.mean()
    groups = [num[cat == v].values for v in cat.unique()]
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups if len(g) > 0)
    ss_total = ((num - grand_mean) ** 2).sum()
    
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0

# Building correlation matrix for mixed data
def build_mixed_correlation(
    X: pd.DataFrame,
    num_cols: list,
    cat_cols: list,
) -> pd.DataFrame:
        
    """
    Building correlation matrix for mixed data

    Description:
        Numerical vs Numerical --> Spearman correlation (No linear assumed)
        Categorical vs Categorical --> Cramer's V
        Numerical vs Categorical --> Correlation ratio

    Args:
        X (pd.DataFrame)    : Training data without transformation.
        num_cols (list)     : List of numerical features.
        cat_cols (list)     : List of categorical features.

    Returns:
        pd.DataFrame: The mixed data correlation matrix with n*n shape.

    Notes:
        - N/A.
    """

    print("=== Processing ===\n[Correlation matrix for mixed data]")

    cols   = num_cols + cat_cols
    n      = len(cols)
    matrix = np.zeros((n, n))
    col_idx = {c: i for i, c in enumerate(cols)}
    
    # Numerical vs Numerical --> Spearman correlation
    if len(num_cols) >= 2:
        corr_arr, _ = spearmanr(X[num_cols])
        if len(num_cols) == 2:
            corr_arr = np.array([[1.0, corr_arr], [corr_arr, 1.0]])
        for i, c1 in enumerate(num_cols):
            for j, c2 in enumerate(num_cols):
                matrix[col_idx[c1], col_idx[c2]] = abs(corr_arr[i, j])
    
    # Categorical vs Categorical --> Cramer's V
    for c1 in cat_cols:
        for c2 in cat_cols:
            val = 1.0 if c1 == c2 else cramers_v(X[c1], X[c2])
            matrix[col_idx[c1], col_idx[c2]] = val
    
    # Numerical vs Categorical --> Correlation ratio
    for num in num_cols:
        for cat in cat_cols:
            val = correlation_ratio(X[cat], X[num])
            matrix[col_idx[num], col_idx[cat]] = val
            matrix[col_idx[cat], col_idx[num]] = val
    
    # Fill diagonal with 1
    np.fill_diagonal(matrix, 1.0)
    matrix = pd.DataFrame(matrix, index = cols, columns = cols)
    
    print(f"Correlation matrix shape: {matrix.shape}")
    
    return matrix

# Plot correlation matrix
def plot_matrix(
    matrix: pd.DataFrame
) -> None:
    
    """
    Plot correlation matrix

    Description:
        Plot correlation mixed data matrix result.

    Args:
        matrix (pd.DataFrame): Correlation matrix with n*n shape.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    plt.figure(figsize = (21, 10))
    plt.title(f'Correlation matrix')
    ax = sns.heatmap(
        matrix,
        annot = False,
        xticklabels = True,
        yticklabels = True,
        cmap = 'RdYlGn_r',
        cbar = False
    )
    ax.xaxis.tick_top() # x-axis on top
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation = 90)
    return plt.show()
