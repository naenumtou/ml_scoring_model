
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.create_factors import _lag_cols
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = UserWarning)

# Helper functions
# Overall ODR
def _odr(
    data_group: pd.DataFrame,
    cons: list
) -> float:
    
    """
    ODR Calculation.

    Description:
        ODR Calculation in the sample level.

    Args:
        data_group (pd.DataFrame)   : Input dataframe.
        cons (list)                 : List of unique primary key from train/test split.

    Returns:
        Float: ODR from sample.

    Notes:
        - N/A.
    """    

    s = data_group.loc[cons]
    return s['y_sum'].sum() / s['y_count'].sum()

# Monthly ODR
def _monthly_odr(
    data_group: pd.DataFrame,
    cons: np.array
) -> pd.Series:

    """
    Monthly ODR Calculation.

    Description:
        ODR Calculation in the month interval level.

    Args:
        data_group (pd.DataFrame)   : Input dataframe.
        cons (list)                 : List of unique primary key from train/test split.

    Returns:
        pd.Series: A series of monthly ODR from sample.

    Notes:
        - N/A.
    """ 

    s = data_group.loc[cons]
    g = s.groupby('Monthkey')
    return g['y_sum'].sum() / g['y_count'].sum()

# Sort and flag
def prepare_dataframe(
    df: pd.DataFrame,
    id_col: str,
    period_col: str,
    default_col: str,
    default_flag: int
) -> pd.DataFrame:
    
    """
    Sort by primary key and period and flag the target for modeling.

    Description:
        Raw transaction must be sorted from historical to current.
        To create the flag of target in the model.

    Args:
        df (pd.DataFrame)   : Input dataframe.
        id_col (str)        : Primary key.
        period_col (str)    : Period key for sorting.
        default_col (str)   : Default column as the target for modeling.
        default_flag (int)  : Default value that greater than the value will be considered as default.

    Returns:
        pd.DataFrame: DataFrame with sorted and flagged the target.

    Notes:
        - N/A.
    """

    print("=== Processing ===\n[Sort and create default flag]")

    df = df.sort_values(by = [id_col, period_col]).copy()

    flag_label = ["X", "30", "60", "90"][default_flag - 1] if 1 <= default_flag <= 4 else None
    df[f"def{flag_label}"] = np.where(
        df[default_col].ge(default_flag),
        1 ,0
    )

    return df

# Forward performance windows
def forward_ever_default(
    df: pd.DataFrame,
    id_col: str,
    default_col: str,
    n_lags: int
) -> pd.DataFrame:
    
    """
    Create forward-1 until forward-n columns for target.
    Uses a single groupby().transform(lambda).

    Description:
        The n-lags of column features are created by primary key. 
        bal{w}          Account balance lagged w months.

    Args:
        df (pd.DataFrame)   : Input dataframe.
        id_col (str)        : Primary key.
        default_col (str)   : Default column that already flag (0, 1) for modeling.
        n_lags (int)        : Defined n-lags for forward performance windows creation.

    Returns:
        pd.DataFrame: DataFrame with forward performance columns and ever default flag appended.

    Notes:
        - N/A.
    """

    print("=== Processing ===\n[Forward performance windows and ever default]")

    # Forward performance windows
    grouped = df.groupby(id_col)[default_col]
    shifted = {
        f"{default_col}{i}": grouped.shift(-i).astype(np.float16)
        for i in range(1, n_lags + 1)
    }
    df = df.assign(**shifted)

    # Ever default flag
    cols = _lag_cols(default_col, n_lags)
    window = df[cols]
    df[f"ever_default_{n_lags}"] = window.eq(1).any(axis = 1).astype(np.uint8)

    return df

# Find valid contract
def find_valid_contract(
    df: pd.DataFrame,
    id_col: str,
    period_col: list[str],
    n: int = 12
) -> list:
    
    """
    Find the new originated contract.

    Description:
        The new originated contract is invaild for modeling.
        Those will be excluded for modeling sample.

    Args:
        df (pd.DataFrame)   : Input dataframe.
        id_col (str)        : Primary key.
        period_col (list)   : Period key for consider E.g., MOB or Date.
        n (int)             : Set 12 as default same as observation windows.

    Returns:
        List: List of valid contract.

    Notes:
        - N/A.
    """

    max_mob = df.groupby(id_col)[period_col[0]].transform('max')
    max_month = df.groupby(id_col)[period_col[1]].transform('max')
    valid_id = df.loc[(max_mob >= n) & (max_month >= n), id_col].unique().tolist()

    return valid_id

# Drop unused columns
def drop_cols(
    df: pd.DataFrame,
    default_col: str,
    n_lags: int
) -> pd.DataFrame:
    
    """
    Drop unused columns.

    Description:
        Preserving the memory by dropping unused/finished columns.

    Args:
        df (pd.DataFrame)   : Input dataframe.
        default_col (str)   : Default column that already flag (0, 1) for modeling.
        n_lags (int)        : Defined n-lags for forward performance windows dropping.

    Returns:
        pd.DataFrame: DataFrame with forward performance columns dropped.

    Notes:
        - N/A.
    """
    
    flag_cols_to_drop = [f'{default_col}{i}' for i in range(1, n_lags + 1)]
    df = df.drop(columns = flag_cols_to_drop)

    return df

# Plot waterfall
def plot_exclusion(
    log: list
) -> None:
    
    """
    Plot waterfall exclusion.

    Description:
        Plot count of waterfall exclusion on each criteria.

    Args:
        log (list): List of excluded counts.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A
    """

    df_plot = (
        pd.DataFrame(log, columns = ['Criteria', 'Before', 'After'])
        .set_index('Criteria')
    )

    colorY = '#ffd500' #Set color theme --> Yellow
    colorG = '#808080' #Set color theme --> Gray    
    colors = ['red'] * len(df_plot)
    colors[0] = colorG
    colors[-1] = colorY

    fig, ax = plt.subplots(figsize = (10, 6))
    ax.bar(df_plot.index, df_plot["Before"], color = colors)
    ax.bar(df_plot.index, df_plot["After"], color = 'white')
    ax.set_yticklabels([f"{int(x):,}" for x in ax.get_yticks()])
    ax.set_title("Waterfall exclusion")
    ax.set_xlabel("Criteria")
    ax.set_ylabel("Number of observation")
    ax.tick_params(axis = "x", rotation = 90)
    plt.tight_layout()

    return plt.show()

# Train/Test Split custom function
def modified_train_test(
    data: pd.DataFrame,
    id_col: str,
    default_col: str,
    period_col: str,
    test_size: float,
    threshold1: float,
    threshold2: float,
    max_iter: int = 1000,
    random_state: int = None,
    n_jobs: int = -1,
    batch_size: int = 50,        
    patience: int = 100                      
) -> tuple[pd.DataFrame, pd.DataFrame]:
        
    """
    Custom function for train/test split.

    Description:
        The problem with simple train/test split of behavioral score is contract.
        The contract should be either in training set or testing set NOT both datasets.
        Even it assumed different behaviour by different transaction, the same contract should be in single dataset.
        The simple train/test split function with 'stratify' on target cannot solve this issue.
        The custom function is made to solve this issue.
        By creating train/test datasets with the same risk proflie of overall and monthly levels.
        
    Args:
        data (pd.DataFrame) : Input dataframe.
        id_col (str)        : Primary key.
        period_col (str)    : Period key for consider (Date).
        default_col (str)   : Ever default column that already flag (0, 1) for modeling.
        test_size (float)   : The proportion of testing size compared to training size.
        threshold1 (float)  : The difference of ODR in overall level.
        threshold2 (float)  : The difference of monthly ODR.
        max_iter (int)      : The maximum iteration for spliting dataset.
        random_state (int)  : To reproduce the result.
        n_jobs (int)        : -1 is the algorithm to use all logical processors (cores and threads).
        batch_size (int)    : Parallelism hyperparameter defining the number of training samples
        patience (int)      : Early stopping when the difference of ODR is not improved 
                            100 means not improving for 100 iterations

    Returns:
        pd.DataFrame: Training dataset.
        pd.DataFrame: Testing dataset.

    Notes:
        - N/A.
    """

    print("=== Processing ===\n[Equal target train/test Split]")

    con_unique = data[id_col].unique().tolist()
    total_odr = data[default_col].sum() / data[default_col].count()
    monthly_odr = data.groupby(period_col)[default_col].sum() / data.groupby(period_col)[default_col].count()

    # Unique primary key
    con_stats = data.groupby(id_col).agg(
        y_sum = (default_col, 'sum'),
        y_count = (default_col, 'count')
    ) 
    # Unique primary key per month
    con_monthly = data.groupby([id_col, period_col]).agg(
        y_sum = (default_col, 'sum'),
        y_count = (default_col, 'count')
    )

    def compute_score(seed):
        con_train, con_test = train_test_split(con_unique, test_size = test_size, random_state = seed)

        train_odr, test_odr = _odr(con_stats, con_train), _odr(con_stats, con_test)
        train_monthly, test_monthly = _monthly_odr(con_monthly, con_train), _monthly_odr(con_monthly, con_test)

        d_total = max(
            abs(total_odr - train_odr),
            abs(total_odr - test_odr),
            abs(train_odr - test_odr)
        )
        d_monthly = pd.concat(
            [
                abs(monthly_odr - train_monthly),
                abs(monthly_odr - test_monthly),
                abs(train_monthly - test_monthly)        
            ],
            axis = 1
        ).max(axis = 1).max()

        return d_total + d_monthly, d_total, d_monthly, con_train, con_test

    # Main loop --> batch run and early stopping
    best_score, best_split = np.inf, None
    no_improve = 0
    base_seed  = random_state or 0

    for batch_start in range(0, max_iter, batch_size):
        seeds = range(base_seed + batch_start, base_seed + batch_start + batch_size)

        # Parallel batch run
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_score)(s) for s in seeds
        )

        for score, d_total, d_monthly, con_train, con_test in results:
            if d_total <= threshold1 and d_monthly <= threshold2:
                print(f'[INFO]: Done at iteration {batch_start + 1}')
                mask = data[id_col].isin(con_train)
                return data[mask], data[~mask]

            if score < best_score:
                best_score = score
                best_split = (con_train, con_test)
                no_improve = 0
            else:
                no_improve += 1

        # Early stopping
        if no_improve >= patience:
            print(f'[INFO]: Early stopping at iteration: {batch_start + batch_size} --> No improvement for {patience} rounds')
            break

    print(f'[INFO]: Total ODR Difference: {best_score:.4f}')
    mask = data[id_col].isin(best_split[0])

    return data[mask], data[~mask]

# Plot overall
def plot_overall(
    all_data: pd.DataFrame,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    default_col: str
) -> None:
  
    """
    Plot overall ODR Sample.

    Description:
        Plot overall defaualt rates from difference sample.

    Args:
        all_data (pd.DataFrame)     : Input total dataframe.
        train_data (pd.DataFrame)   : Input training dataframe.
        test_data (pd.DataFrame)    : Input testing dataframe.
        default_col (str)           : Ever default column that already flag (0, 1) for modeling.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    data = pd.concat(
        [
            all_data[default_col].rename('Total'),
            train_data[default_col].rename('Train'),
            test_data[default_col].rename('Test')
        ],
      axis = 1
    )

    colorY = '#ffd500' #Set color theme --> Yellow
    colorG = '#808080' #Set color theme --> Gray  

    fig, axs = plt.subplots(1, 3, figsize = (20, 6))
    axs = axs.ravel()
    fig.suptitle('Train/Test Split target ratio')
    for i, col in enumerate(data.columns):
        sizes = [
            data[col].sum(),
            data[col].count() - data[col].sum()
        ]
        axs[i].pie(
            sizes,
            explode = (0.1, 0),
            labels = ['Bad', 'Good'],
            colors = [colorY, colorG],
            autopct = '%1.2f%%',
            startangle = 90,
            labeldistance = 0.7
        )
        axs[i].axis('equal')
        axs[i].set_title(f'{col} dataset')

    return plt.show()

# Plot monthly
def plot_monthly(
    all_data: pd.DataFrame,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    default_col: str,
    period_col: str,
) -> None:
    
    """
    Plot monthly ODR Sample.

    Description:
        Plot monthly default rates from difference sample.

    Args:
        all_data (pd.DataFrame)     : Input total dataframe.
        train_data (pd.DataFrame)   : Input training dataframe.
        test_data (pd.DataFrame)    : Input testing dataframe.
        default_col (str)           : Ever default column that already flag (0, 1) for modeling.
        period_col (str)            : Period key for plotting ('AS_OF_DATE') --> For datatime lable.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    dates = pd.date_range(
        all_data[period_col].min(),
        all_data[period_col].max(),
        freq = 'ME'
    )
    data_list = [all_data, test_data, train_data]
    data_labels = ['Total dataset', 'Test dataset', 'Train dataset']

    colorY = '#ffd500' #Set color theme --> Yellow
    colorG = '#808080' #Set color theme --> Gray

    plt.figure(figsize = (10, 6))
    for data, label in zip(data_list, data_labels):
        if label == 'Train dataset':
            color = colorY
            linewidth = 4
            linestyle = '-'
        elif label == 'Test dataset':
            color = colorG
            linewidth = 4
            linestyle = '-'
        else:
            color = colorG
            linewidth = 2
            linestyle = '--'
        plt.plot(
            dates,
            data.groupby([period_col])[default_col].sum() / data.groupby([period_col])[default_col].count(),
            color = color,
            linewidth = linewidth,
            linestyle = linestyle,
            label = label
        )
    plt.gca().set_yticklabels([f'{y * 100:.2f}%' for y in plt.gca().get_yticks()])
    plt.title('Compared monthly ODR')
    plt.xlabel('Date')
    plt.ylabel('ODR')
    plt.legend(frameon = True, facecolor = 'white')

    return plt.show()
