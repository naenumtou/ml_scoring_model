
import warnings
import numpy as np
import pandas as pd

from pandas import CategoricalDtype

warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = UserWarning)

# Helper functions
def _safe_div(
    numerator: pd.Series,
    denominator: pd.Series
) -> pd.Series:

    """
    Divide two Series and return 0 where denominator equal 0.

    Description:
        Safe divide where denominator is 0.

    Args:
        numerator (pd.Series)       : Series of numerator.
        denominatory (pd.Series)    : Series of denominatory.

    Returns:
        pd.Series: Divided values for a feature.

    Notes:
        - Forced to 0 to aviod any error cases.
    """

    return np.where(denominator == 0, 0, numerator / denominator)

def _lag_cols(
    base: str,
    n: int
) -> list[str]:

    """
    Lagging columns functions.

    Description:
        Lagging columns only used for the calculation.

    Args:
        base (str)  : Columns name for lagging.
        n (int)     : Window to lag the columns.

    Returns:
        List: List of lag column names.

    Notes:
        - N/A.
    """

    return [f"{base}{i}" for i in range(1, n + 1)]

def _consecutive_run_vectorised(
    df: pd.DataFrame,
    base: str,
    window: int,
    threshold: int
) -> np.ndarray:
    
    """
    Compute the longest consecutive months with condition.

    Description:
        Counting the longest consecutive months based on the condition.
        E.g., 'base{i} >= threshold' or 'base{i} == threshold'
        Fully vectorised with NumPy — avoids the slow row-by-row Python loop.

    Args:
        df (pd.DataFrame)   : Input dataframe.
        base (str)          : The input features that need to count.
        window (int)        : The observing windows.
        threshold (int)     : The value of interest.

    Returns:
        np.array: A 1-D numpy array of length len(df).

    Notes:
        - N/A.
    """
    
    cols = _lag_cols(base, window)

    # Shape: (n_rows, window)  — True where condition holds
    if threshold == 1:
        arr = df[cols].values >= threshold
    elif threshold == 2:
        arr = df[cols].values == threshold
    else:
        print("No target found. Recheck parameters.")

    n_rows, n_cols = arr.shape
    run = np.zeros(n_rows, dtype = np.int32)
    current = np.zeros(n_rows, dtype = np.int32)

    for col_idx in range(n_cols):
        current = np.where(arr[:, col_idx], current + 1, 0)
        run = np.maximum(run, current)

    return run

def _consecutive_long_vectorised(
    df: pd.DataFrame,
    base: str,
    window: int,
    threshold: int
) -> np.ndarray:
    
    """
    Compute the longest consecutive months.

    Description:
        Counting the longest consecutive months based on the condition.
        E.g., 'base{i} >= threshold'
        Fully vectorised with NumPy — avoids the slow row-by-row Python loop.

    Args:
        df (pd.DataFrame)   : Input dataframe.
        base (str)          : The input features that need to count.
        window (int)        : The observing windows.
        threshold (int)     : The value of interest.

    Returns:
        np.array: A 1-D numpy array of length len(df).

    Notes:
        - N/A.
    """
    
    cols = _lag_cols(base, window)

    # Shape: (n_rows, window)  — True where condition holds
    arr = df[cols].values >= threshold
    n_rows, n_cols = arr.shape
    run = np.zeros(n_rows, dtype = np.int32)
    current = np.zeros(n_rows, dtype = np.int32)

    for col_idx in range(n_cols):
        current = np.where(arr[:, col_idx], current + 1, 0)
        run = np.maximum(run, current)

    return run


# Rename and sort
def prepare_dataframe(
    df: pd.DataFrame,
    id_col: str,
    period_col: str,
    cols_rename: dict
) -> pd.DataFrame:
    
    """
    Sort by primary key and period and rename raw columns.

    Description:
        To prepare the raw dataframe ready for factors creation.
        Raw transaction must be sorted from historical to current. 

    Args:
        df (pd.DataFrame)   : Input dataframe.
        id_col (str)        : Primary key.
        period_col (str)    : Period key for sorting.
        cols_rename (dict)  : Columns rename matching.

    Returns:
        pd.DataFrame: DataFrame with renamed columns and sorted.

    Notes:
        - N/A.
    """

    df = df.sort_values(by = [id_col, period_col]).copy()

    rename_map = {k: v for k, v in cols_rename.items() if k in df.columns}
    df = df.rename(columns = rename_map)

    return df

# Lagging features
def create_lag_features(
    df: pd.DataFrame,
    id_col: str,
    cols_lag: list,
    n_lags: int
) -> pd.DataFrame:
    
    """
    Create lag-1 until lag-n columns for each feature.
    Uses a single groupby().transform(lambda) per feature.

    Description:
        The n-lags of column features are created by primary key. 
        'n_lags' is windows refering to the observation months.
        bal{n}          Account balance lagged n_lags months.
        pay{n}          Payment amount lagged n_lags months.
        pay_types{n}    Payment types lagged n_lags months.
        due{n}          Due amount lagged n_lags months.
        ovd{n}          Overdue amount lagged n_lags months.
        del{n}          Delinquency status lagged n_lags months.

    Args:
        df (pd.DataFrame)   : Input dataframe.
        id_col (str)        : Primary key.
        cols_lag (list)     : List of columns renamed matching.
        n_lags (int)        : Defined n-lags for factors creation.

    Returns:
        pd.DataFrame: DataFrame with lag columns appended.

    Notes:
        - N/A.
    """

    print("=== Processing ===\n[Lagging features creation]")

    for feat in cols_lag:
        if feat not in df.columns:
            print("No features found. Recheck parameters.")
            continue
        grouped = df.groupby(id_col)[feat]
        lag_dict = {f"{feat}{i}": grouped.shift(i) for i in range(1, n_lags + 1)}
        df = pd.concat([df, pd.DataFrame(lag_dict, index = df.index)], axis = 1)

    return df

# Balance features
def create_balance_features(
    df: pd.DataFrame,
    month_ranges: list[int],
    feature_col: str,
    init_col: str = None,
) -> pd.DataFrame:
        
    """
    Create account balance aspect features.

    Description:
        'w' and 's' are windows refering to the observation months.
        avg_bal_{w}                 Average balance over last w months.
        max_bal_{w}                 Maximum balance over last w months.
        min_bal_{w}                 Minimum balance over last w months.
        bal_to_avg_bal_{w}          Current balance to average balance over last w months.
        bal_to_max_bal_{w}          Current balance to maximum balance over last w months.
        bal_to_min_bal_{w}          Current balance to minimum balance over last w months.
        avg_bal_{w}_to_fin          Average balance over last w months to initial balance.
        max_bal_{w}_to_fin          Maximum balance over last w months to initial balance.
        min_bal_{w}_to_fin          Minimum balance over last w months to initial balance.
        bal_to_fin                  Current balance to initial balance.
        avg_bal_{s}_to_avg_bal_{w}  Average balance s month to average balance over last w months (s < w).
        avg_bal_{s}_to_max_bal_{w}  Average balance s month to maximum balance over last w months (s <= w).
        avg_bal_{s}_to_min_bal_{w}  Average balance s month to minimum balance over last w months (s <= w).
        
    Args:
        df (pd.DataFrame)           : Input dataframe.
        month_ranges (list)         : Observation months ranges.
        feature_col (str)           : The raw feature for processing.
        init_col (str, optional)    : Initial or original financial amount.
                                    If None, financial amount is not defined. Features are not created.

    Returns:
        pd.DataFrame: DataFrame with features balance related columns appended.

    Notes:
        - The financial amount is optional depending on data available.
    """

    print("=== Processing ===\n[Account balance features creation]")

    for w in month_ranges:
        cols = _lag_cols(feature_col, w)
        window = df[cols]
        df[f"avg_{feature_col}_{w}"] = window.sum(axis = 1) / w
        df[f"max_{feature_col}_{w}"] = window.max(axis = 1)
        df[f"min_{feature_col}_{w}"] = window.min(axis = 1)

    for w in month_ranges:
        df[f"{feature_col}_to_avg_{feature_col}_{w}"] = _safe_div(df[feature_col], df[f"avg_{feature_col}_{w}"])
        df[f"{feature_col}_to_max_{feature_col}_{w}"] = _safe_div(df[feature_col], df[f"max_{feature_col}_{w}"])
        df[f"{feature_col}_to_min_{feature_col}_{w}"] = _safe_div(df[feature_col], df[f"min_{feature_col}_{w}"])

        if init_col is None:
            continue
        else:
            df[f"avg_{feature_col}_{w}_to_{init_col}"] = _safe_div(df[f"avg_{feature_col}_{w}"], df[init_col])
            df[f"max_{feature_col}_{w}_to_{init_col}"] = _safe_div(df[f"max_{feature_col}_{w}"], df[init_col])
            df[f"min_{feature_col}_{w}_to_{init_col}"] = _safe_div(df[f"min_{feature_col}_{w}"], df[init_col])

    if init_col is None:
        pass
    else:
        df[f"{feature_col}_to_{init_col}"] = df[feature_col] / df[init_col]
    
    # Average balance to longer windows where s < w
    for w in month_ranges:
        for s in month_ranges:
            if s >= w:
                continue
            df[f"avg_{feature_col}_{s}_to_avg_{feature_col}_{w}"] = _safe_div(df[f"avg_{feature_col}_{s}"], df[f"avg_{feature_col}_{w}"])

    # Average balance to max or min longer windows where s <= w
    for w in month_ranges:
        for s in month_ranges:
            if s > w:
                continue
            df[f"avg_{feature_col}_{s}_to_max_{feature_col}_{w}"] = _safe_div(df[f"avg_{feature_col}_{s}"], df[f"max_{feature_col}_{w}"])
            df[f"avg_{feature_col}_{s}_to_min_{feature_col}_{w}"] = _safe_div(df[f"avg_{feature_col}_{s}"], df[f"min_{feature_col}_{w}"])

    return df

# Due and overdue features
def create_due_ovd_features(
    df: pd.DataFrame,
    month_ranges: list[int],
    feature_col: str,
    init_col: str = None,
) -> pd.DataFrame:
    
    """
    Create due and overdue amounts aspect features.

    Description:
        'w' and 's' are windows refering to the observation months.
        Due amount related features:
            avg_due_{w}                 Average due amount over last w months.
            max_due_{w}                 Maximum due amount over last w months.
            min_due_{w}                 Minimum due amount over last w months.
            due_to_avg_due_{w}          Current due amount to average due amount over last w months.
            due_to_max_due_{w}          Current due amount to maximum due amount over last w months.
            due_to_min_due_{w}          Current due amount to minimum due amount over last w months.
            avg_due_{w}_to_fin          Average due amount over last w months to initial balance.
            max_due_{w}_to_fin          Maximum due amount over last w months to initial balance.
            min_due_{w}_to_fin          Minimum due amount over last w months to initial balance.
            due_to_fin                  Current due amount to initial balance.
            due{w}_to_fin               Past w months (1-6 months) due amount to initial balance.
            avg_due_{s}_to_avg_due_{w}  Average due amount s month to average due amount over last w months (s < w).
            avg_due_{s}_to_max_due_{w}  Average due amount s month to maximum due amount over last w months (s <= w).
            avg_due_{s}_to_min_due_{w}  Average due amount s month to minimum due amount over last w months (s <= w).
            n_last_max_due_{w}          Number of months since maximum due amount in last w months.
        
        Overdue amount related features:
            avg_ovd_{w}                 Average overdue amount over last w months.
            max_ovd_{w}                 Maximum overdue amount over last w months.
            min_ovd_{w}                 Minimum overdue amount over last w months.
            ovd_to_avg_ovd_{w}          Current overdue amount to average overdue amount over last w months.
            ovd_to_max_ovd_{w}          Current overdue amount to maximum overdue amount over last w months.
            ovd_to_min_ovd_{w}          Current overdue amount to minimum overdue amount over last w months.
            avg_ovd_{w}_to_fin          Average overdue amount over last w months to initial balance.
            max_ovd_{w}_to_fin          Maximum overdue amount over last w months to initial balance.
            min_ovd_{w}_to_fin          Minimum overdue amount over last w months to initial balance.
            ovd_to_fin                  Current overdue amount to initial balance.
            ovd{w}_to_fin               Past w months (1-6 months) overdue amount to initial balance.
            avg_ovd_{s}_to_avg_ovd_{w}  Average overdue amount s month to average overdue amount over last w months (s < w).
            avg_ovd_{s}_to_max_ovd_{w}  Average overdue amount s month to maximum overdue amount over last w months (s <= w).
            avg_ovd_{s}_to_min_ovd_{w}  Average overdue amount s month to minimum overdue amount over last w months (s <= w).
            n_last_max_ovd_{w}          Number of months since maximum overdue amount in last w months.
        
    Args:
        df (pd.DataFrame)           : Input dataframe.
        month_ranges (list)         : Observation months ranges.
        feature_col (str)           : The raw feature for processing.
        init_col (str, optional)    : Initial or original financial amount.
                                    If None, financial amount is not defined. Features are not created.

    Returns:
        pd.DataFrame: DataFrame with features due and overdue amounts related columns appended.

    Notes:
        - The financial amount is optional depending on data available.
        - n_last_max_due_{w} returns as CategoricalDtype.
    """

    if feature_col == 'due':
        print("=== Processing ===\n[Due amount features creation]")
    elif feature_col == 'ovd':
        print("=== Processing ===\n[Overdue amount features creation]")
    else:
        print("No features found. Recheck parameters.")

    for w in month_ranges:
        cols = _lag_cols(feature_col, w)
        window = df[cols]
        df[f"avg_{feature_col}_{w}"] = window.sum(axis = 1) / w
        df[f"max_{feature_col}_{w}"] = window.max(axis = 1)
        df[f"min_{feature_col}_{w}"] = window.min(axis = 1)
    
    for w in month_ranges:
        df[f"{feature_col}_to_avg_{feature_col}_{w}"] = _safe_div(df[feature_col], df[f"avg_{feature_col}_{w}"])
        df[f"{feature_col}_to_max_{feature_col}_{w}"] = _safe_div(df[feature_col], df[f"max_{feature_col}_{w}"])
        df[f"{feature_col}_to_min_{feature_col}_{w}"] = _safe_div(df[feature_col], df[f"min_{feature_col}_{w}"])
    
        if init_col is None:
            continue
        else:
            df[f"avg_{feature_col}_{w}_to_{init_col}"] = _safe_div(df[f"avg_{feature_col}_{w}"], df[init_col])
            df[f"max_{feature_col}_{w}_to_{init_col}"] = _safe_div(df[f"max_{feature_col}_{w}"], df[init_col])
            df[f"min_{feature_col}_{w}_to_{init_col}"] = _safe_div(df[f"min_{feature_col}_{w}"], df[init_col])

    if init_col is None:
        pass
    else:
        df[f"{feature_col}_to_{init_col}"] = df[feature_col] / df[init_col]
        for w in range(1, (max(month_ranges) // 2) + 1):
            df[f"{feature_col}{w}_to_{init_col}"] = df[f"{feature_col}{w}"] / df[init_col]
        
    # Average due amount to longer windows where s < w
    for w in month_ranges:
        for s in month_ranges:
            if s >= w:
                continue
            df[f"avg_{feature_col}_{s}_to_avg_{feature_col}_{w}"] = _safe_div(df[f"avg_{feature_col}_{s}"], df[f"avg_{feature_col}_{w}"])

    # Average due amount to max or min longer windows where s <= w
    for w in month_ranges:
        for s in month_ranges:
            if s > w:
                continue
            df[f"avg_{feature_col}_{s}_to_max_{feature_col}_{w}"] = _safe_div(df[f"avg_{feature_col}_{s}"], df[f"max_{feature_col}_{w}"])
            df[f"avg_{feature_col}_{s}_to_min_{feature_col}_{w}"] = _safe_div(df[f"avg_{feature_col}_{s}"], df[f"min_{feature_col}_{w}"])     

    # Number of months since maximum due amount
    for w in month_ranges:
        cols = _lag_cols(feature_col, w)
        window = df[cols]
        cat_type = CategoricalDtype(categories = [i for i in range(1, w + 1)], ordered = True)
        df[f"n_last_max_{feature_col}_{w}"] = (
            window.eq(df[f"max_{feature_col}_{w}"], axis = 0).values.argmax(axis = 1) + 1
        )
        df[f"n_last_max_{feature_col}_{w}"] = df[f"n_last_max_{feature_col}_{w}"].astype(cat_type)

    return df

# Payment features
def create_pay_features(
    df: pd.DataFrame,
    month_ranges: list[int],
    feature_col: list[str],
    init_col: list[str] = None
) -> pd.DataFrame:
    
    """
    Create payment amount aspect features.

    Description:
        'w' is windows refering to the observation months.
        avg_pay_{w}                 Average repayment over last w months.
        max_pay_{w}                 Maximum repayment over last w months.
        min_pay_{w}                 Minimum repayment over last w months.
        pay_to_instal               Current repayment to instalment amount.
        avg_pay_{w}_to_instal       Average repayment over last w months to instalment amount.
        max_pay_{w}_to_instal       Maximum repayment over last w months to instalment amount.
        min_pay_{w}_to_instal       Minimum repayment over last w months to instalment amount.
        pay_to_due                  Current repayment to due amount.
        avg_pay_{w}_to_due          Average repayment over last w months to due amount.
        max_pay_{w}_to_due          Maximum repayment over last w months to due amount.
        min_pay_{w}_to_due          Minimum repayment over last w months to due amount.
        n_partial_pay_{w}           Number of months that partial payment made over last w months (1 = partial payment)
        n_fully_pay_{w}             Number of months that fully payment made over last w months (2 = fully payment)
        n_over_pay_{w}              Number of months that over payment made over last w months (3 = over payment)
        n_fully_to_n_partial_{w}    Number of fully to partial payments over last w months (returns -1 if both zero)
        n_over_to_n_fully_{w}       Number of over to fully payments over last w months (returns -1 if both zero)
        any_pmt_run_{w}             Longest consecutive count on any payment made (pay types >= 1)
        full_pmt_run_{w}            Longest consecutive count on full payment made (pay types == 2)
        
    Args:
        df (pd.DataFrame)           : Input dataframe.
        month_ranges (list)         : Observation months ranges.
        feature_col (list)          : List raw features for processing. **MUST BE** --> [pay, pay_types].
        init_col (str, optional)    : Initial or original instalment and due amount. E.g., [instal, due].
                                    If None, instalment and due amount are not defined. Features are not created.

    Returns:
        pd.DataFrame: DataFrame with features payment amount related columns appended.

    Notes:
        - The instalment and due amount are optional depending on data available.
        - n_partial_pay_{w}, n_fully_pay_{w}, n_over_pay_{w}, any_pmt_run_{w} and full_pmt_run_{w}
        return as CategoricalDtype
    """

    print("=== Processing ===\n[Repayment features creation]")

    for w in month_ranges:
        cols = _lag_cols(feature_col[0], w)
        window = df[cols]
        df[f"avg_{feature_col[0]}_{w}"] = window.sum(axis = 1) / w
        df[f"max_{feature_col[0]}_{w}"] = window.max(axis = 1)
        df[f"min_{feature_col[0]}_{w}"] = window.min(axis = 1)

    if init_col is None:
        pass
    else:
        for col in init_col:
            df[f"{feature_col[0]}_to_{col}"] = _safe_div(df[f"{feature_col[0]}"], df[col])
            for w in month_ranges:
                df[f"avg_{feature_col[0]}_{w}_to_{col}"] = _safe_div(df[f"avg_{feature_col[0]}_{w}"], df[col])
                df[f"max_{feature_col[0]}_{w}_to_{col}"] = _safe_div(df[f"max_{feature_col[0]}_{w}"], df[col])
                df[f"min_{feature_col[0]}_{w}_to_{col}"] = _safe_div(df[f"min_{feature_col[0]}_{w}"], df[col])

    # Payment types count
    for w in month_ranges:
        cols = _lag_cols(feature_col[1], w)
        window = df[cols]
        for code, label in enumerate(["partial", "fully", "over"], start = 1):
            df[f"n_{label}_{feature_col[0]}_{w}"] = window.eq(code).sum(axis = 1)

        # Ratio fully / partial --> returns −1 when both zero
        both_zero = (df[f"n_fully_{feature_col[0]}_{w}"] == 0) & (df[f"n_partial_{feature_col[0]}_{w}"] == 0)
        df[f"n_fully_to_n_partial_{w}"] = np.where(
            both_zero,
            -1,
            _safe_div(df[f"n_fully_{feature_col[0]}_{w}"], df[f"n_partial_{feature_col[0]}_{w}"]),
        )

        # Ratio over / fully --> returns −1 when both zero
        both_zero = (df[f"n_over_{feature_col[0]}_{w}"] == 0) & (df[f"n_fully_{feature_col[0]}_{w}"] == 0)
        df[f"n_over_to_n_fully_{w}"] = np.where(
            both_zero,
            -1,
            _safe_div(df[f"n_over_{feature_col[0]}_{w}"], df[f"n_fully_{feature_col[0]}_{w}"]),
        )

    # Covert to caterogical
    for w in month_ranges:
        cat_type = CategoricalDtype(categories = [i for i in range(1, w + 1)], ordered = True)
        for _, label in enumerate(["partial", "fully", "over"]):
            df[f"n_{label}_{feature_col[0]}_{w}"] = df[f"n_{label}_{feature_col[0]}_{w}"].astype(cat_type)

    # Consecutive-run features (vectorised)
    for w in month_ranges:
        cat_type = CategoricalDtype(categories = [i for i in range(1, w + 1)], ordered = True)
        df[f"full_pmt_run_{w}"] = _consecutive_run_vectorised(df, feature_col[1], w, threshold = 2)
        df[f"full_pmt_run_{w}"] = df[f"full_pmt_run_{w}"].astype(cat_type)
        df[f"any_pmt_run_{w}"]  = _consecutive_run_vectorised(df, feature_col[1], w, threshold = 1)
        df[f"any_pmt_run_{w}"] = df[f"any_pmt_run_{w}"].astype(cat_type)

    return df

# Delinquency features
def create_delinquency_features(
    df: pd.DataFrame,
    month_ranges: list[int],
    feature_col: str,
    n_lags: int = 12,
) -> pd.DataFrame:
    
    """
    Create delinquency status aspect features.

    Description:
        'w' is windows refering to the observation months.
        max_del_{w}                 Maximum delinquency status over last w months.
        ever_{x|30|60|90}_dpd_{w}   A contract has overdue days more than (x) days over last w months.
        n_{x|30|60|90}_dpd_{w}      Number of times that overdue days more than (x) day days over last w months.
        n_month_last_{x|30|60|90}   Months since last event --> return 0 if never reach.
        delq_{x|30|60|90}_run_{w}   Longest consecutive months thatoverdue days more than (x) days over the last w months.
        
    Args:
        df (pd.DataFrame)           : Input dataframe.
        month_ranges (list)         : Observation months ranges.
        feature_col (list)          : The raw feature for processing.
        n_lags (int)                : The observation window. Default as 12 months.

    Returns:
        pd.DataFrame: DataFrame with delinquency status related columns appended.

    Notes:
        - 'n_lags' can be changed.
        - All features on this function return as CategoricalDtype
    """

    print("=== Processing ===\n[Delinquency features creation]")
    
    dpd_labels = ["x", "30", "60", "90"] #Define label for columns name
    
    for w in month_ranges:
        cols = _lag_cols(feature_col, w)
        window = df[cols]
        cat_type = CategoricalDtype(categories = [i for i in range(0, len(dpd_labels) + 1)], ordered = True) #Possible delinquency values
        df[f"max_{feature_col}_{w}"] = window.max(axis = 1)
        df[f"max_{feature_col}_{w}"] = df[f"max_{feature_col}_{w}"].astype(cat_type)

    for w in month_ranges:
        cols = _lag_cols(feature_col, w)
        window = df[cols]
        cat_type_ever = CategoricalDtype(categories = [0, 1], ordered = True) #Ever values (0, 1)
        cat_type = CategoricalDtype(categories = [i for i in range(0, w + 1)], ordered = True) #Start from 0 when count not match
        for threshold, label in enumerate(dpd_labels, start = 1):
            mask = window.ge(threshold)
            df[f"ever_{label}_dpd_{w}"] = mask.any(axis = 1).astype(int)
            df[f"ever_{label}_dpd_{w}"] = df[f"ever_{label}_dpd_{w}"].astype(cat_type_ever)
            df[f"n_{label}_dpd_{w}"] = mask.sum(axis = 1)
            df[f"n_{label}_dpd_{w}"] = df[f"n_{label}_dpd_{w}"].astype(cat_type)

    # Months since last status uses full 12-month history
    cols = _lag_cols(feature_col, n_lags)
    full_window = df[cols].values
    cat_type = CategoricalDtype(categories = [i for i in range(0, n_lags + 1)], ordered = True) #Start from 0 when count not match
    for threshold, label in enumerate(dpd_labels, start = 1):
        condition = full_window == threshold
        any_hit = condition.any(axis = 1)
        df[f"n_month_last_{label}"] = np.where(
            any_hit,
            condition.argmax(axis = 1) + 1,
            0,
        )
        df[f"n_month_last_{label}"] = df[f"n_month_last_{label}"].astype(cat_type)

    # Consecutive delinquency run (vectorised)
    for w in month_ranges:
        cat_type = CategoricalDtype(categories = [i for i in range(0, w + 1)], ordered = True) #Start from 0 when count not match
        for threshold, label in enumerate(dpd_labels, start = 1):
            df[f"{feature_col}_{label}_run_{w}"] = _consecutive_long_vectorised(
                df, feature_col, w, threshold = threshold
            )
            df[f"{feature_col}_{label}_run_{w}"] = df[f"{feature_col}_{label}_run_{w}"].astype(cat_type)

    return df
