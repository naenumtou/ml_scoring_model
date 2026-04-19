
import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from category_encoders import TargetEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = UserWarning)

# Categorical features encoder
def cats_cv_target_encode(
    X: pd.DataFrame,
    y: pd.Series,
    cols_types: list,
    n_splits: int = 5,
    smoothing: float = 10.0,
    random_state: int = 42,
) -> pd.DataFrame:
    
    """
    Encoding of categorical features by target encoder.

    Description:
        The categorical features need to be transformed before modelling process.
        The target encoding is selected to use as the transformer.
        The Leakage-safe Target Encoding by K-Fold cross validation encoding.
        The features are encoded by target from other folds (NOT Target from own fold).

    Args:
        X (pd.DataFrame)    : Features training data.
        y (pd.DataFrame)    : Target training data.
        cols_types (list)   : List of categorical features.
        n_splits (int)      : Number of folds.
        smoothing (float)   : The amount of mixing of the target mean.
        random_state (int)  : The controls the randomness.

    Returns:
        pd.DataFrame: Categorical features encoded data.

    Notes:
        - Applicable only for categorical features.
        - The function can handle missing values.
    """

    print("=== Processing ===\n[CV Target Encoding]")
    print(f"Total features: {X.shape[1]}\nTotal categorical features: {len(cols_types)}")

    X_enc = X.copy()
    global_means = {col: y.mean() for col in cols_types} #Global means calculation for missing values
    kf = KFold(
        n_splits = n_splits,
        shuffle = True,
        random_state = random_state
    )

    for tr_idx, val_idx in kf.split(X):
        te = TargetEncoder(
            cols = cols_types,
            smoothing = smoothing,
            handle_unknown = 'value', #For unseen category --> global mean
            handle_missing = 'value', #For missing values --> global mean
        )
        te.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        encoded_val = te.transform(X.iloc[val_idx])[cols_types]

        # Missing values
        for col in cols_types:
            encoded_val[col] = encoded_val[col].fillna(global_means[col])
        
        X_enc.iloc[val_idx, X_enc.columns.get_indexer(cols_types)] = encoded_val.values

    # Missing values
    for col in cols_types:
        X_enc[col] = X_enc[col].fillna(global_means[col])

    # From cat cols --> float
    X_enc[cols_types] = X_enc[cols_types].astype(float)

    return X_enc

# Numerical features impute missing values
def num_mice_impute(
    X: pd.DataFrame,
    cols_types: list,
    max_iter: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    
    """
    Multiple Imputation by Chained Equations (MICE) for numerical features.

    Description:
        The missing values of numerical features need to be imputed before modelling process.
        The MICE by BayesianRidge is selected to use as the imputor.
        The imputation process is only preliminary process before features selection.
        The missing values are not allowed to features selection process. 

    Args:
        X (pd.DataFrame)    : Features training data.
        cols_types (list)   : List of numerical features.
        max_iter (int)      : Maximum number of imputation rounds to perform.
        random_state (int)  : The controls the randomness.

    Returns:
        pd.DataFrame: Numerical features imputed missing values data.

    Notes:
        - Applicable only for numerical features.
        - The BayesianRidge can be changed due to it is only preliminary process.
    """

    print("=== Processing ===\n[MICE for missing values]")
    print(f"Total features: {X.shape[1]}\nTotal numerical features: {len(cols_types)}")
    
    X[cols_types] = X[cols_types].replace([np.inf, -np.inf], np.nan)
    missing_train = X[cols_types].isna().sum()
    has_missing = missing_train[missing_train > 0].index.tolist()

    print(f"Total missing values in numerical features: {len(has_missing)}")
    
    # Checking for missing values
    if not has_missing:
        print("Missing values not found\nEnd process")
        return X.copy()
    
    # MICE for missing values found
    imputer = IterativeImputer(
        estimator = BayesianRidge(
            max_iter = max_iter
        ),
        max_iter = max_iter,
        random_state = random_state,
        verbose = 0,
    )
    X_train_imp = X.copy()
    X_train_imp[has_missing] = imputer.fit_transform(X[has_missing])
    
    # Checking for missing values again
    remaining_train = X_train_imp[has_missing].isna().sum().sum()
    print(f"Total missing values in numerical features after imputation: {remaining_train}")

    return X_train_imp