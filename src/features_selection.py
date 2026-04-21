
import warnings
import pandas as pd

from lightgbm import LGBMClassifier

warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = UserWarning)

# Features selection
def run_boruta(
    X_encoded: pd.DataFrame,
    y: pd.Series,
    cat_cols: list = None,
    pass_threshold: float = 1.0,
    random_state: int = 42,
) -> tuple[list, pd.DataFrame]:
    
    """
    Run Boruta with LightGBM for features selection.

    Description:
        Leveraging Boruta concept to perform the featuers selection.
        Data shadowing is used to reduce the information bias by random generator concept.
        The LightGBM is the core model that running data real/shadow with 'gain' important.
        The threshold is estimated from data shadow and appiled to real data as cut-off.

    Args:
        X_encoded (pd.DataFrame)            : Encoded training data for features selection.
        y (pd.DataFrame)                    : Target training data.
        cols_types (list, optional)         : List of categorical features. 
                                            If None, categorical features are not defined.
        pass_threshold (float, optional)    : The percent of defined cut-off.
                                            If None, use pure cut-off.
        random_state (int)                  : The controls the randomness.

    Returns:
        list            : List of selected features
        pd.DataFrame    : Summary of Features Importance of all features for documenation.

    Notes:
        - The cut-off threshold is used .max() that can be relaxed if a few features passed.
        - The relaxing criteria can be .mean(), xx% of .max(), ect.
        - This function is using xx% of .max() to allow more features.
    """

    print("=== Processing ===\n[Features selection by LightGBM]")
    print(f"Total features: {X_encoded.shape[1]}\nTotal shadow features: {X_encoded.shape[1] * 2}")

    X_shadow = X_encoded.apply(
        lambda x: x.sample(frac = 1, random_state = random_state).values
    )
    X_shadow.columns = ['shadow_' + col for col in X_encoded.columns]
    X_boruta = pd.concat([X_encoded, X_shadow], axis = 1)
    model = LGBMClassifier(
        importance_type = 'gain',
        n_jobs = -1,
        random_state = random_state,
        verbosity = -1
    )
    model.fit(
        X_boruta,
        y,
        categorical_feature = cat_cols
    )
    features_imp = pd.Series(
        model.feature_importances_,
        index = X_boruta.columns,
        name = 'importance'
    )
    threshold = features_imp[X_shadow.columns].mean() * pass_threshold
    confirmed_features = features_imp[X_encoded.columns][features_imp[X_encoded.columns] > threshold].index.tolist()
    features_imp_output = features_imp[X_encoded.columns].reset_index().rename(
        columns = {
            'index': 'feature'
        }
    )
    features_imp_output['importance'] = features_imp[X_shadow.columns].tolist()
    features_imp_output = features_imp_output.sort_values(
        by = ['importance'],
        ascending = False,
        ignore_index = True
    )

    print(f"Total selected features: {len(confirmed_features)}")
    print(f"Cut-off selected threshold: {threshold:.2f}")

    return confirmed_features, features_imp_output