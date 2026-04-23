
import warnings
import pandas as pd
import numpy as np
import optuna

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = UserWarning)

def run_optuna(    
    X_train,
    y_train,
    cat_cols: list,
    n_trials: int = 50
) -> tuple:
        
    """
    Optuna for CatBoost model.

    Description:
        Run Optuna hyperparameter search for CatBoost model.
        Refitting best parameters on full training set to get final model.

    Args:
        X_train (pd.DataFrame)  : The mixed data correlation matrix.
        y_train (pd.Series)     : The mixed data correlation matrix.
        cats_cols (list)        : List of final categorical features that being used to define index in dataframe.
        n_trials (int)          : Number of n estimators for training model.
    
    Returns:
        Model object: The best model output from catboost.CatBoostClassifier.
        Study object: The output from optuna.study.Study.

    Notes:
        - N/A
    """

    print("=== Processing ===\n[Model building]")

    # For imbalance target --> manual calculation class weight from true dataset
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale = n_neg / n_pos

    print(f"Class distribution: Non-default = {n_neg:.0f}, Default = {n_pos:.0f}, Scale = {scale:.2f}")

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 1000),
            "depth": trial.suggest_int("depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log = True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3, 30, log = True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight_multiplier", 0.5, 2) * scale,
            "random_strength": trial.suggest_float("random_strength", 0.1, 10, log = True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "eval_metric": "AUC",
            "verbose": False,
            "random_seed": 42
        }

        cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
        aucs = []

        for tr_idx, val_idx in cv.split(X_train, y_train):
            X_tr = X_train.iloc[tr_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_train.iloc[tr_idx]
            y_val = y_train.iloc[val_idx]

            model = CatBoostClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set = (X_val, y_val),
                early_stopping_rounds = 50,
                cat_features = cat_cols,
                verbose = False
            )
            pred = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, pred))

        return np.mean(aucs)

    # Run study
    study = optuna.create_study(direction = "maximize")
    study.optimize(
        objective,
        n_trials = n_trials,
        show_progress_bar = True
    )

    # Re-fitting best model on full train set
    best_params = study.best_params.copy()

    # Convert multiplier to actual scale
    multiplier = best_params.pop("scale_pos_weight_multiplier")
    best_params["scale_pos_weight"] = multiplier * scale
    best_params.update(
        {
            "eval_metric" : "AUC",
            "verbose" : False,
            "random_seed" : 42,
        }
    )

    # Fitting final model
    best_model = CatBoostClassifier(**best_params)
    best_model.fit(
        X_train,
        y_train,
        cat_features = cat_cols,
    )

    print(f"Best AUC: {study.best_value:.4f}")

    return best_model, study