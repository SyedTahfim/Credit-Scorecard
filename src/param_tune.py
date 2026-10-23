import pandas as pd
import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

FIXED_PARAMS = {
    "penalty": "l2",
    "solver": "saga",
    "random_state": 42,
}

def tune_model(X: pd.DataFrame, y: np.array, n_trials: int = 5) -> dict:
    def objective(trial):
        params = {
            **FIXED_PARAMS,
            "C": trial.suggest_float("C", 1e-4, 100, log=True),
            "max_iter": trial.suggest_int("max_iter", 1000, 5000, step=1000),
        }
        model = LogisticRegression(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Best CV AUC:", study.best_value)
    print("Best params:", study.best_params)
    return {**FIXED_PARAMS, **study.best_params}