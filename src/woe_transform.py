# woe_transform.py
"""WoETransformer: optimal binning, IV and WoE transformation."""

import joblib
import pandas as pd
from optbinning import OptimalBinning


class WoETransformer:

    def __init__(self, woe_config: dict):
        self.target = woe_config["target"]
        self.solver = woe_config["solver"]
        self.optb_models = {}
        self.iv_table = pd.Series(dtype=float)

    def fit(self, data: pd.DataFrame) -> "WoETransformer":
        X = data.drop(columns=[self.target])
        y = data[self.target].values

        self.optb_models = {}
        iv = {}
        for col in X.columns:
            dtype = "numerical" if pd.api.types.is_numeric_dtype(X[col]) else "categorical"
            optb = OptimalBinning(name=col, dtype=dtype, solver=self.solver)
            optb.fit(X[col].values, y)
            self.optb_models[col] = optb
            iv[col] = optb.binning_table.build().loc["Totals", "IV"]

        self.iv_table = pd.Series(iv).sort_values(ascending=False)
        return self

    def transform(self, data: pd.DataFrame, variables: list = None) -> pd.DataFrame:
        cols = list(self.iv_table.index) if variables is None else list(variables)

        not_fitted = [c for c in cols if c not in self.optb_models]
        if not_fitted:
            raise KeyError(f"Variables not fitted: {not_fitted}")
        not_in_data = [c for c in cols if c not in data.columns]
        if not_in_data:
            raise KeyError(f"Variables missing from data: {not_in_data}")

        woe = {
            f"{col}_woe": self.optb_models[col].transform(
                data[col].values,
                metric="woe",
                metric_missing="empirical",   # missing gets its own bin's WoE, not 0
                metric_special="empirical",
            )
            for col in cols
        }
        return pd.DataFrame(woe, index=data.index)

    def fit_transform(self, data: pd.DataFrame, variables: list = None) -> pd.DataFrame:
        return self.fit(data).transform(data, variables)

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "WoETransformer":
        return joblib.load(path)