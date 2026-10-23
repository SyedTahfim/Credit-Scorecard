# feature_engineering.py
"""
Feature engineering on the processed train data:
1. Fit optimal binning and compute IV; keep variables above the IV threshold.
2. WoE transformation.
3. Drop correlated variables using correlation and VIF thresholds.
4. Test stability with stratified CV (AUC, KS, coefficient CV, sign consistency).
5. Save the fitted transformer and the stable variable list.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor

from woe_transform import WoETransformer


class FeatureEngineering:

    def __init__(self, data_cfg: dict, woe_config: dict, corr_vif_config: dict):
        self.data_cfg = data_cfg
        self.woe_config = woe_config
        self.corr_vif_config = corr_vif_config
        self.woe = None
        self.strong_vars = []
        self.vif_drop_cols = []
        self.stable_var = []

    def fit_transform_woe(self, train_df: pd.DataFrame):
        target = self.data_cfg["target_variable"]
        y = train_df[target]

        self.woe = WoETransformer(self.woe_config).fit(train_df)  # target stays inside
        self.strong_vars = [
            v for v, iv in self.woe.iv_table.items()
            if iv > self.woe_config["iv_threshold"]
        ]
        X_woe = self.woe.transform(train_df, variables=self.strong_vars)
        return X_woe, y

    # Transform only; use on train, test and holdout
    def transform_woe(self, df: pd.DataFrame, variables: list = None):
        target = self.data_cfg["target_variable"]
        variables = self.strong_vars if variables is None else variables
        X = self.woe.transform(df.drop(columns=target), variables=variables)
        y = df[target]
        return X, y

    @staticmethod
    def calc_vif(X: pd.DataFrame) -> pd.Series:
        Xc = sm.add_constant(X)
        vif = pd.Series(
            [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])],
            index=Xc.columns,
        )
        return vif.drop("const")

    def drop_corr_cols(self, X: pd.DataFrame):
        corr_thresh = self.corr_vif_config["corr_threshold"]
        vif_thresh = self.corr_vif_config["vif_threshold"]
        X = X.copy()
        dropped = []

        while True:
            corr = X.corr().abs()
            mask = (corr > corr_thresh) & (corr < 1)
            mask = mask & np.triu(np.ones(corr.shape, dtype=bool), k=1)
            r, c = np.where(mask)
            if len(r) == 0:
                break

            pairs = sorted(
                zip(corr.index[r], corr.columns[c]),
                key=lambda p: corr.loc[p[0], p[1]],
                reverse=True,
            )

            vif = self.calc_vif(X)  # recomputed on the current X each pass
            drop = None
            for a, b in pairs:
                candidate = a if vif[a] >= vif[b] else b
                if vif[candidate] > vif_thresh:
                    drop = candidate
                    break
            if drop is None:
                break

            dropped.append(drop)
            X = X.drop(columns=[drop])

        self.vif_drop_cols = dropped
        return X, dropped

    @staticmethod
    def ks_statistic(y_true, y_pred) -> float:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        return np.max(tpr - fpr)

    def variable_stability(self, X: pd.DataFrame, y: pd.Series, variables: list,
                       n_splits: int = 10, random_state: int = 42,
                       cv_max: float = 0.10, sign_min: float = 90):
        X = X[variables]   # raw columns that survived IV and correlation/VIF
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        auc_train, auc_val, ks_train, ks_val, coefs = [], [], [], [], []
        woe_cols = None

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Fresh WoE fit on this fold's training rows only
            woe = WoETransformer(self.woe_config).fit(pd.concat([X_train, y_train], axis=1))
            X_train_woe = woe.transform(X_train)
            X_val_woe = woe.transform(X_val)

            if woe_cols is None:
                woe_cols = X_train_woe.columns
            X_train_woe, X_val_woe = X_train_woe[woe_cols], X_val_woe[woe_cols]

            model = LogisticRegression(max_iter=1000).fit(X_train_woe, y_train)
            pred_train = model.predict_proba(X_train_woe)[:, 1]
            pred_val = model.predict_proba(X_val_woe)[:, 1]

            auc_train.append(roc_auc_score(y_train, pred_train))
            auc_val.append(roc_auc_score(y_val, pred_val))
            ks_train.append(self.ks_statistic(y_train, pred_train))
            ks_val.append(self.ks_statistic(y_val, pred_val))
            coefs.append(model.coef_[0])

        rows = []
        for name, train_scores, val_scores in [("AUC", auc_train, auc_val), ("KS", ks_train, ks_val)]:
            rows.append({
                "Metric": name,
                "train_mean": np.mean(train_scores), "train_std": np.std(train_scores),
                "val_mean": np.mean(val_scores), "val_std": np.std(val_scores),
                "train_cv": np.std(train_scores) / np.mean(train_scores),
                "val_cv": np.std(val_scores) / np.mean(val_scores),
                "overfit_gap": np.mean(train_scores) - np.mean(val_scores),
            })
        model_stability = pd.DataFrame(rows).round(4)


        # Variable-level stability
        coefs = pd.DataFrame(coefs, columns=woe_cols)
        mean, std = coefs.mean(), coefs.std()
        coef_cv = (std / mean.abs()).replace([np.inf, -np.inf], np.nan)
        sign_pct = coefs.apply(lambda c: (np.sign(c) == np.sign(c.mean())).mean() * 100)

        coef_stability_df = pd.DataFrame({
            "Feature": woe_cols,
            "coeff_mean": mean.values,
            "coeff_std": std.values,
            "coeff_cv": coef_cv.values,
            "Sign Consistency Pct": sign_pct.round(2).values,
        })
        coef_stability_df["CV_Flag"] = np.where(coef_stability_df["coeff_cv"] < cv_max, "Pass", "Fail")
        coef_stability_df["Sign_Flag"] = np.where(coef_stability_df["Sign Consistency Pct"] >= sign_min, "Pass", "Fail")
        coef_stability_df["Overall_Stability"] = np.where(
            (coef_stability_df["CV_Flag"] == "Pass") & (coef_stability_df["Sign_Flag"] == "Pass"),
            "Pass", "Fail",
        )
        coef_stability_df = coef_stability_df.sort_values("Sign Consistency Pct", ascending=False).round(4)

        return model_stability, coef_stability_df

    def select_variables(self, coef_stability_df: pd.DataFrame):
        df = coef_stability_df.copy()
        df["IV"] = df["Feature"].str.removesuffix("_woe").map(self.woe.iv_table)
        self.stable_var = (
            df.loc[df["Overall_Stability"] == "Pass", "Feature"]
            .str.removesuffix("_woe").tolist()
        )
        return self.stable_var, df
    
    def save_woe_table(self, path) -> pd.DataFrame:
        frames = []
        for var in self.stable_var:
            optb = self.woe.optb_models[var]
            bt = optb.binning_table.build().iloc[:-1].copy()  # drop the Totals row
            bt.insert(0, "variable", var)
            frames.append(bt.reset_index(drop=True))
        woe_df = pd.concat(frames, ignore_index=True)
        woe_df.to_csv(path, index=False)
        return woe_df
    
    def save_outputs(self) -> None:
        out_dir = Path(self.data_cfg["data_path"])
        self.woe.save(out_dir / "woe_transformer.joblib")
        self.save_woe_table(out_dir / "woe_table.csv")
        with open(out_dir / "stable_vars.json", "w") as f:
            json.dump(self.stable_var, f)