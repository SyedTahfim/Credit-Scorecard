# Import Libraries 
import argparse
import logging 
import json
import joblib
import pandas as pd
import yaml
from pathlib import Path

from train_test_split import split_data
from data_prep import preprocess_data
from feature_engineering import FeatureEngineering
from param_tune import tune_model
from train import model_train
from test import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def data_stage(cfg):
    train_df, test_df = split_data(cfg['data'], cfg['train_test_split'])
    train_proc = preprocess_data(train_df, cfg['prep']) 
    test_proc = preprocess_data(test_df, cfg['prep']) [train_proc.columns]
    data_dir = Path(cfg["data"]["data_path"]) 
    train_proc.to_csv(data_dir / cfg["data"]["train_prcd"], index=False)
    test_proc.to_csv(data_dir / cfg["data"]["test_prcd"], index=False)
    return train_proc, test_proc

def feature_stage(cfg, train_proc, test_proc):
    target = cfg["data"]["target_variable"]
    fe = FeatureEngineering(cfg["data"], cfg["woe"], cfg["corr_vif"])

    X_woe, y = fe.fit_transform_woe(train_proc)
    X_clean, dropped = fe.drop_corr_cols(X_woe)
    log.info("Dropped %d correlated columns: %s", len(dropped), dropped)

    variables = X_clean.columns.str.removesuffix("_woe").tolist()
    _, coef_stability_df = fe.variable_stability(train_proc.drop(columns=target), y, variables)
    stable_vars, _ = fe.select_variables(coef_stability_df)
    fe.save_outputs()

    X_train, y_train = fe.transform_woe(train_proc, variables=stable_vars)
    X_test, y_test = fe.transform_woe(test_proc, variables=stable_vars)
    return fe, stable_vars, X_train, y_train, X_test, y_test

def save_woe_test(cfg, X_test, y_test):
    test_woe = X_test.copy()
    test_woe[cfg['data']['target_variable']] = y_test
    test_woe.to_csv(cfg['data']['test_woe_path'],index=True,
                    index_label = "row_id")

def tune_stage(cfg, X_train, y_train):
    best_params = tune_model(X_train, y_train)
    out = Path(cfg["data"]["model_path"]) / "best_params.json"
    out.write_text(json.dumps(best_params, indent=2))
    return best_params

def fit_stage(cfg, X_train, y_train, best_params):
    model, run_id = model_train(X_train, y_train, best_params, cfg["model"]["threshold"])
    model_dir = Path(cfg["data"]["model_path"])
    joblib.dump(model, model_dir / "model.joblib")

    coef_df = pd.DataFrame({
        "variable": X_train.columns.str.removesuffix("_woe"),
        "coefficient": model.coef_[0],
    })
    coef_df.loc[len(coef_df)] = ["intercept", model.intercept_[0]]
    coef_df.to_csv(model_dir / "coefficients.csv", index=False)
    return model, run_id

def validation_stage(cfg, model, run_id, X_test, y_test):
    evaluate_model(model, X_test, y_test, cfg["model"]["threshold"], run_id)

def main(config_path: str):
    cfg = load_config(config_path)
    Path(cfg["data"]["model_path"]).mkdir(parents=True, exist_ok=True)
    train_proc, test_proc = data_stage(cfg)
    fe, stable_vars, X_train, y_train, X_test, y_test = feature_stage(cfg, train_proc, test_proc)
    save_woe_test(cfg,X_test,y_test)
    best_params = tune_stage(cfg, X_train, y_train)
    model, run_id = fit_stage(cfg, X_train, y_train, best_params)
    validation_stage(cfg, model, run_id, X_test, y_test)
    # next: scorecard scaling


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    main(parser.parse_args().config)






