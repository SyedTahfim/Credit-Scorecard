# Import necessary libraries
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score


def model_train(X_train: pd.DataFrame, y_train: np.array,
                params: dict, threshold: float):
    """
    Trains a logistic regressionn model and logs the results with MLflow. 
    
    Args: 
        df: train data including selected variables. 
        target_col: Name of the target variable

    """
    with mlflow.start_run() as run:
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        mlflow.log_params(params)
        mlflow.log_param("threshold", threshold)

        train_proba = model.predict_proba(X_train)[:, 1]
        train_pred = (train_proba >= threshold).astype(int)
        mlflow.log_metric("train_auc", roc_auc_score(y_train, train_proba))
        mlflow.log_metric("train_recall", recall_score(y_train, train_pred))
        mlflow.sklearn.log_model(model, artifact_path="model")

    return model, run.info.run_id




