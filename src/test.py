# validation.py
import mlflow
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, recall_score


def evaluate_model(model, X_test, y_test, threshold: float, run_id: str = None):
    """
    Evaluates a trained logistic regression model on test data.

    Args:
        model: Trained model.
        X_test: WoE-transformed test features.
        y_test: Test labels.
        threshold: probability cutoff for classification.
        run_id: MLflow run_id from model_train; if given, test metrics are logged to that run.
    """
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    auc = roc_auc_score(y_test, proba)
    recall = recall_score(y_test, preds)

    print("Test AUC:", round(auc, 4))
    print("Classification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    if run_id is not None:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("test_auc", auc)
            mlflow.log_metric("test_recall", recall)