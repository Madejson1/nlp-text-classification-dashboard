import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def get_positive_probabilities(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            return scores

        return (scores - min_score) / (max_score - min_score)

    return model.predict(X)


def evaluate_model(model_name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = get_positive_probabilities(model, X_test)

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def build_predictions_dataframe(model, X_test, y_test, original_texts) -> pd.DataFrame:
    y_pred = model.predict(X_test)
    y_proba = get_positive_probabilities(model, X_test)

    df = pd.DataFrame(
        {
            "text": original_texts,
            "actual": y_test,
            "predicted": y_pred,
            "spam_probability": y_proba,
        }
    )

    df["actual_label"] = df["actual"].map({0: "ham", 1: "spam"})
    df["predicted_label"] = df["predicted"].map({0: "ham", 1: "spam"})
    df["is_correct"] = df["actual"] == df["predicted"]
    df["error_type"] = "correct"
    df.loc[(df["actual"] == 0) & (df["predicted"] == 1), "error_type"] = "false_positive"
    df.loc[(df["actual"] == 1) & (df["predicted"] == 0), "error_type"] = "false_negative"

    return df