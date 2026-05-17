import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH,
    METRICS_DIR,
    PREDICTIONS_DIR,
    FIGURES_DIR,
    TEST_SIZE,
    RANDOM_STATE,
)
from src.preprocessing import load_dataset
from src.modeling import get_models, tune_logistic_regression
from src.evaluation import (
    evaluate_model,
    build_predictions_dataframe,
    get_positive_probabilities,
)
from src.visualization import save_confusion_matrix, save_roc_curve


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_PATH)

    X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
        df["text_clean"],
        df["target"],
        df["text"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )

    models = get_models()
    metrics = []
    fitted_models = {}

    for model_name, model in models.items():
        print(f"Training: {model_name}")
        model.fit(X_train, y_train)
        fitted_models[model_name] = model
        metrics.append(evaluate_model(model_name, model, X_test, y_test))

    print("Tuning logistic regression")
    tuned_model, best_params, best_cv_score = tune_logistic_regression(X_train, y_train)
    fitted_models["logistic_regression_tuned"] = tuned_model

    tuned_metrics = evaluate_model("logistic_regression_tuned", tuned_model, X_test, y_test)
    tuned_metrics["best_cv_f1"] = best_cv_score
    tuned_metrics["best_params"] = str(best_params)
    metrics.append(tuned_metrics)

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.sort_values("f1", ascending=False)

    metrics_path = METRICS_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    best_model_name = metrics_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    predictions_df = build_predictions_dataframe(
        best_model,
        X_test,
        y_test.reset_index(drop=True),
        text_test.reset_index(drop=True),
    )

    predictions_path = PREDICTIONS_DIR / "test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    y_pred = best_model.predict(X_test)
    y_proba = get_positive_probabilities(best_model, X_test)

    save_confusion_matrix(
        y_test,
        y_pred,
        FIGURES_DIR / "confusion_matrix_best_model.png",
    )
    save_roc_curve(
        y_test,
        y_proba,
        FIGURES_DIR / "roc_curve_best_model.png",
    )

    print("\nDone.")
    print(f"Best model: {best_model_name}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Predictions saved to: {predictions_path}")


if __name__ == "__main__":
    main()