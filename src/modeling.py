from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from src.config import RANDOM_STATE


def get_models() -> dict:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer()),
                ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "linear_svm": Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer()),
                ("model", CalibratedClassifierCV(LinearSVC(random_state=RANDOM_STATE))),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=150,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=100,
                        learning_rate=0.08,
                        max_depth=3,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def get_tuning_grid() -> dict:
    return {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [1000, 3000, 5000],
        "model__C": [0.5, 1.0, 2.0],
    }


def tune_logistic_regression(X_train, y_train):
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=get_tuning_grid(),
        scoring="f1",
        cv=3,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_