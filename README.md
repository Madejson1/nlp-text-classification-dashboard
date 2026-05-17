# nlp-text-classification-dashboard
NLP text classification pipeline using TF-IDF and multiple machine learning models (Logistic Regression, SVM, Random Forest, XGBoost). Performed preprocessing, feature engineering, hyperparameter tuning and model evaluation, followed by interactive Power BI dashboard for performance analysis, error inspection and prediction confidence.

## Goal

Build and evaluate machine learning models for classifying text messages as spam or ham.

## Tech stack

- Python
- pandas
- NumPy
- SciPy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- TF-IDF
- hyperparameter tuning
- Power BI-ready CSV exports

## What the project covers

- text preprocessing
- feature engineering with TF-IDF
- model training
- model comparison
- hyperparameter tuning
- evaluation using accuracy, precision, recall, F1 and ROC-AUC
- visualizations
- export of predictions and metrics for Power BI

## Dataset

The project uses a spam/ham SMS-style dataset.

Expected file:

```text
data/raw/spam.csv
```

Expected columns:

```text
label,text
ham,example message
spam,example message
```

A small synthetic dataset is included so the project runs immediately.

## Run

Create environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the pipeline:

```bash
python -m src.run_pipeline
```


Then open the notebook:

```bash
notebooks/report.ipynb
```
and run the final cells to generate Power BI input files.


Generated outputs:

```text
outputs/metrics/model_metrics.csv
outputs/predictions/test_predictions.csv
outputs/figures/confusion_matrix_best_model.png
outputs/figures/roc_curve_best_model.png
```

## Power BI Dashboard

1. Open the template:
   dashboard/nlp_text_classification_dashboard.pbit

2. When prompted, set the parameter:

ProjectRoot = path to the project root directory

Example:
C:\path\to\nlp-text-classification-dashboard\

3. Make sure the pipeline has been run:

python -m src.run_pipeline
