from pathlib import Path


RANDOM_STATE = 42

DATA_PATH = Path("data/raw/spam.csv")

OUTPUT_DIR = Path("outputs")
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
FIGURES_DIR = OUTPUT_DIR / "figures"

TEST_SIZE = 0.25