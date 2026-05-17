import re
import pandas as pd


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    expected_columns = {"label", "text"}
    if not expected_columns.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {expected_columns}")

    df = df[["label", "text"]].dropna()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(["ham", "spam"])]

    df["text_clean"] = df["text"].apply(clean_text)
    df["target"] = df["label"].map({"ham": 0, "spam": 1})

    return df