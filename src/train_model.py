import json

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, METRICS_PATH, MODEL_PATH, RANDOM_STATE
from src.model_utils import build_pipeline, save_model


def load_dataset() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    required_columns = {"text", "label"}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data = data.dropna(subset=["text", "label"])
    data["label"] = data["label"].str.lower().str.strip()
    data = data[data["label"].isin(["fake", "real"])]

    if data.empty:
        raise ValueError("Dataset has no valid rows after cleaning.")

    return data


def train() -> dict:
    data = load_dataset()
    stratify = data["label"] if data["label"].nunique() > 1 else None

    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    model = build_pipeline()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=["fake", "real"],
        ).tolist(),
        "labels": ["fake", "real"],
        "training_rows": int(len(x_train)),
        "testing_rows": int(len(x_test)),
    }

    save_model(model, MODEL_PATH)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    result = train()
    print("Model trained successfully.")
    print(f"Accuracy: {result['accuracy']:.2%}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
