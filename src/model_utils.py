import re
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word_tfidf",
                            TfidfVectorizer(
                                preprocessor=clean_text,
                                stop_words="english",
                                ngram_range=(1, 3),
                                max_features=12000,
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "char_tfidf",
                            TfidfVectorizer(
                                preprocessor=clean_text,
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                max_features=8000,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=2.0,
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)
