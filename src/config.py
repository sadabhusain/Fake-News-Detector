from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "fake_news_sample.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "fake_news_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
RANDOM_STATE = 42

