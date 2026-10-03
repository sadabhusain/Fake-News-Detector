# Fake News & Email Detection

A Python machine learning project for detecting whether a news article or email message is likely to be **real/normal** or **fake/suspicious**. The project uses text preprocessing, word and character TF-IDF vectorization, and a Logistic Regression classifier. It also includes a Streamlit web app for live predictions.

## Project Structure

```text
15May/
├── app.py
├── data/
│   └── fake_news_sample.csv
├── models/
│   └── .gitkeep
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── model_utils.py
│   ├── predict.py
│   └── train_model.py
└── README.md
```

## How It Works

1. News text is cleaned by lowercasing, removing links, punctuation, and extra spaces.
2. Word and character TF-IDF convert the cleaned text into numerical features.
3. Logistic Regression learns patterns from labelled fake and real news examples.
4. The trained model predicts whether new input text is fake/suspicious or real/normal.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Train The Model

```powershell
python -m src.train_model
```

This creates:

```text
models/fake_news_model.joblib
models/metrics.json
```

## Run The Web App

You can run the app from an external window so it keeps running even if VS Code is closed.

- Double-click `run_app.bat` from the `15May` folder.
- Or run this from PowerShell:

```powershell
cd 15May
.\run_app.bat
```

Then open the local URL shown in the new browser window or terminal.

## Dataset Note

This project includes a small sample dataset so the app can run immediately. For a stronger final submission, replace or extend `data/fake_news_sample.csv` with a larger dataset such as the Kaggle Fake and Real News Dataset. Keep the same columns:

```text
text,label
```

Where `label` must be:

```text
fake
real
```

For email examples, use `fake` for phishing or suspicious emails and `real` for normal legitimate emails.
