from pathlib import Path

import streamlit as st

from src.config import MODEL_PATH
from src.model_utils import load_model
from src.predict import predict_news


st.set_page_config(
    page_title="Fake News Detection",
    page_icon="F",
    layout="centered",
)


def apply_styles():
    st.markdown(
        """
        <style>
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="manage-app-button"],
        #MainMenu,
        footer,
        header {
            visibility: hidden;
            display: none;
        }

        .stApp {
            background:
                linear-gradient(135deg, rgba(37, 99, 235, 0.10), rgba(22, 163, 74, 0.08)),
                #f4f7fb;
            color: #111827;
        }

        .block-container {
            max-width: 760px;
            padding-top: 1.2rem;
            padding-bottom: 1.5rem;
        }

        .main-box {
            background:
                linear-gradient(135deg, rgba(37, 99, 235, 0.95), rgba(14, 165, 233, 0.86)),
                #2563eb;
            border: 1px solid #dbe3ef;
            border-radius: 10px;
            padding: 1.25rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
            border-bottom: 5px solid #f59e0b;
        }

        .project-title {
            font-size: 2rem;
            font-weight: 800;
            color: #ffffff;
            margin: 0 0 0.25rem;
        }

        .team {
            color: #e0f2fe;
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }

        .note {
            color: #f8fafc;
            line-height: 1.5;
            margin-bottom: 1rem;
        }

        .result-box {
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
            border: 1px solid #dbe3ef;
            background: #ffffff;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        }

        .result-label {
            font-size: 1.35rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }

        .small-text {
            color: #64748b;
            font-size: 0.9rem;
        }

        .stTextArea textarea {
            background: #ffffff;
            color: #111827;
            border-radius: 8px;
            border: 1px solid #bfdbfe;
        }

        .stButton button {
            background: linear-gradient(135deg, #2563eb, #0ea5e9);
            color: white;
            border: 0;
            border-radius: 8px;
            font-weight: 700;
            box-shadow: 0 8px 16px rgba(37, 99, 235, 0.22);
        }

        .stButton button:hover {
            background: #1d4ed8;
            color: white;
            border: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


def show_result(result: dict):
    is_fake = result["label"] == "fake"
    label = "Fake or Suspicious" if is_fake else "Real or Normal"
    color = "#dc2626" if is_fake else "#16a34a"

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label" style="color: {color};">{label}</div>
            <div>Confidence: <strong>{result["confidence"]:.2%}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Show probabilities"):
        for class_name, probability in result["probabilities"].items():
            st.write(f"{class_name.title()}: {probability:.2%}")


def main():
    apply_styles()

    st.markdown(
        """
        <div class="main-box">
            <div class="project-title">Fake News & Email Detection</div>
            <div class="team">Prepared by: Sadab Hussian, Ajay and Devesh</div>
            <div class="note">
                Enter a news headline, article text, or email message. The system will
                predict whether the text is likely to be real/normal or fake/suspicious.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not Path(MODEL_PATH).exists():
        st.warning("Model not found. Run `python -m src.train_model` first.")
        st.stop()

    model = get_model()
    sample_text = "Dear user your account will be blocked today click this link and enter your password."

    news_text = st.text_area(
        "News or email text",
        value=sample_text,
        height=140,
        placeholder="Paste news or email text here...",
    )

    if st.button("Check News", type="primary"):
        if not news_text.strip():
            st.error("Please enter news or email text.")
            return

        result = predict_news(model, news_text)
        show_result(result)

    st.markdown(
        '<p class="small-text">Model used: Word/character TF-IDF with Logistic Regression. It can check news text and email messages.</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
