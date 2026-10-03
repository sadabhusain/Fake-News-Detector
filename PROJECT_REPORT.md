# Fake News & Email Detection System

## Project Title

**Fake News & Email Detection System Using Machine Learning**

## Submitted By

**Sachin, Ajay and Devesh**

## Abstract

Fake news and suspicious emails are common problems in today's digital world. False information can spread quickly through news articles, social media posts, messages, and email communication. Similarly, phishing emails can mislead users into sharing sensitive information such as passwords, bank details, OTPs, or personal data.

This project presents a machine learning based system that can classify text as **Real/Normal** or **Fake/Suspicious**. The system accepts a news headline, article text, or email message from the user and predicts whether the content is likely to be genuine or fake. The project uses text preprocessing, TF-IDF feature extraction, and Logistic Regression for classification. A simple Streamlit web interface is provided so users can easily test news or email text.

## Introduction

The internet has made information sharing fast and easy, but it has also increased the spread of fake news and fraudulent emails. Fake news may contain exaggerated claims, false reports, or misleading information. Suspicious emails may ask users to click unknown links, share passwords, provide OTPs, or make urgent payments.

Manual detection of such content is difficult because a large amount of text is shared every day. Machine learning can help by learning patterns from labelled examples and predicting whether new text appears genuine or suspicious.

This project focuses on building a simple and practical detection system using Python and machine learning.

## Objectives

- To develop a machine learning model for detecting fake or suspicious text.
- To classify both news content and email messages.
- To preprocess text data and convert it into numerical features.
- To train a classification model using TF-IDF and Logistic Regression.
- To create a simple graphical web interface using Streamlit.
- To display prediction results with confidence scores.

## Scope Of The Project

The system can be used to check:

- News headlines
- Short news articles
- Email messages
- Phishing-style messages
- Suspicious claims or forwarded text

The system gives output as:

- **Real or Normal**
- **Fake or Suspicious**

## Technologies Used

- **Programming Language:** Python
- **Machine Learning Library:** scikit-learn
- **Data Handling:** pandas
- **Model Saving:** joblib
- **Web Interface:** Streamlit
- **Dataset Format:** CSV

## System Requirements

### Hardware Requirements

- Processor: Intel i3 or above
- RAM: 4 GB or above
- Storage: Minimum 500 MB free space

### Software Requirements

- Python 3.11 or compatible version
- Windows operating system
- Required Python packages from `requirements.txt`
- Web browser such as Chrome or Edge

## Dataset Description

The project uses a CSV dataset named:

```text
data/fake_news_sample.csv
```

The dataset contains two columns:

```text
text,label
```

### Column Details

| Column | Description |
|---|---|
| text | News article, headline, or email message |
| label | Target class: `real` or `fake` |

### Labels

| Label | Meaning |
|---|---|
| real | Real news or normal email |
| fake | Fake news or suspicious/phishing email |

The dataset includes examples of real news, fake news, normal emails, and phishing-style emails.

## Methodology

The project follows these steps:

1. Collect text data with labels.
2. Clean and preprocess the text.
3. Convert text into numerical features using TF-IDF.
4. Train a Logistic Regression model.
5. Save the trained model.
6. Build a Streamlit web app.
7. Accept user input and predict the result.
8. Show the prediction and confidence score.

## Text Preprocessing

Before training the model, the text is cleaned using the following steps:

- Convert text to lowercase.
- Remove URLs.
- Remove punctuation and special characters.
- Remove extra spaces.
- Keep useful words and numbers.

This makes the text easier for the machine learning model to understand.

## Feature Extraction

Machine learning models cannot directly understand text. Therefore, the text is converted into numerical form using **TF-IDF Vectorization**.

TF-IDF stands for **Term Frequency-Inverse Document Frequency**. It gives importance to words based on how frequently they appear in one document and how unique they are across all documents.

This project uses:

- Word-level TF-IDF
- Character-level TF-IDF

Using both word and character features helps the model detect patterns in news text as well as email messages.

## Algorithm Used

The classification algorithm used in this project is:

**Logistic Regression**

Logistic Regression is a supervised machine learning algorithm used for classification problems. It is simple, fast, and effective for text classification tasks.

## Model Pipeline

The model pipeline contains:

1. Text cleaning
2. Word TF-IDF vectorization
3. Character TF-IDF vectorization
4. Feature combination
5. Logistic Regression classifier

## Project Modules

### 1. `app.py`

This file contains the Streamlit web application. It provides the user interface where the user can enter news or email text and get the prediction result.

### 2. `src/model_utils.py`

This file contains helper functions for:

- Cleaning text
- Building the machine learning pipeline
- Saving the model
- Loading the model

### 3. `src/train_model.py`

This file trains the model using the CSV dataset. It also creates evaluation metrics and saves the trained model.

### 4. `src/predict.py`

This file contains the prediction function. It takes user input and returns:

- Predicted label
- Confidence score
- Class probabilities

### 5. `data/fake_news_sample.csv`

This file contains the labelled dataset used for training.

### 6. `models/fake_news_model.joblib`

This is the saved trained machine learning model.

## Working Of The System

1. The user opens the web app.
2. The user enters a news headline, article, or email message.
3. The app loads the trained model.
4. The text is cleaned and converted into TF-IDF features.
5. The Logistic Regression model predicts the class.
6. The app displays whether the text is real/normal or fake/suspicious.
7. The confidence score is also shown.

## Output

The system gives one of the following outputs:

- **Real or Normal**
- **Fake or Suspicious**

It also shows a confidence percentage for the prediction.

## Example Inputs And Outputs

### Example 1

**Input:**

```text
Health department announces vaccination camp schedule on official portal.
```

**Output:**

```text
Real or Normal
```

### Example 2

**Input:**

```text
Dear user your bank account will close today click this link and enter OTP.
```

**Output:**

```text
Fake or Suspicious
```

### Example 3

**Input:**

```text
Secret miracle pill cures all diseases overnight.
```

**Output:**

```text
Fake or Suspicious
```

## Model Evaluation

The model is trained and tested using the available sample dataset. During testing, the model achieved high accuracy on the sample data. The system also stores model metrics in:

```text
models/metrics.json
```

The current sample dataset is suitable for project demonstration. For real-world use, a larger dataset should be used.

## Advantages

- Simple and easy to use.
- Can detect both fake news and suspicious emails.
- Uses machine learning instead of fixed rules only.
- Shows confidence score.
- Clean web interface.
- Easy to expand with more training data.

## Limitations

- The model depends on the quality and size of the dataset.
- It may not detect every type of fake news or phishing email.
- It is trained on a sample dataset, so real-world accuracy may vary.
- It does not verify facts from live internet sources.
- It only analyzes text and does not check attachments or links directly.

## Future Enhancements

- Use a larger real-world dataset.
- Add multilingual support.
- Add URL safety checking.
- Add email header analysis.
- Use advanced deep learning models.
- Add user feedback to improve predictions.
- Deploy the app online.

## Conclusion

The Fake News & Email Detection System successfully demonstrates how machine learning can be used to classify text as real/normal or fake/suspicious. The project uses Python, TF-IDF vectorization, Logistic Regression, and Streamlit to create a practical text classification application.

The system can help users identify misleading news and suspicious emails. Although the current version is designed for academic demonstration, it can be improved further by using larger datasets and advanced models.

## How To Run The Project

Open PowerShell in the project folder and run:

```powershell
python -m src.train_model
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

