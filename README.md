# Transformer-Based Social Media Sentiment Analysis

A machine learning and NLP project that classifies social media text into **positive, neutral, and negative sentiment** using TF-IDF with Logistic Regression and a pre-trained Twitter-RoBERTa Transformer. The project also includes fine-tuning, confidence scores, batch analysis, and rule-based derived intelligence.

## Overview

Understanding public opinion on social media is important for identifying user satisfaction, complaints, and emerging issues. This project compares a traditional machine learning approach with Transformer-based sentiment analysis and provides an interactive interface for analyzing individual text inputs and CSV datasets.

## Features

- **Single Text Analysis:** Predict sentiment for individual text inputs.
- **Batch Analysis:** Upload a CSV file to analyze multiple text records.
- **Multiple Models:** Compare TF-IDF + Logistic Regression, pre-trained RoBERTa, and fine-tuned RoBERTa.
- **Confidence Scores:** Display prediction confidence and class probabilities.
- **Derived Intelligence:** Use rule-based logic to identify emotion, urgency, topic, and explanations.
- **Evaluation Dashboard:** Review model performance using accuracy, Macro-F1, and confusion matrices.
- **Interactive Interface:** Explore predictions and results through a Streamlit application.

## Models Used

### 1. TF-IDF + Logistic Regression

TF-IDF converts text into numerical feature vectors, while Logistic Regression classifies those vectors into sentiment categories. This serves as the traditional machine learning baseline.

### 2. Pre-trained Twitter-RoBERTa

We use the Hugging Face model:

`cardiffnlp/twitter-roberta-base-sentiment-latest`

This Transformer model produces sentiment predictions for negative, neutral, and positive classes.

### 3. Fine-tuned RoBERTa

The pre-trained model was fine-tuned using a stratified sample of 3,000 training records and evaluated on the complete 999-record validation dataset.

## Dataset

**Dataset:** Twitter Entity Sentiment Analysis

**Source:** [Kaggle — Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

The dataset contains social media text labeled with sentiment categories. The project uses the training and validation CSV files for preprocessing, baseline training, Transformer evaluation, and fine-tuning.

Expected dataset files:

```text
data/
├── twitter_training.csv
└── twitter_validation.csv
```

Place the downloaded dataset files in the `data/` directory before running the project.

## Methodology

1. Load and preprocess the Twitter sentiment dataset.
2. Clean text and prepare sentiment labels.
3. Train a TF-IDF + Logistic Regression baseline.
4. Evaluate the pre-trained Twitter-RoBERTa model.
5. Fine-tune RoBERTa using a smaller training subset.
6. Compare the models using accuracy, Macro-F1, and confusion matrices.
7. Integrate the models into a Streamlit application.
8. Apply separate rule-based logic to derive emotion, urgency, topic, and explanations.

### Architecture

```text
Social Media Text / CSV
          |
          v
   Text Preprocessing
          |
          v
  Model Selection
     /          \
    v            v
TF-IDF +       Twitter-
Logistic       RoBERTa
Regression       |
    |         Fine-tuned
    |          RoBERTa
     \           /
      v         v
   Sentiment Prediction
          |
          v
 Sentiment + Confidence
          |
          v
 Rule-Based Intelligence
          |
          v
 Emotion | Urgency | Topic
          |
          v
    Streamlit Results
```

**Note:** Sentiment and confidence are produced by the selected sentiment model. Emotion, urgency, topic, and explanations are derived separately using rule-based logic.

## Experimental Results

The following results were obtained from the project's validation experiments.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 98.00% | 0.9797 |
| Pre-trained Twitter-RoBERTa | 57.66% | 0.5718 |
| Fine-tuned RoBERTa | 70.27% | 0.7062 |

### Observations

- The TF-IDF + Logistic Regression baseline achieved the highest measured scores on this validation split.
- Fine-tuning improved RoBERTa's accuracy from 57.66% to 70.27%.
- The fine-tuned model improved Macro-F1 from 0.5718 to 0.7062.
- The baseline's unusually high score should be interpreted cautiously. Dataset overlap, duplicate records, label distribution, and possible dataset-specific patterns should be investigated before generalizing the result.

These results describe the evaluated validation dataset and do not guarantee equivalent performance on new social media posts.

## Technology Stack

- **Language:** Python
- **NLP / Transformers:** Hugging Face Transformers
- **Deep Learning:** PyTorch
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly
- **Web Application:** Streamlit
- **Pre-trained Model:** Twitter-RoBERTa

## Project Structure

```text
transformer-sentiment-analysis/
├── app.py
├── inference.py
├── preprocessing.py
├── intelligence.py
├── evaluate.py
├── finetune_small.py
├── test_model.py
├── check_model.py
├── requirements.txt
├── data/
│   ├── twitter_training.csv
│   └── twitter_validation.csv
├── results/
│   ├── metrics.json
│   ├── transformer_predictions.csv
│   └── roberta_finetuned_small/
└── README.md
```

The exact files and result artifacts may vary depending on which generated outputs are included in the repository. Large model artifacts and datasets may be kept outside GitHub if necessary.

## Installation and Setup

### Prerequisites

- Python 3.9–3.12
- Git
- Internet access for downloading the pre-trained model on the first run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/transformer-sentiment-analysis.git
cd transformer-sentiment-analysis
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the virtual environment's Python executable directly.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the dataset

Download the dataset from Kaggle and place the CSV files in the `data/` directory using the expected filenames.

### 5. Run the application

```bash
streamlit run app.py
```

Streamlit will display a local URL, typically:

`http://localhost:8501`

Open the URL in your browser to use the application.

## Evaluation Metrics

The project uses:

- **Accuracy:** Proportion of correctly classified validation examples.
- **Macro-F1:** Average F1-score across classes, giving each class equal weight.
- **Confusion Matrix:** Shows the relationship between true labels and predicted labels.
- **Class Probabilities:** Shows the model's estimated probability for each sentiment class.

Prediction confidence for an individual text is different from the overall accuracy measured across a validation dataset.

## Limitations

- The baseline's high accuracy needs further verification on independent data.
- Social media text can contain sarcasm, slang, emojis, and mixed sentiment.
- Dataset-specific labels may not perfectly represent general sentiment.
- Emotion, urgency, and topic detection use rule-based logic and may not handle every context correctly.
- Model performance can vary on unseen data and across different types of social media content.
- Fine-tuning on a small subset may limit generalization.

## Future Improvements

- Validate models on a separate, independently collected dataset.
- Investigate data leakage and duplicate-text overlap.
- Improve handling of sarcasm, negation, slang, and multilingual text.
- Explore stronger emotion and topic classification approaches.
- Optimize inference speed and model deployment.
- Add monitoring and reporting for real-world use.

## Contributors

Developed as a team project for a course-level Generative AI hackathon at the Department of Information Science and Engineering, Vidyavardhaka College of Engineering, Mysuru.

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Cardiff NLP Twitter-RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)


---

**Project:** Transformer-Based Social Media Sentiment Analysis  
**Application:** Interactive sentiment analysis with model comparison and derived intelligence
