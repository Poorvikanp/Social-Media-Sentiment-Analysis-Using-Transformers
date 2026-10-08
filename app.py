import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from inference import predict as pretrained_predict
from inference import predict_batch as pretrained_predict_batch
from inference_finetuned import predict as finetuned_predict
from inference_finetuned import predict_batch as finetuned_predict_batch
from intelligence import analyze
from preprocessing import clean_transformer

st.set_page_config(
    page_title="SocialPulse AI",
    page_icon="📊",
    layout="wide",
)

# ---------------------------
# Model selection
# ---------------------------

MODEL_CHOICES = {
    "Pre-trained RoBERTa": "pretrained",
    "Fine-tuned RoBERTa": "finetuned",
}


def _get_predict_fn():
    mode = st.session_state.get("model_mode", "pretrained")
    if mode == "finetuned":
        return finetuned_predict, finetuned_predict_batch, "Fine-tuned RoBERTa"
    return pretrained_predict, pretrained_predict_batch, "Pre-trained RoBERTa"


# ---------------------------
# Styling / helpers
# ---------------------------

SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "neutral": "#95a5a6",
    "negative": "#e74c3c",
}

RESULTS_DIR = "results"
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")
PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "transformer_predictions.csv")
FINETUNED_DIR = os.path.join(RESULTS_DIR, "roberta_finetuned_small")
FINETUNED_METRICS_PATH = os.path.join(FINETUNED_DIR, "metrics.json")
FINETUNED_PREDICTIONS_PATH = os.path.join(FINETUNED_DIR, "predictions.csv")


def _saved_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def _saved_predictions():
    if not os.path.exists(PREDICTIONS_PATH):
        return None
    return pd.read_csv(PREDICTIONS_PATH)


def _finetuned_metrics():
    if not os.path.exists(FINETUNED_METRICS_PATH):
        return None
    with open(FINETUNED_METRICS_PATH, "r") as f:
        return json.load(f)


def _finetuned_predictions():
    if not os.path.exists(FINETUNED_PREDICTIONS_PATH):
        return None
    return pd.read_csv(FINETUNED_PREDICTIONS_PATH)


# ---------------------------
# Pages
# ---------------------------

def page_single_analysis():
    st.title("📊 SocialPulse AI")
    st.subheader("Social Media Sentiment Intelligence")

    predict_fn, _, model_label = _get_predict_fn()

    # Demo examples
    demo_examples = [
        "I absolutely love this product! It works perfectly.",
        "The service was terrible and I am extremely disappointed.",
        "The package arrived today.",
        "The application keeps crashing and I need help immediately.",
        "Great experience overall, would recommend to friends.",
        "I hate waiting forever for a simple response.",
    ]

    with st.expander("🎯 Demo Examples", expanded=False):
        cols = st.columns(3)
        for i, ex in enumerate(demo_examples):
            if cols[i % 3].button(ex[:40] + "…", key=f"demo_{i}"):
                st.session_state["input_text"] = ex

    input_text = st.text_area(
        "Enter a social media post",
        value=st.session_state.get("input_text", ""),
        height=120,
        placeholder="Paste a tweet, review, or social media post here...",
    )

    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("Please enter some text first.")
            return

        with st.spinner("Analyzing…"):
            try:
                result = predict_fn(input_text)
            except Exception as e:
                st.error(f"Model error: {e}")
                return

        sentiment = result["sentiment"]
        confidence = result["confidence"]
        scores = result["scores"]

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Sentiment", sentiment.upper())
        col2.metric("Confidence", f"{confidence * 100:.1f}%")
        col3.metric("Model", model_label)

        # Probability chart
        score_df = pd.DataFrame(
            [{"Class": k.title(), "Probability": v} for k, v in scores.items()]
        )
        fig = px.bar(
            score_df,
            x="Class",
            y="Probability",
            color="Class",
            color_discrete_map={
                "Negative": "#e74c3c",
                "Neutral": "#95a5a6",
                "Positive": "#2ecc71",
            },
            range_y=[0, 1],
            height=300,
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Probability")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Sentiment and confidence are produced by the selected RoBERTa model.")

        # Derived Intelligence
        derived = analyze(input_text, sentiment)

        st.markdown("---")
        st.markdown("### 🧠 Derived Intelligence")
        st.caption("Rule-based analysis based on keywords and patterns in the text.")

        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        dcol1.metric("Emotion", derived["emotion"])
        dcol2.metric("Urgency", derived["urgency"])
        dcol3.metric("Topic", derived["topic"])
        dcol4.metric("Sentiment", sentiment.title())

        st.info(derived["explanation"])


def page_batch_analysis():
    st.title("📁 Batch Analysis")
    st.subheader("Upload a CSV of social media posts")

    _, predict_batch_fn, model_label = _get_predict_fn()

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV file containing a `content` column.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    text_col = None
    for candidate in ["content", "text", "tweet", "message"]:
        if candidate in df.columns:
            text_col = candidate
            break

    if text_col is None:
        st.error(f"No text column found. Expected one of: content, text, tweet, message. Found: {list(df.columns)}")
        return

    texts = df[text_col].dropna().astype(str).tolist()
    if not texts:
        st.warning("No text rows found.")
        return

    with st.spinner(f"Predicting {len(texts)} rows with {model_label}…"):
        try:
            preds = predict_batch_fn([clean_transformer(t) for t in texts], batch_size=16)
        except Exception as e:
            st.error(f"Batch prediction error: {e}")
            return

    out = pd.DataFrame({"text": texts})
    out["sentiment"] = [p["sentiment"] for p in preds]
    out["confidence"] = [p["confidence"] for p in preds]

    st.markdown("### Results")
    st.dataframe(out, use_container_width=True)

    st.markdown("### Distribution")
    fig = px.histogram(
        out,
        x="sentiment",
        color="sentiment",
        color_discrete_map={
            "positive": "#2ecc71",
            "neutral": "#95a5a6",
            "negative": "#e74c3c",
        },
    )
    fig.update_layout(xaxis_title="", yaxis_title="Count", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.metric("Average confidence", f"{out['confidence'].mean():.3f}")


def page_results():
    st.title("📈 Experiment Results")
    st.subheader("Model comparison on the validation split")

    metrics = _saved_metrics()
    finetuned_metrics = _finetuned_metrics()

    if not metrics:
        st.warning("No saved metrics found. Run `python evaluate.py` first.")
        return

    # Build comparison data
    rows = []
    for name, vals in metrics.items():
        rows.append(
            {
                "Model": name,
                "Accuracy": vals.get("accuracy"),
                "Macro-F1": vals.get("macro_f1"),
            }
        )

    # Add fine-tuned RoBERTa if available
    if finetuned_metrics:
        rows.append(
            {
                "Model": "Fine-tuned RoBERTa (3k samples)",
                "Accuracy": finetuned_metrics.get("accuracy"),
                "Macro-F1": finetuned_metrics.get("macro_f1"),
            }
        )

    res_df = pd.DataFrame(rows)
    st.dataframe(res_df, use_container_width=True)

    # Fine-tuned detailed metrics
    if finetuned_metrics:
        st.markdown("### Fine-tuned RoBERTa Detailed Metrics")
        finetuned_detail = pd.DataFrame(
            [
                {"Metric": "Accuracy", "Value": finetuned_metrics.get("accuracy")},
                {"Metric": "Macro-F1", "Value": finetuned_metrics.get("macro_f1")},
                {"Metric": "Macro-Precision", "Value": finetuned_metrics.get("macro_precision")},
                {"Metric": "Macro-Recall", "Value": finetuned_metrics.get("macro_recall")},
                {"Metric": "Training Samples", "Value": finetuned_metrics.get("training_samples")},
                {"Metric": "Validation Samples", "Value": finetuned_metrics.get("validation_samples")},
                {"Metric": "Epochs", "Value": finetuned_metrics.get("epochs")},
                {"Metric": "Training Time (seconds)", "Value": finetuned_metrics.get("training_time_seconds")},
                {"Metric": "Early Stopping Triggered", "Value": finetuned_metrics.get("early_stopped")},
            ]
        )
        st.dataframe(finetuned_detail, use_container_width=True)

        # Fine-tuned confusion matrix
        finetuned_preds = _finetuned_predictions()
        if finetuned_preds is not None and "true_label" in finetuned_preds.columns and "predicted_label" in finetuned_preds.columns:
            from sklearn.metrics import confusion_matrix

            y_true = finetuned_preds["true_label"]
            y_pred = finetuned_preds["predicted_label"]

            labels = [0, 1, 2]
            cm = confusion_matrix(y_true, y_pred, labels=labels)

            st.markdown("### Fine-tuned RoBERTa Confusion Matrix")
            st.caption("Rows = true labels, Columns = predicted labels")
            cm_df = pd.DataFrame(
                cm,
                index=["Negative", "Neutral", "Positive"],
                columns=["Negative", "Neutral", "Positive"],
            )
            st.dataframe(cm_df, use_container_width=True)

    # Notes
    st.markdown(
        """
        ### Notes
        - **TF-IDF + Logistic Regression** was trained on the project dataset and evaluated on the validation split.
        - **Pre-trained Twitter RoBERTa** result represents direct inference without fine-tuning on this dataset.
        - **Fine-tuned RoBERTa** was trained on 3,000 samples from the project dataset with early stopping based on validation Macro-F1.
        - On this Twitter Entity Sentiment validation split, TF-IDF with Logistic Regression achieved higher measured performance than both direct and fine-tuned RoBERTa.
        - Fine-tuning substantially improved RoBERTa, increasing accuracy from 57.66% to 70.27% and Macro-F1 from 57.18% to 70.62%.
        - These results illustrate the impact of representation and domain adaptation.
        """
    )

    # Pre-trained RoBERTa confusion matrix if available
    preds = _saved_predictions()
    if preds is not None and "transformer_pred" in preds.columns:
        from sklearn.metrics import confusion_matrix

        y_true = preds["label"]
        y_pred = preds["transformer_pred"]

        labels = ["negative", "neutral", "positive"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        st.markdown("### Pre-trained RoBERTa Confusion Matrix")
        st.caption("Rows = true labels, Columns = predicted labels")
        cm_df = pd.DataFrame(cm, index=[l.title() for l in labels], columns=[l.title() for l in labels])
        st.dataframe(cm_df, use_container_width=True)


def page_about():
    st.title("ℹ️ About SocialPulse AI")
    st.markdown(
        """
        **Project:** Sentiment Analysis of Social Media Text Using Transformers

        **Product:** SocialPulse AI

        **Dataset:** Twitter Entity Sentiment (Kaggle)

        **Models compared:**
        1. TF-IDF + Logistic Regression (baseline)
        2. Pre-trained Twitter RoBERTa (transformer)
        3. Fine-tuned RoBERTa (3,000 samples, 1 epoch, early stopping)

        **UI features:**
        - Single-text sentiment prediction
        - Rule-based derived intelligence (emotion, urgency, topic)
        - Batch CSV analysis
        - Experiment results page

        **Note:** Derived intelligence (emotion, urgency, topic) is rule-based and NOT a Transformer prediction.
        """
    )


# ---------------------------
# Main
# ---------------------------

def main():
    with st.sidebar:
        st.title("SocialPulse AI")
        model_choice = st.radio(
            "Model Selection",
            list(MODEL_CHOICES.keys()),
            index=0,
            key="model_selection",
        )
        st.session_state["model_mode"] = MODEL_CHOICES[model_choice]

        st.markdown("---")
        page = st.radio(
            "Navigate",
            [
                "Single Analysis",
                "Batch Analysis",
                "Results",
                "About",
            ],
            key="page_navigation",
        )

    if page == "Single Analysis":
        page_single_analysis()
    elif page == "Batch Analysis":
        page_batch_analysis()
    elif page == "Results":
        page_results()
    else:
        page_about()


if __name__ == "__main__":
    main()
