import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from inference import predict_batch
from preprocessing import (
    clean_baseline,
    clean_transformer,
    load_twitter,
)


LABELS = [
    "negative",
    "neutral",
    "positive",
]

os.makedirs("results", exist_ok=True)


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

train = load_twitter(
    "data/twitter_training.csv"
)

val = load_twitter(
    "data/twitter_validation.csv"
)


print(
    f"Train rows: {len(train)} | "
    f"Validation rows: {len(val)}"
)

print(
    "\nValidation label counts:"
)

print(
    val["label"].value_counts()
)


metrics = {}


# --------------------------------------------------
# Evaluation helper
# --------------------------------------------------

def report(name, y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    metrics[name] = {
        "accuracy": round(
            accuracy,
            4
        ),
        "macro_f1": round(
            macro_f1,
            4
        ),
    }

    print(
        f"\n=== {name} ==="
    )

    print(
        f"Accuracy: {accuracy:.3f}"
    )

    print(
        f"Macro-F1: {macro_f1:.3f}"
    )

    print(
        classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            zero_division=0,
        )
    )

    print(
        "Confusion matrix "
        "(rows=true, cols=pred):"
    )

    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=LABELS,
        )
    )


# --------------------------------------------------
# BASELINE
# TF-IDF + Logistic Regression
# --------------------------------------------------

print(
    "\nTraining baseline..."
)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)

X_train = vectorizer.fit_transform(
    train["content"].map(
        clean_baseline
    )
)

X_val = vectorizer.transform(
    val["content"].map(
        clean_baseline
    )
)

classifier = LogisticRegression(
    max_iter=1000
)

classifier.fit(
    X_train,
    train["label"]
)

baseline_predictions = classifier.predict(
    X_val
)

report(
    "Baseline: TF-IDF + Logistic Regression",
    val["label"],
    baseline_predictions,
)


# --------------------------------------------------
# MAIN MODEL
# Pre-trained RoBERTa
# --------------------------------------------------

print(
    "\nRunning Transformer..."
)

predictions = predict_batch(
    val["content"]
        .map(clean_transformer)
        .tolist(),
    batch_size=16,
)

val["transformer_pred"] = [
    p["sentiment"]
    for p in predictions
]

val["transformer_conf"] = [
    round(
        p["confidence"],
        4
    )
    for p in predictions
]

report(
    "Transformer: pre-trained RoBERTa",
    val["label"],
    val["transformer_pred"],
)


# --------------------------------------------------
# Save predictions
# --------------------------------------------------

val[
    [
        "content",
        "label",
        "transformer_pred",
        "transformer_conf",
    ]
].to_csv(
    "results/transformer_predictions.csv",
    index=False,
)


# --------------------------------------------------
# Save metrics
# --------------------------------------------------

with open(
    "results/metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )


print(
    "\nSaved:"
)

print(
    "results/metrics.json"
)

print(
    "results/transformer_predictions.csv"
)

print(
    "\nFinal metrics:"
)

print(
    json.dumps(
        metrics,
        indent=2
    )
)