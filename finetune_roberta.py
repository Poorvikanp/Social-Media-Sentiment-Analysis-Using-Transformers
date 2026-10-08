import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ============================================================
# 1. Configuration
# ============================================================

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

TRAIN_PATH = "data/twitter_training.csv"
VAL_PATH = "data/twitter_validation.csv"

OUTPUT_DIR = "results/roberta_finetuned"

LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# ============================================================
# 2. Load Dataset
# ============================================================

COLS = ["tweetid", "entity", "sentiment", "content"]


def load_data(path):
    df = pd.read_csv(
        path,
        header=None,
        names=COLS
    )

    # Remove rows without text
    df = df.dropna(subset=["content"])

    # Remove duplicate tweets
    df = df.drop_duplicates(subset="content")

    # Convert labels to lowercase
    df["sentiment"] = df["sentiment"].str.lower()

    # Dataset contains "irrelevant".
    # For our 3-class problem, map it to neutral.
    df["sentiment"] = df["sentiment"].replace({
        "irrelevant": "neutral"
    })

    # Convert text labels to integers
    df["label"] = df["sentiment"].map(LABEL2ID)

    # Remove anything that could not be mapped
    df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)

    return df[["content", "label"]].reset_index(drop=True)


train_df = load_data(TRAIN_PATH)
val_df = load_data(VAL_PATH)

print("\n================ DATASET =================")
print("Training samples :", len(train_df))
print("Validation samples:", len(val_df))

print("\nTraining label distribution:")
print(train_df["label"].value_counts().sort_index())

print("\nValidation label distribution:")
print(val_df["label"].value_counts().sort_index())


# ============================================================
# 3. Convert Pandas → Hugging Face Dataset
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df,
    preserve_index=False
)


# ============================================================
# 4. Load Tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 5. Tokenization
# ============================================================

def tokenize_function(examples):

    return tokenizer(
        examples["content"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


print("Tokenizing training data...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

print("Tokenizing validation data...")

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True
)


# Remove original text column
train_dataset = train_dataset.remove_columns(["content"])
val_dataset = val_dataset.remove_columns(["content"])


# Tell Hugging Face that "label" is the target
train_dataset = train_dataset.rename_column(
    "label",
    "labels"
)

val_dataset = val_dataset.rename_column(
    "label",
    "labels"
)


# ============================================================
# 6. Load Pre-trained RoBERTa
# ============================================================

print("\nLoading RoBERTa model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)


# ============================================================
# 7. Evaluation Metrics
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro"
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


# ============================================================
# 8. Training Configuration
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    # CPU-safe small experiment
    num_train_epochs=1,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    # Evaluate after each epoch
    eval_strategy="epoch",

    # Save after each epoch
    save_strategy="epoch",

    logging_strategy="steps",
    logging_steps=100,

    learning_rate=2e-5,

    weight_decay=0.01,

    load_best_model_at_end=True,

    metric_for_best_model="macro_f1",

    greater_is_better=True,

    report_to="none",

    fp16=False,

    dataloader_num_workers=0,
)


# ============================================================
# 9. Trainer
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    tokenizer=tokenizer,

    compute_metrics=compute_metrics,
)


# ============================================================
# 10. Fine-tuning
# ============================================================

print("\n============================================")
print("STARTING RoBERTa FINE-TUNING")
print("============================================\n")

trainer.train()


# ============================================================
# 11. Evaluation
# ============================================================

print("\n============================================")
print("EVALUATING FINE-TUNED RoBERTa")
print("============================================\n")

results = trainer.evaluate()

print("\nEvaluation Results:")
print(results)


# ============================================================
# 12. Detailed Classification Report
# ============================================================

predictions = trainer.predict(val_dataset)

predicted_labels = np.argmax(
    predictions.predictions,
    axis=-1
)

true_labels = predictions.label_ids

print("\nClassification Report:")
print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=[
            "negative",
            "neutral",
            "positive"
        ],
        digits=4
    )
)


# ============================================================
# 13. Save Model
# ============================================================

print("\nSaving fine-tuned model...")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n============================================")
print("DONE")
print("Model saved to:", OUTPUT_DIR)
print("============================================")