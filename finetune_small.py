import os
import time
import json
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ============================================================
# 1. Configuration
# ============================================================

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

TRAIN_PATH = "data/twitter_training.csv"
VAL_PATH = "data/twitter_validation.csv"

OUTPUT_DIR = "results/roberta_finetuned_small"

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

# Small controlled experiment
TRAIN_SAMPLES = 3000
RANDOM_SEED = 42

COLS = ["tweetid", "entity", "sentiment", "content"]


# ============================================================
# 2. Load Dataset
# ============================================================

def load_data(path):
    df = pd.read_csv(
        path,
        header=None,
        names=COLS
    )

    df = df.dropna(subset=["content"])
    df = df.drop_duplicates(subset="content")
    df["sentiment"] = df["sentiment"].str.lower()
    df["sentiment"] = df["sentiment"].replace({
        "irrelevant": "neutral"
    })
    df["label"] = df["sentiment"].map(LABEL2ID)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    return df[["content", "label"]].reset_index(drop=True)


print("Loading datasets...")
train_df = load_data(TRAIN_PATH)
val_df = load_data(VAL_PATH)

# Sample exactly 3000 training samples
train_df = train_df.sample(n=min(TRAIN_SAMPLES, len(train_df)), random_state=RANDOM_SEED).reset_index(drop=True)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

print("\nTraining label distribution:")
print(train_df["label"].value_counts().sort_index())

print("\nValidation label distribution:")
print(val_df["label"].value_counts().sort_index())


# ============================================================
# 3. Convert to Hugging Face Dataset
# ============================================================

train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
val_dataset = Dataset.from_pandas(val_df, preserve_index=False)


# ============================================================
# 4. Tokenizer
# ============================================================

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(
        examples["content"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


print("Tokenizing...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(["content"])
val_dataset = val_dataset.remove_columns(["content"])

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")


# ============================================================
# 5. Model
# ============================================================

print("\nLoading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)


# ============================================================
# 6. Metrics
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")
    precision = precision_score(labels, predictions, average="macro")
    recall = recall_score(labels, predictions, average="macro")
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": precision,
        "macro_recall": recall,
    }


# ============================================================
# 7. Training Arguments
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=250,
    logging_strategy="steps",
    logging_steps=50,
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
# 8. Trainer with Early Stopping
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
)


# ============================================================
# 9. Pre-training summary
# ============================================================

total_steps = len(train_dataset) // (
    training_args.per_device_train_batch_size
    * max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)
)

print("\n" + "=" * 50)
print("FINE-TUNING CONFIGURATION")
print("=" * 50)
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Total training steps: {total_steps}")
print(f"Expected configuration: CPU")
print("=" * 50 + "\n")


# ============================================================
# 10. Fine-tuning
# ============================================================

print("\n" + "=" * 50)
print("STARTING RoBERTa FINE-TUNING")
print("=" * 50 + "\n")

start_time = time.time()

train_result = trainer.train()

end_time = time.time()
training_time = end_time - start_time

print(f"\nTraining completed in {training_time:.2f} seconds")


# ============================================================
# 11. Evaluation
# ============================================================

print("\n" + "=" * 50)
print("EVALUATING FINE-TUNED MODEL")
print("=" * 50 + "\n")

eval_results = trainer.evaluate()
print("\nEvaluation Results:")
print(eval_results)


# ============================================================
# 12. Detailed Report
# ============================================================

predictions = trainer.predict(val_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids

print("\nClassification Report:")
print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=["negative", "neutral", "positive"],
        digits=4
    )
)

print("\nConfusion Matrix (rows=true, cols=pred):")
cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1, 2])
print(cm)


# ============================================================
# 13. Early stopping status
# ============================================================

early_stopped = getattr(trainer.state, "stopped_early", False)
print(f"\nEarly stopping triggered: {early_stopped}")


# ============================================================
# 14. Save model
# ============================================================

print("\nSaving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Model saved to:", OUTPUT_DIR)


# ============================================================
# 15. Save metrics
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

metrics = {
    "training_samples": int(len(train_dataset)),
    "validation_samples": int(len(val_dataset)),
    "epochs": 1,
    "accuracy": float(eval_results.get("eval_accuracy", 0.0)),
    "macro_f1": float(eval_results.get("eval_macro_f1", 0.0)),
    "macro_precision": float(eval_results.get("eval_macro_precision", 0.0)),
    "macro_recall": float(eval_results.get("eval_macro_recall", 0.0)),
    "training_time_seconds": float(training_time),
    "early_stopped": bool(early_stopped),
    "model_name": MODEL_NAME,
    "label_mapping": ID2LABEL,
}

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("Metrics saved to:", os.path.join(OUTPUT_DIR, "metrics.json"))


# ============================================================
# 16. Save predictions
# ============================================================

val_texts = val_df["content"].tolist()
pred_df = pd.DataFrame({
    "content": val_texts,
    "true_label": true_labels,
    "predicted_label": predicted_labels,
})

# Add per-class scores if available
if hasattr(predictions, "predictions") and len(predictions.predictions.shape) == 2:
    pred_df["score_negative"] = predictions.predictions[:, 0]
    pred_df["score_neutral"] = predictions.predictions[:, 1]
    pred_df["score_positive"] = predictions.predictions[:, 2]

pred_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
print("Predictions saved to:", os.path.join(OUTPUT_DIR, "predictions.csv"))


# ============================================================
# 17. Final summary
# ============================================================

print("\n" + "=" * 50)
print("FINE-TUNING COMPLETE")
print("=" * 50)
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Training time: {training_time:.2f} seconds")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro-F1: {metrics['macro_f1']:.4f}")
print(f"Macro-Precision: {metrics['macro_precision']:.4f}")
print(f"Macro-Recall: {metrics['macro_recall']:.4f}")
print(f"Early stopped: {early_stopped}")
print(f"Model saved to: {OUTPUT_DIR}")
print("=" * 50)
