from transformers import AutoConfig

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

cfg = AutoConfig.from_pretrained(MODEL)

print("Model:", MODEL)
print("id2label:", cfg.id2label)