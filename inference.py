from functools import lru_cache
from transformers import pipeline


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

VALID = {"negative", "neutral", "positive"}


@lru_cache(maxsize=1)
def get_pipe():
    """
    Load the Transformer once and reuse it.
    """
    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        top_k=None,
        truncation=True,
        max_length=512,
    )


def _scores(raw):
    """
    Convert Transformer output into:
    {
        "negative": probability,
        "neutral": probability,
        "positive": probability
    }
    """

    # Different Transformers versions can return
    # list or list-of-lists.
    if raw and isinstance(raw[0], list):
        raw = raw[0]

    return {
        r["label"].strip().lower(): float(r["score"])
        for r in raw
    }


def _pack(scores):
    """
    Select the class having the highest probability.
    """

    label = max(scores, key=scores.get)

    return {
        "sentiment": label,
        "confidence": scores[label],
        "scores": scores,
    }


def predict(text: str):
    """
    Predict sentiment for one text.
    """

    raw = get_pipe()(text)

    return _pack(_scores(raw))


def predict_batch(texts, batch_size: int = 16):
    """
    Predict sentiment for multiple texts.
    """

    outputs = get_pipe()(list(texts), batch_size=batch_size)

    return [_pack(_scores(output)) for output in outputs]