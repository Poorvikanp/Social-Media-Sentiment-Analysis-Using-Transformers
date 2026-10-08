import re


def _contains_any(text: str, keywords):
    lowered = text.lower()
    return any(k in lowered for k in keywords)


def emotion(text: str) -> str:
    if _contains_any(text, ["love", "great", "amazing", "awesome", "excellent", "happy", "glad", "wonderful", "perfect", "thank", "best"]):
        return "Joy"
    if _contains_any(text, ["angry", "terrible", "hate", "worst", "furious", "annoyed", "frustrated", "awful", "horrible", "mad"]):
        return "Anger"
    if _contains_any(text, ["sad", "disappointed", "upset", "unhappy", "depressed", "sorry", "regret"]):
        return "Sadness"
    if _contains_any(text, ["confused", "unclear", "don't understand", "unclear", "lost", "help", "question"]):
        return "Confusion"
    return "Neutral / Unknown"


def urgency(text: str) -> str:
    if _contains_any(text, ["urgent", "immediately", "asap", "emergency", "critical", "right now", "now"]):
        return "High"
    if _contains_any(text, ["soon", "quickly", "issue", "problem", "broken", "fix", "help"]):
        return "Medium"
    return "Low"


def topic(text: str) -> str:
    if _contains_any(text, ["price", "cost", "payment", "billing", "charge", "free", "cheap", "expensive", "refund", "money"]):
        return "Finance / Pricing"
    if _contains_any(text, ["app", "website", "login", "error", "crash", "bug", "account", "password", "update", "tech", "loading"]):
        return "Technical"
    if _contains_any(text, ["delivery", "order", "shipping", "shipment", "arrived", "package", "late", "delay", "tracking"]):
        return "Delivery"
    if _contains_any(text, ["product", "item", "device", "quality", "design", "material", "works", "broken"]):
        return "Product"
    if _contains_any(text, ["service", "support", "customer", "staff", "representative", "agent", "response"]):
        return "Customer Service"
    return "General"


def explanation(text: str, sentiment: str, emotion: str, urgency: str, topic: str) -> str:
    parts = []
    parts.append(f"Detected sentiment: {sentiment}.")

    if emotion != "Neutral / Unknown":
        parts.append(f"Emotion indicator suggests {emotion.lower()}.")

    if urgency != "Low":
        parts.append(f"Urgency level is {urgency.lower()}.")

    if topic != "General":
        parts.append(f"Topic appears to be {topic.lower()}.")

    return " ".join(parts)


def analyze(text: str, sentiment: str) -> dict:
    detected_emotion = emotion(text)
    detected_urgency = urgency(text)
    detected_topic = topic(text)
    detected_explanation = explanation(text, sentiment, detected_emotion, detected_urgency, detected_topic)

    return {
        "emotion": detected_emotion,
        "urgency": detected_urgency,
        "topic": detected_topic,
        "explanation": detected_explanation,
    }
