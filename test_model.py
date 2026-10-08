from inference import predict, VALID


tests = [
    "I absolutely love this product!",
    "The package arrived today.",
    "The service was terrible.",
]


for text in tests:

    result = predict(text)

    assert result["sentiment"] in VALID

    print(
        f"{result['sentiment']:9} "
        f"{result['confidence']:.3f}  "
        f"{text}"
    )


print("OK")