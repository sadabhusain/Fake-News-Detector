from sklearn.pipeline import Pipeline


def predict_news(model: Pipeline, text: str) -> dict:
    predicted_label = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_
    probability_map = {
        class_name: float(probability)
        for class_name, probability in zip(classes, probabilities)
    }

    return {
        "label": predicted_label,
        "confidence": probability_map[predicted_label],
        "probabilities": probability_map,
    }

