def predict(model, data):
    # placeholder predictions
    return [model.get("model", "baseline") for _ in data]
