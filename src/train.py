from my_project.features import build_features


def train(data):
    features = build_features(data)
    # placeholder model object
    return {"model": "baseline", "rows": len(features)}


if __name__ == "__main__":
    sample = [{"x": 1}, {"x": 2}]
    model = train(sample)
    print(model)
