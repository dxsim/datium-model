from src.utils import build_features


def test_build_features_minimal():
    rows = [
        {"x": 1, "Sold_Amount": 10},
        {"x": 2, "Sold_Amount": 20},
    ]
    X, y = build_features(rows)
    assert X.shape == (2, 1)
    assert y.shape[0] == 2
