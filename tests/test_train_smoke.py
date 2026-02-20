from src.train import train


def test_train_smoke():
    rows = [
        {"x1": 1, "x2": 2, "Sold_Amount": 100},
        {"x1": 2, "x2": 1, "Sold_Amount": 120},
        {"x1": 3, "x2": 1, "Sold_Amount": 140},
        {"x1": 4, "x2": 3, "Sold_Amount": 180},
        {"x1": 5, "x2": 5, "Sold_Amount": 220},
        {"x1": 6, "x2": 8, "Sold_Amount": 260},
    ]
    model = train(
        rows,
        n_splits=3,
        lgbm_params={
            "objective": "regression",
            "verbosity": -1,
            "random_state": 42,
            "n_estimators": 20,
            "learning_rate": 0.1,
            "num_leaves": 15,
        },
    )
    assert model["model"] == "lightgbm_regressor"
    assert model["rows"] == 6
    assert model["cv"]["n_splits"] == 3
    assert "mae" in model["cv"]["mean_metrics"]
    assert "rmse" in model["cv"]["mean_metrics"]
