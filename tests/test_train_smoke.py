from my_project.train import train


def test_train_smoke():
    model = train([{"x": 1}, {"x": 2}])
    assert model["model"] == "baseline"
    assert model["rows"] == 2
