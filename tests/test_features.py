from my_project.features import build_features


def test_build_features_passthrough():
    rows = [{"a": 1}]
    assert build_features(rows) == rows
