import pandas as pd

from src.utils import TARGET_COL, build_features


def predict(model, data, target_col=TARGET_COL):
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if target_col in df.columns:
        X, _ = build_features(df, target_col=target_col)
    else:
        X = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return model.predict(X)
