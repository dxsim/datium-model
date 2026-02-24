import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


DROP_COLS = [
    "AvgWholesale",
    "AvgRetail",
    "GoodWholesale",
    "GoodRetail",
    "TradeMin",
    "TradeMax",
    "PrivateMax",
]

CORE_NUMERIC_COLS = ["NewPrice", "Sold_Amount", "Age_Comp_Months", "KM"]
TARGET_COL = "Sold_Amount"

CAT_COLS = [
    "MakeCode",
    "FamilyCode",
    "BodyStyleDescription",
    "DriveCode",
    "GearTypeDescription",
    "GearLocationDescription",
    "FuelTypeDescription",
    "InductionDescription",
    "BuildCountryOriginDescription",
]

NUM_COLS = [
    "GearNum",
    "DoorNum",
    "EngineSize",
    "Cylinders",
    "FuelCapacity",
    "NewPrice",
    "WarrantyYears",
    "WarrantyKM",
    "KM",
]

GEAR_TYPE_MAP = {
    "Sports Automatic Single Clutch": "Sports Automatic",
    "Sports Automatic Dual Clutch": "Sports Automatic",
    "Seq. Manual Auto-Single Clutch": "Manual",
    "Manual Auto-clutch - H Pattern": "Manual",
}

INDUCTION_MAP = {
    "Turbo Intercooled": "Turbo",
    "Supercharged Intercooled": "Supercharged",
    "Twin Turbo Intercooled": "Turbo",
}

FUEL_MAP = {
    "Petrol - Unleaded ULP": "Petrol",
    "Petrol - Premium ULP": "Petrol",
    "Petrol or LPG (Dual)": "Petrol or LPG",
}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_project_root(start):
    start = Path(start).resolve()
    for p in [start] + list(start.parents):
        if (p / "data" / "raw").exists():
            return p
    raise FileNotFoundError("Could not find project root containing data/raw")


def load_raw_datasets(train_path, test_path):
    train_df = pd.read_csv(train_path, sep="\t", encoding="utf-8-sig", low_memory=False)
    test_df = pd.read_csv(test_path, sep="\t", encoding="utf-8-sig", low_memory=False)
    train_df["dataset_split"] = "train"
    test_df["dataset_split"] = "test"
    return pd.concat([train_df, test_df], ignore_index=True)


def filter_dataset(
    df,
    missing_threshold_pct=5.0,
    max_age_months=246,
    drop_zero_target=True,
    drop_target_gt_newprice=True,
):
    out = df.copy()
    out = out.drop(columns=DROP_COLS, errors="ignore")

    missing_pct = (out.isna().mean() * 100).sort_values(ascending=False)
    keep_cols = missing_pct[missing_pct <= float(missing_threshold_pct)].index.tolist()
    out = out[keep_cols]

    for col in CORE_NUMERIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if drop_zero_target and TARGET_COL in out.columns:
        out = out[out[TARGET_COL] != 0]

    if drop_target_gt_newprice and {"Sold_Amount", "NewPrice"}.issubset(out.columns):
        out = out[out["Sold_Amount"] <= out["NewPrice"]]

    if {"Age_Comp_Months", "NewPrice"}.issubset(out.columns):
        out = out[
            (out["Age_Comp_Months"] >= 0)
            & (out["Age_Comp_Months"] <= max_age_months)
            & (out["NewPrice"] >= 0)
        ]
    return out.reset_index(drop=True)


def _fill_mode(df, col):
    if col in df.columns:
        mode = df[col].mode(dropna=True)
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])
    return df


def _fill_median(df, col):
    if col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        df[col] = series.fillna(series.median())
    return df


def prepare_model_dataframe(df):
    out = df.copy()
    for col in CAT_COLS:
        out = _fill_mode(out, col)
    for col in NUM_COLS:
        out = _fill_median(out, col)

    for col in [TARGET_COL, "NewPrice", "KM"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out = out[out[col].notna()]

    if "Sold_Date" in out.columns:
        out["Sold_Date"] = pd.to_datetime(out["Sold_Date"], errors="coerce")
    if "Compliance_Date" in out.columns:
        out["Compliance_Date"] = pd.to_datetime(
            out["Compliance_Date"], format="%m/%Y", errors="coerce"
        )

    if {"Sold_Date", "Compliance_Date"}.issubset(out.columns):
        out = out[out["Sold_Date"].notna() & out["Compliance_Date"].notna()]
        out["AgeDays"] = (out["Sold_Date"] - out["Compliance_Date"]).dt.days
    else:
        out["AgeDays"] = pd.to_numeric(out.get("Age_Comp_Months"), errors="coerce") * 30

    if "GearTypeDescription" in out.columns:
        out["GearTypeDescription"] = out["GearTypeDescription"].replace(GEAR_TYPE_MAP)
    if "InductionDescription" in out.columns:
        out["InductionDescription"] = out["InductionDescription"].replace(INDUCTION_MAP)
    if "FuelTypeDescription" in out.columns:
        out["FuelTypeDescription"] = out["FuelTypeDescription"].replace(FUEL_MAP)

    out = out.drop(columns=["Sold_Date", "Compliance_Date"], errors="ignore")
    out = out.dropna(subset=[TARGET_COL, "AgeDays"]).reset_index(drop=True)
    out["AgeDays"] = out["AgeDays"].astype(float)
    return out


def label_encode_columns(df, columns):
    out = df.copy()
    label_maps = {}
    for col in columns:
        if col not in out.columns:
            continue
        values = out[col].astype(str)
        labels = sorted(values.unique().tolist())
        mapping = {label: idx for idx, label in enumerate(labels)}
        out[col] = values.map(mapping).astype(int)
        label_maps[col] = mapping
    return out, label_maps


def to_training_matrix(df, label_map_path=None, log_target=True, log_newprice=True):
    out = df.copy()
    cat_cols = out.select_dtypes(include=["object", "bool"]).columns.tolist()
    out, label_maps = label_encode_columns(out, cat_cols)

    if log_target and TARGET_COL in out.columns:
        out[TARGET_COL] = np.log1p(pd.to_numeric(out[TARGET_COL], errors="coerce").clip(lower=0))

    if log_newprice and "NewPrice" in out.columns:
        out["NewPrice"] = np.log1p(pd.to_numeric(out["NewPrice"], errors="coerce").clip(lower=0))

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna().reset_index(drop=True)

    if label_map_path is not None:
        label_map_path = Path(label_map_path)
        label_map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump(label_maps, f, indent=2, sort_keys=True)
    return out, label_maps


def build_training_matrix_from_raw(
    project_root=None,
    missing_threshold_pct=5.0,
    max_age_months=246,
):
    project_root = find_project_root(project_root or Path.cwd())
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    preprocessing_dir = project_root / "notebooks" / "preprocessing"
    processed_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_dir.mkdir(parents=True, exist_ok=True)

    train_path = raw_dir / "DatiumTrain.rpt"
    test_path = raw_dir / "DatiumTest.rpt"
    filtered_path = processed_dir / "dataset_filtered.csv"
    model_path = processed_dir / "dataset_model.csv"
    matrix_path = processed_dir / "dataset_training_matrix.csv"
    label_map_path = preprocessing_dir / "label_encoding_map.json"

    raw_df = load_raw_datasets(train_path=train_path, test_path=test_path)
    filtered_df = filter_dataset(
        raw_df,
        missing_threshold_pct=missing_threshold_pct,
        max_age_months=max_age_months,
    )
    model_df = prepare_model_dataframe(filtered_df)
    training_df, label_maps = to_training_matrix(model_df, label_map_path=label_map_path)

    filtered_df.to_csv(filtered_path, index=False)
    model_df.to_csv(model_path, index=False)
    training_df.to_csv(matrix_path, index=False)

    return {
        "training_df": training_df,
        "target_col": TARGET_COL,
        "paths": {
            "filtered": str(filtered_path),
            "model": str(model_path),
            "training_matrix": str(matrix_path),
            "label_map": str(label_map_path),
        },
        "label_maps": label_maps,
    }


def build_features(data, target_col=TARGET_COL):
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if target_col not in df.columns:
        raise ValueError("target column '{}' not found".format(target_col))

    y = pd.to_numeric(df[target_col], errors="coerce")
    X = df.drop(columns=[target_col]).copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    valid_mask = y.notna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)
    return X, y


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate(y_true, y_pred):
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred)}


def summarize_folds(fold_metrics):
    if not fold_metrics:
        return {"mae": None, "rmse": None}
    return {
        "mae": sum(m["mae"] for m in fold_metrics) / len(fold_metrics),
        "rmse": sum(m["rmse"] for m in fold_metrics) / len(fold_metrics),
    }


def kfold_indices(n_samples, n_splits=5, shuffle=True, random_state=42):
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n_splits > n_samples:
        raise ValueError("n_splits cannot exceed number of samples")

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate((indices[:start], indices[stop:]))
        current = stop
        yield train_idx, val_idx
