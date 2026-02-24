import argparse
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.utils import (
    TARGET_COL,
    build_features,
    evaluate,
    filter_dataset,
    find_project_root,
    kfold_indices,
    load_yaml,
    prepare_model_dataframe,
    summarize_folds,
)

MODEL_FEATURES = [
    "NewPrice",
    "AgeDays",
    "KM",
    "Age_Comp_Months",
    "Height",
    "MakeCode",
    "YearGroup",
    "Make",
    "BuildCountryOriginDescription",
    "AverageKM",
    "KerbWeight",
    "SaleCategory",
    "Branch",
    "RearRimDesc",
    "VIN",
    "PowerRPMTo",
    "FrontRimDesc",
    "GoodKM",
    "RearTyreSize",
    "VFactsSegment",
]


def _build_lgbm_params_from_config(train_cfg):
    # Non-model keys used elsewhere in the pipeline.
    excluded = {
        "n_splits",
        "random_state",
        "missing_threshold_pct",
        "max_age_months",
        "n_iter",
        "sample_frac",
        "tuning_features",
        "early_stopping_rounds",
        "hyperparameter_grid",
    }
    random_state = int(train_cfg.get("random_state", 42))

    params = {
        "objective": "regression",
        "verbosity": -1,
        "random_state": random_state,
    }

    # If provided, prefer explicit nested config: train.lgbm_params
    nested = train_cfg.get("lgbm_params", {})
    if isinstance(nested, dict) and nested:
        params.update(nested)
        return params

    # Backward compatibility: read model params directly under train.
    for key, value in train_cfg.items():
        if key in excluded:
            continue
        if key == "lgbm_n_jobs":
            params["n_jobs"] = value
            continue
        params[key] = value
    return params


def _ensure_columns(df, columns):
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def _is_categorical_col(series):
    return str(series.dtype) in ("object", "bool", "category")


def _fit_label_maps(reference_df, feature_cols):
    label_maps = {}
    for col in feature_cols:
        if col in reference_df.columns and _is_categorical_col(reference_df[col]):
            labels = sorted(reference_df[col].astype(str).unique().tolist())
            label_maps[col] = {label: idx for idx, label in enumerate(labels)}
    return label_maps


def _apply_label_maps(df, label_maps):
    out = df.copy()
    for col, mapping in label_maps.items():
        if col not in out.columns:
            continue
        out[col] = out[col].astype(str).map(mapping).fillna(-1).astype(int)
    return out


def _cast_numeric_with_train_stats(train_df, other_df, feature_cols, exclude_cols=None):
    exclude_cols = set(exclude_cols or [])
    train_out = train_df.copy()
    other_out = other_df.copy()
    for col in feature_cols:
        if col in exclude_cols:
            continue
        train_out[col] = pd.to_numeric(train_out[col], errors="coerce")
        other_out[col] = pd.to_numeric(other_out[col], errors="coerce")
        fill_value = train_out[col].median()
        if pd.isna(fill_value):
            fill_value = 0.0
        train_out[col] = train_out[col].fillna(fill_value)
        other_out[col] = other_out[col].fillna(fill_value)
    return train_out, other_out


def _build_train_and_test_matrices(project_root, missing_threshold_pct=5.0, max_age_months=246):
    raw_dir = project_root / "data" / "raw"
    artifacts_dir = project_root / "artifacts"
    encoder_dir = artifacts_dir / "encoder"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    encoder_dir.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(raw_dir / "DatiumTrain.rpt", sep="\t", encoding="utf-8-sig", low_memory=False)
    test_raw = pd.read_csv(raw_dir / "DatiumTest.rpt", sep="\t", encoding="utf-8-sig", low_memory=False)

    train_pre = prepare_model_dataframe(
        filter_dataset(train_raw, missing_threshold_pct=missing_threshold_pct, max_age_months=max_age_months)
    )
    test_pre = prepare_model_dataframe(
        filter_dataset(test_raw, missing_threshold_pct=missing_threshold_pct, max_age_months=max_age_months)
    )

    selected_cols = MODEL_FEATURES + [TARGET_COL]
    train_pre = _ensure_columns(train_pre, selected_cols)[selected_cols]
    test_pre = _ensure_columns(test_pre, selected_cols)[selected_cols]

    # Requirement: encode DatiumTest first and use that mapping.
    label_maps = _fit_label_maps(test_pre, MODEL_FEATURES)
    test_pre = _apply_label_maps(test_pre, label_maps)
    train_pre = _apply_label_maps(train_pre, label_maps)

    train_pre, test_pre = _cast_numeric_with_train_stats(
        train_pre, test_pre, MODEL_FEATURES, exclude_cols=label_maps.keys()
    )
    train_pre[TARGET_COL] = pd.to_numeric(train_pre[TARGET_COL], errors="coerce")
    test_pre[TARGET_COL] = pd.to_numeric(test_pre[TARGET_COL], errors="coerce")
    train_pre = train_pre[train_pre[TARGET_COL].notna()].reset_index(drop=True)
    test_pre = test_pre[test_pre[TARGET_COL].notna()].reset_index(drop=True)

    encoder_path = encoder_dir / "label_encoding_map.json"
    with open(encoder_path, "w", encoding="utf-8") as f:
        json.dump(label_maps, f, indent=2, sort_keys=True)

    return train_pre, test_pre, encoder_path


def train(
    data,
    target_col=TARGET_COL,
    n_splits=5,
    lgbm_params=None,
):
    X, y = build_features(data=data, target_col=target_col)

    if len(X) < n_splits:
        raise ValueError("not enough rows for n_splits={}".format(n_splits))

    if not lgbm_params:
        raise ValueError("lgbm_params must be provided from config")
    params = dict(lgbm_params)
    random_state = int(params.get("random_state", 42))

    fold_metrics = []
    for fold_id, (train_idx, val_idx) in enumerate(
        kfold_indices(len(X), n_splits=n_splits, shuffle=True, random_state=random_state),
        start=1,
    ):
        model = lgb.LGBMRegressor(**params)
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        metrics = evaluate(y_val, pred)
        metrics["fold"] = fold_id
        fold_metrics.append(metrics)

    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(X, y)

    return {
        "model": "lightgbm_regressor",
        "rows": int(len(X)),
        "target_col": target_col,
        "feature_cols": list(X.columns),
        "estimator": final_model,
        "cv": {
            "n_splits": int(n_splits),
            "fold_metrics": fold_metrics,
            "mean_metrics": summarize_folds(fold_metrics),
        },
    }


def run_training_from_config(config_path):
    cfg = load_yaml(config_path)
    train_cfg = cfg.get("train", {})

    n_splits = int(train_cfg.get("n_splits", 5))
    random_state = int(train_cfg.get("random_state", 42))
    missing_threshold_pct = float(train_cfg.get("missing_threshold_pct", 5.0))
    max_age_months = int(train_cfg.get("max_age_months", 246))
    mlflow_experiment = cfg.get("mlflow_experiment", "datium-model")
    run_name = cfg.get("run_name", "train-lightgbm")

    project_root = find_project_root(Path.cwd())
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, encoder_path = _build_train_and_test_matrices(
        project_root=project_root,
        missing_threshold_pct=missing_threshold_pct,
        max_age_months=max_age_months,
    )

    lgbm_params = _build_lgbm_params_from_config(train_cfg)

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "n_rows_train": int(len(train_df)),
                "n_rows_test": int(len(test_df)),
                "n_cols": int(train_df.shape[1]),
                "target_col": TARGET_COL,
                "n_splits": n_splits,
                "missing_threshold_pct": missing_threshold_pct,
                "max_age_months": max_age_months,
                "n_model_features": len(MODEL_FEATURES),
                **{f"lgbm_{k}": v for k, v in lgbm_params.items()},
            }
        )

        result = train(
            data=train_df,
            target_col=TARGET_COL,
            n_splits=n_splits,
            lgbm_params=lgbm_params,
        )

        for fold in result["cv"]["fold_metrics"]:
            step = int(fold["fold"])
            mlflow.log_metric("cv_mae", float(fold["mae"]), step=step)
            mlflow.log_metric("cv_rmse", float(fold["rmse"]), step=step)

        mean_metrics = result["cv"]["mean_metrics"]
        mlflow.log_metrics(
            {
                "cv_mae_mean": float(mean_metrics["mae"]),
                "cv_rmse_mean": float(mean_metrics["rmse"]),
            }
        )

        # Evaluate on transformed test holdout.
        X_train = train_df[MODEL_FEATURES]
        y_train = train_df[TARGET_COL]
        X_test = test_df[MODEL_FEATURES]
        y_test = test_df[TARGET_COL]

        train_r2 = float(result["estimator"].score(X_train, y_train))
        train_rmse = float(np.sqrt(mean_squared_error(y_train, result["estimator"].predict(X_train))))
        test_r2 = float(result["estimator"].score(X_test, y_test))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, result["estimator"].predict(X_test))))

        holdout_pred = result["estimator"].predict(X_test)
        holdout_metrics = evaluate(y_test, holdout_pred)
        mlflow.log_metrics(
            {
                "train_r2": train_r2,
                "train_rmse": train_rmse,
                "test_r2": test_r2,
                "test_rmse": test_rmse,
                "holdout_mae": float(holdout_metrics["mae"]),
                "holdout_rmse": float(holdout_metrics["rmse"]),
            }
        )

        model_path = artifacts_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(result["estimator"], f)
        features_path = artifacts_dir / "model_features.json"
        features_path.write_text(json.dumps(MODEL_FEATURES, indent=2), encoding="utf-8")

        model_info_path = artifacts_dir / "model_info.txt"
        model_info_path.write_text(
            "model=lightgbm_regressor\n"
            "cv_mae_mean={:.6f}\n"
            "cv_rmse_mean={:.6f}\n"
            "train_r2={:.6f}\n"
            "train_rmse={:.6f}\n"
            "test_r2={:.6f}\n"
            "test_rmse={:.6f}\n"
            "holdout_mae={:.6f}\n"
            "holdout_rmse={:.6f}\n".format(
                float(mean_metrics["mae"]),
                float(mean_metrics["rmse"]),
                train_r2,
                train_rmse,
                test_r2,
                test_rmse,
                float(holdout_metrics["mae"]),
                float(holdout_metrics["rmse"]),
            ),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(encoder_path), artifact_path="encoder")
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(features_path), artifact_path="model")
        mlflow.log_artifact(str(model_info_path), artifact_path="model")

        result["train_eval"] = {"r2": train_r2, "rmse": train_rmse}
        result["test_eval"] = {"r2": test_r2, "rmse": test_rmse}
        return result


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM with simple CV + MLflow")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/best_hyperparameters.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    result = run_training_from_config(args.config)
    print("model:", result["model"])
    print("rows:", result["rows"])
    print("cv:", result["cv"]["mean_metrics"])
    print("Train R2: {:.4f}".format(result["train_eval"]["r2"]))
    print("Train RMSE: {:.4f}".format(result["train_eval"]["rmse"]))
    print("Test  R2: {:.4f}".format(result["test_eval"]["r2"]))
    print("Test RMSE: {:.4f}".format(result["test_eval"]["rmse"]))


if __name__ == "__main__":
    main()
