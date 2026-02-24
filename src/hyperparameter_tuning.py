import argparse
import json
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import yaml
from sklearn.model_selection import ParameterSampler

from src.train import MODEL_FEATURES, _build_train_and_test_matrices
from src.utils import TARGET_COL, evaluate, find_project_root, kfold_indices, load_yaml


def run_tuning_from_config(config_path):
    cfg = load_yaml(config_path)
    train_cfg = cfg.get("train", {})

    random_state = int(train_cfg.get("random_state", 42))
    verbosity = int(train_cfg.get("verbosity", 1))
    n_splits = int(train_cfg.get("n_splits", 3))
    n_iter = int(train_cfg.get("n_iter", 20))
    sample_frac = float(train_cfg.get("sample_frac", 0.5))
    lgbm_n_jobs = int(train_cfg.get("lgbm_n_jobs", 4))
    early_stopping_rounds = int(train_cfg.get("early_stopping_rounds", 50))
    missing_threshold_pct = float(train_cfg.get("missing_threshold_pct", 5.0))
    max_age_months = int(train_cfg.get("max_age_months", 246))
    tuning_features = train_cfg.get("tuning_features") or MODEL_FEATURES[:10]
    param_grid = train_cfg.get("hyperparameter_grid")
    if not isinstance(param_grid, dict) or not param_grid:
        raise ValueError(
            "Missing 'train.hyperparameter_grid' in config. "
            "Use configs/hyperparameter_tuning.yaml."
        )

    mlflow_experiment = cfg.get("mlflow_experiment", "datium-model")
    run_name = cfg.get("run_name", "grid-search-lightgbm")

    project_root = find_project_root(Path.cwd())
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, encoder_path = _build_train_and_test_matrices(
        project_root=project_root,
        missing_threshold_pct=missing_threshold_pct,
        max_age_months=max_age_months,
    )

    X_train = train_df[MODEL_FEATURES]
    y_train = train_df[TARGET_COL]
    X_test = test_df[MODEL_FEATURES]
    y_test = test_df[TARGET_COL]

    # Tune faster on sampled rows and reduced feature subset.
    X_tune = X_train[tuning_features].copy()
    y_tune = y_train.copy()
    if 0 < sample_frac < 1:
        sample_n = max(1000, int(len(X_tune) * sample_frac))
        sample_n = min(sample_n, len(X_tune))
        sampled = X_tune.sample(n=sample_n, random_state=random_state)
        X_tune = sampled.reset_index(drop=True)
        y_tune = y_tune.loc[sampled.index].reset_index(drop=True)

    sampled_params = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=random_state))
    best_params = None
    best_rmse = float("inf")

    for candidate in sampled_params:
        fold_rmses = []
        for train_idx, val_idx in kfold_indices(
            len(X_tune), n_splits=n_splits, shuffle=True, random_state=random_state
        ):
            model = lgb.LGBMRegressor(
                objective="regression",
                random_state=random_state,
                verbosity=verbosity,
                n_jobs=lgbm_n_jobs,
                **candidate,
            )
            X_fold_train = X_tune.iloc[train_idx]
            y_fold_train = y_tune.iloc[train_idx]
            X_fold_val = X_tune.iloc[val_idx]
            y_fold_val = y_tune.iloc[val_idx]

            model.fit(
                X_fold_train,
                y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
            pred = model.predict(X_fold_val)
            fold_rmses.append(evaluate(y_fold_val, pred)["rmse"])

        mean_rmse = float(np.mean(fold_rmses))
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = candidate

    if best_params is None:
        raise RuntimeError("Randomized tuning failed to produce best parameters.")

    # Refit best params on full training data and evaluate holdout.
    best_model = lgb.LGBMRegressor(
        objective="regression",
        random_state=random_state,
        verbosity=verbosity,
        n_jobs=lgbm_n_jobs,
        **best_params,
    )
    best_model.fit(X_train, y_train)
    test_pred = best_model.predict(X_test)
    test_metrics = evaluate(y_test, test_pred)

    best_hyperparameters = list(best_params.keys())
    best_values = list(best_params.values())

    output = {
        "best_params": best_params,
        "best_hyperparameters": best_hyperparameters,
        "best_values": best_values,
        "best_cv_score_neg_rmse": float(-best_rmse),
        "holdout_mae": float(test_metrics["mae"]),
        "holdout_rmse": float(test_metrics["rmse"]),
    }

    best_params_path = artifacts_dir / "best_hyperparameters.json"
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Save best params in train.yaml-like format.
    tune_only_keys = {
        "hyperparameter_grid",
        "n_iter",
        "sample_frac",
        "tuning_features",
        "early_stopping_rounds",
        "lgbm_n_jobs",
    }
    best_train_cfg = {k: v for k, v in train_cfg.items() if k not in tune_only_keys}
    best_train_cfg.update(best_params)
    best_yaml_payload = {
        "run_name": cfg.get("run_name", "prod-run"),
        "mlflow_experiment": mlflow_experiment,
        "train": best_train_cfg,
    }
    best_yaml_path = project_root / "configs" / "best_hyperparameters.yaml"
    with open(best_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(best_yaml_payload, f, sort_keys=False)

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("search_type", "RandomizedSearchCV-Manual")
        mlflow.log_param("cv", n_splits)
        mlflow.log_param("scoring", "neg_root_mean_squared_error")
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("verbosity", verbosity)
        mlflow.log_param("n_iter", n_iter)
        mlflow.log_param("sample_frac", sample_frac)
        mlflow.log_param("lgbm_n_jobs", lgbm_n_jobs)
        mlflow.log_param("early_stopping_rounds", early_stopping_rounds)
        mlflow.log_param("n_rows_train", int(len(train_df)))
        mlflow.log_param("n_rows_tune", int(len(X_tune)))
        mlflow.log_param("n_rows_test", int(len(test_df)))
        mlflow.log_param("n_tuning_features", int(len(tuning_features)))
        mlflow.log_param("param_grid", json.dumps(param_grid, sort_keys=True))

        for k, v in best_params.items():
            mlflow.log_param("best_" + k, v)

        mlflow.log_metric("best_cv_score_neg_rmse", float(-best_rmse))
        mlflow.log_metric("holdout_mae", float(test_metrics["mae"]))
        mlflow.log_metric("holdout_rmse", float(test_metrics["rmse"]))

        mlflow.log_artifact(str(encoder_path), artifact_path="encoder")
        mlflow.log_artifact(str(best_params_path), artifact_path="tuning")
        mlflow.log_artifact(str(best_yaml_path), artifact_path="tuning")

    output["best_yaml_path"] = str(best_yaml_path)
    return output


def main():
    parser = argparse.ArgumentParser(description="Fast randomized tuning for LightGBM hyperparameters")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hyperparameter_tuning.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    result = run_tuning_from_config(args.config)
    print("Best hyperparameters:", result["best_hyperparameters"])
    print("Best values:", result["best_values"])
    print("Best CV score (neg RMSE):", result["best_cv_score_neg_rmse"])
    print("Holdout MAE:", result["holdout_mae"])
    print("Holdout RMSE:", result["holdout_rmse"])
    print("Best config YAML:", result["best_yaml_path"])


if __name__ == "__main__":
    main()
