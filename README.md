# datium-model

Simple baseline ML project for vehicle price modeling with cross-validation.

## 1. Setup

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

## 2. Run training

```bash
python -m src.train --config configs/train.yaml
```

This runs the LightGBM training pipeline, saves artifacts under `artifacts/`, and logs metrics to MLflow.

## 3. Run tests

```bash
pytest -q
```

## 4. Run MLflow UI

```bash
./.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

Open `http://localhost:5000` in your browser.

## 5. Data files

Raw files expected at:

- `data/raw/DatiumTrain.rpt`
- `data/raw/DatiumTest.rpt`

## 6. Notebooks

- EDA: `notebooks/EDA/eda.ipynb`
- Preprocessing: `notebooks/preprocessing/preprocessing.ipynb`
- Results: `notebooks/results/results.ipynb`

## 7. Results

Latest run (`artifacts/model_info.txt`):

- Model: `lightgbm_regressor`
- CV MAE: `1607.19`
- CV RMSE: `2925.26`
- Train R2: `0.9640`
- Train RMSE: `1930.75`
- Test R2: `0.9323`
- Test RMSE: `2908.32`
- Holdout MAE: `1944.54`
- Holdout RMSE: `2908.32`
