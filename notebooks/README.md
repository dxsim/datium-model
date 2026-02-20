# Notebooks Summary

This folder contains the analysis and preprocessing notebooks used to build the training pipeline.

## 1) EDA Notebook
- Path: `notebooks/EDA/eda.ipynb`
- Purpose:
  - Explore pricing/depreciation behavior and data quality.
  - Identify leakage/anomaly patterns and useful predictive fields.
- Key conclusions:
  - `Sold_Amount` and `NewPrice` are right-skewed; transformed views are more stable.
  - Vehicle value depreciation is strongly related to age (`Age_Comp_Months` / `AgeDays`) and mileage (`KM`).
  - Most records fall into a practical resale window; very old tail records are sparse/noisy.
  - A small set of rows has `Sold_Amount > NewPrice` or zero sale price and should be filtered for modeling.

## 2) Preprocessing Notebook
- Path: `notebooks/preprocessing/preprocessing.ipynb`
- Purpose:
  - Convert raw `.rpt` files into model-ready datasets.
  - Apply cleaning, feature engineering, and encoding consistently.
- Key steps:
  - Drop selected pricing benchmark columns.
  - Keep columns with acceptable missingness, impute selected fields, and remove invalid target rows.
  - Derive `AgeDays` from `Sold_Date` and `Compliance_Date`.
  - Label-encode categorical variables and save mappings.
