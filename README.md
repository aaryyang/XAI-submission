# XAI (LSTM + Explainability) Notebook/Script

This repository contains an exported Colab notebook (`XAI.py`) which trains an LSTM model to predict a stock/index price (example: NIFTY) and computes simple explainability artifacts using LIME and SHAP.

The code was originally authored as a Google Colab notebook and exports to a Python script. It performs data loading, preprocessing, LSTM training (if TensorFlow is available), prediction, evaluation, visualization, and optional explainability using LIME and SHAP.

## What this does
- Loads a time-series CSV containing columns: `Date`, `Open`, `High`, `Low`, `%Chg`, and `Price` (target).
- Builds time-window sequences and trains an LSTM to predict the `Price` column.
- Evaluates predictions (RMSE, MAE, R^2) and saves metrics and artifacts.
- Attempts to run LIME and SHAP explainers on the trained model (if installed).

## Files
- `XAI.py` — exported Colab notebook; main script. Edit `DATA_PATH` and `OUT_DIR` at the top before running locally.
- `requirements.txt` — dependency list used by the notebook (created alongside this README).

## Requirements
- Python 3.8+ recommended
- A GPU is beneficial for training with TensorFlow but not required for small experiments.

Key Python packages:

## Dataset (NIFTY50)

This project commonly uses the NIFTY50 dataset from Kaggle:

- Source: https://www.kaggle.com/datasets/sahilwagh/nifty50

Notes about using that dataset with `XAI.py`:

- The exported script expects a CSV with at least these feature columns: `Open`, `High`, `Low`, `%Chg` and a target column named `Price` (see `FEATURE_COLUMNS` and `TARGET_COLUMN` at the top of `XAI.py`).
- The Kaggle NIFTY50 dataset typically uses `Close` as the closing price column rather than `Price`. If your CSV uses `Close`, either:
  - Rename the `Close` column to `Price` in the CSV before running, or
  - Edit the `TARGET_COLUMN` in `XAI.py` to `"Close"` (search for the line near the top that sets `TARGET_COLUMN`).
- Ensure the `Date` column exists and is parseable; the script tries common variants (`Date`, `date`) and falls back to the first column if neither is present.

If you want, I can update `XAI.py` to accept `Close` as the default target and make the column names configurable via command-line arguments — would you like that?

**About the Dataset:**

This dataset contains annual Gross Domestic Product (GDP) data for all recognized countries covering the years 2020 to 2025. Disputed territories are excluded. The dataset is compiled from IMF sources and is well-suited for studying global economic trends, forecasting, and analyzing the growth or decline of individual economies over this period.



The dataset contains yearly values from 1995 to 2015.



**Columns:**



Date, Price, Open, High, Low, Close, %Chg
