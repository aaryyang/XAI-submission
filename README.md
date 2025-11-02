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
- numpy, pandas, scikit-learn, matplotlib, joblib
- tensorflow (for LSTM model)
- shap, lime (optional, for explainability)

Install dependencies (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r .\requirements.txt
```

Or install individually if you prefer:

```powershell
pip install numpy pandas scikit-learn matplotlib joblib tensorflow shap lime
```

Note: `XAI.py` contains a Colab cell that starts with `!pip install ...`. When running locally as a script, either remove or comment out that line and install dependencies using the commands above.

## Usage
1. Ensure your dataset CSV matches the expected columns: `Date`, `Open`, `High`, `Low`, `%Chg`, `Price`.
2. Edit the top of `XAI.py` and set `DATA_PATH` to the CSV path (absolute or relative) and `OUT_DIR` to a writable output directory.
3. (Optional) Adjust hyperparameters at the top of `XAI.py`: `TIME_STEPS`, `TEST_SIZE`, `LSTM_UNITS`, `DROPOUT`, `BATCH_SIZE`, `EPOCHS`, and `LEARNING_RATE`.
4. Run the script:

```powershell
python .\XAI.py
```

If running in Colab, upload the CSV and run the notebook cells as-is.

## Outputs
By default, outputs are saved to `OUT_DIR` (configured in the script). Typical artifact names:
- `metrics.json` and `metrics.csv` — numeric evaluation results (RMSE, MAE, R2, and whether LSTM was used)
- `explainability.json` — whether LIME/SHAP ran successfully and any errors
- `lstm_model.h5` — trained Keras model (saved when TensorFlow model was used)
- `scaler_X.pkl`, `scaler_y.pkl` — saved scalers used to normalize inputs/targets

Plots (Actual vs Predicted, LIME bar chart, SHAP plots) are displayed during execution (matplotlib or SHAP visualizations). When running headless, consider saving figures instead of showing them.

## Notes & Caveats
- The script expects clean numeric columns; it performs some coercion and drops rows with missing required fields.
- The script was exported from a Colab notebook and contains constructs (like the `!pip install` cell) that are best removed when running outside notebooks.
- Explainability (LIME/SHAP) is approximate here — KernelExplainer and LIME wrapper are relatively slow and intended for demonstration.

## Quick tips
- To run faster/lighter: reduce `EPOCHS` and `BATCH_SIZE` or use fewer `TIME_STEPS`.
- If TensorFlow is not installed or fails, the script falls back to a naive baseline predictor (shifted historical values) — metrics will reflect this.

## License
This README and the added small helper files are released under the MIT License. The original notebook/script licensing is unknown — check the original author/source if you plan to reuse the code publicly.

## Contact / Next steps
- If you'd like, I can:
  - Convert the exported script back to a clean, runnable Python module with CLI flags.
  - Add a small `requirements-lock.txt` with pinned versions and a reproducible Dockerfile.
  - Create a small wrapper to save figures instead of showing them for headless runs.

---
