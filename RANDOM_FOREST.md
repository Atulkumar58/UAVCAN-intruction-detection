# Random Forest model

The project includes a Random Forest alternative to the NumPy MLP. It uses the
same preprocessed 11-feature UAVCAN input and maps every nonzero attack label
to the binary `Attack` class.

Install the Python dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Train the model:

```powershell
.\.venv\Scripts\python.exe train_uavcan_random_forest.py
```

The default outputs are:

- `processed_data/uavcan_random_forest_model.pkl`
- `processed_data/uavcan_random_forest_model_metrics.json`

Evaluate the saved model:

```powershell
.\.venv\Scripts\python.exe test_uavcan_random_forest.py `
  --model processed_data/uavcan_random_forest_model.pkl `
  --input processed_data/uavcan_processed.npz `
  --output processed_data/uavcan_random_forest_predictions.csv
```

Useful training options include `--trees`, `--max-depth`,
`--min-samples-leaf`, `--threshold`, `--seed`, and `--max-samples`.
Random Forest does not require feature standardization, so it uses the original
numeric features from the preprocessed NPZ file.

## API and frontend

After both model files exist, start the API:

```powershell
.\.venv\Scripts\python.exe api_uavcan_model.py
```

The API provides:

- `POST /predict` for the MLP
- `POST /predict-random-forest` for the Random Forest

Both endpoints accept the same JSON CAN-frame body. The included frontend calls
both endpoints for each submitted frame and displays the two predictions and
attack probabilities side by side.
