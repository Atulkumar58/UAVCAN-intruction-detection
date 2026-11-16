# Keras 1D CNN model

Install the model dependencies in the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Train the model from the preprocessed UAVCAN data:

```powershell
.\.venv\Scripts\python.exe train_uavcan_cnn.py
```

The command saves `processed_data\uavcan_attack_cnn.keras` and a JSON sidecar
containing the feature normalization statistics and classification threshold.

Evaluate the saved model:

```powershell
.\.venv\Scripts\python.exe test_uavcan_cnn.py `
  --model processed_data\uavcan_attack_cnn.keras `
  --input processed_data\uavcan_processed.npz
```

Start the API after training. `api_uavcan_model.py` loads both the CNN artifact
and the existing NumPy MLP artifact by default. The frontend displays both
predictions and whether they agree:

```powershell
.\.venv\Scripts\python.exe api_uavcan_model.py
```

Use `--cnn-model` and `--mlp-model` to provide different artifact paths.
