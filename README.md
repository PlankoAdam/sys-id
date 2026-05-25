# sys-id — Transfer Function System Identification

A machine learning platform that identifies continuous-time linear transfer functions from measured step response data. Upload a step response, and the system classifies the transfer function type and estimates its parameters (gain, time constants, damping ratio).

## What it does

Given a measured step response signal, sys-id:
1. Classifies the system into one of six transfer function families
2. Estimates the parameters of the identified model
3. Simulates the fitted model's response for visual validation

### Supported transfer function types

| Type | Transfer function | Parameters |
|---|---|---|
| ORD1 | $\dfrac{K}{Ts+1}$ | $K,\ T$ |
| ORD1_ASTAT | $\dfrac{K}{s}$ | $K$ |
| ORD2_APER | $\dfrac{K}{(T_1s+1)(T_2s+1)}$ | $K,\ T_1,\ T_2$ |
| ORD2_PER | $\dfrac{K}{T^2s^2+2bTs+1}$ | $K,\ T,\ b$ |
| ORD2_ASTAT | $\dfrac{K}{s^2}$ | $K$ |
| ORD2_ASTAT_T | $\dfrac{K}{s(Ts+1)}$ | $K,\ T$ |

## Architecture

```
frontend (Vue 3 + Plotly)   →   backend API (FastAPI)   →   ML models (PyTorch)
        :8080                          :8000
```

**Backend** — FastAPI server exposing two endpoints:
- `POST /predict` — takes a step response array, returns the identified TF type, estimated parameters, and simulated output
- `POST /step` — simulates a step response from given TF parameters (for validation/plotting)

**Frontend** — Vue 3 + TypeScript single-page app:
- CSV upload for measured step response data
- Auto or manual TF-type selection
- Interactive parameter sliders with live Plotly plots
- LaTeX-rendered transfer function equations

## ML pipeline

Training runs offline and produces model files that are mounted into the Docker container at inference time.

### 1. Generate training data

```bash
cd backend
python gen_data.py
```

Randomly samples transfer function parameters within configured ranges (e.g. K ∈ [0.05, 100], T ∈ [0.05, 80]), simulates step responses using the `control` library, and applies three noise levels (0%, 50%, 100% of σ = 0.05) for augmentation. Produces 10,000 training + 1,000 validation samples per TF type, saved as `.npy` files under `step_input/` and `osc_input/`.

### 2. Train models

```bash
python train_models.py
```

Trains two model families:
- **Classifier** — identifies which of the six TF types a response belongs to
- **Regressors** — one per TF type, predicts the numerical parameters

Each family is trained twice: once on step-response data and once on oscillating (square-wave) input data. Trained weights are saved under `step_input/models/` and `osc_input/models/`.

## Helper scripts

All scripts are run from the `backend/` directory.

| Script | Description |
|---|---|
| `gen_datasets.py` | Generate training and validation datasets for all TF types |
| `train_models.py` | Train the classifier and all regressor models |
| `eval.py` | Evaluate trained models — prints metrics for both the classifier and all regressors, for both step and oscillating input modes |
| `predict_rand.py` | Run inference on a random sample from the dataset and print the predicted vs. true TF type and parameters |
| `plot_rand.py` | Plot a random dataset sample (raw signal + noise levels) for visual inspection |
| `test.py` | Ad-hoc testing and experimentation |

## Running with Docker

```bash
docker compose up
```

- Frontend: http://localhost:8080
- Backend API: http://localhost:8000

The compose file mounts `./backend/step_input/models` and `./backend/osc_input/models` into the container, so you can retrain and redeploy without rebuilding the image.

## Running locally (development)

**Backend**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt        # or requirements-cuda.txt for GPU
fastapi dev server.py --port 8000
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

## Configuration

Training behaviour and parameter ranges are controlled by `backend/src/core/config.py`. Key settings:

| Setting | Default | Description |
|---|---|---|
| `TRAIN_NUM_SAMPLES` | 10 000 | Samples generated per TF type |
| `VAL_NUM_SAMPLES` | 1 000 | Validation samples per TF type |
| `TIME_POINTS` | 200 | Number of time points per step response |
| `STEP_TIME_END_MIN/MAX` | 5 – 1000 s | Simulation window range |
| `NOISE_STD` | 0.05 | Relative noise standard deviation |
| `O1_K_MIN/MAX` | 0.05 – 100 | ORD1 gain range |
| `O1_T_MIN/MAX` | 0.05 – 80 s | ORD1 time constant range |
| `O2PER_B_MIN/MAX` | 0.1 – 0.9 | ORD2_PER damping ratio range |

Edit this file to change the training distribution before regenerating data and retraining.
