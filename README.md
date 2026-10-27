<div align="center">

# 🚕 Uber Fare Prediction

### Predict ride fares with 9 machine learning models, served through a FastAPI web app.

<br/>

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=for-the-badge&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

[Results](#-results) · [Quick start](#-quick-start) · [Web app & API](#-web-app--api) · [Features](#-feature-engineering) · [Methodology](#-methodology)

</div>

---

## 🎯 Overview

An end-to-end ML project that predicts Uber fares from pickup time, GPS coordinates and passenger count. It trains and compares **9 models** on **44,377 rides**, then serves the best one through a **FastAPI** backend with a web interface for predictions and retraining.

<table>
<tr>
<td align="center" width="25%"><h3>0.726</h3>R² score<br/><sub>CatBoost</sub></td>
<td align="center" width="25%"><h3>$1.37</h3>Mean absolute error<br/><sub>per ride</sub></td>
<td align="center" width="25%"><h3>9</h3>Models<br/><sub>compared</sub></td>
<td align="center" width="25%"><h3>13</h3>Features<br/><sub>engineered</sub></td>
</tr>
</table>

---

## 📊 Results

Evaluated on a held-out 20% test set.

| Model | R² | RMSE | MAE | Notes |
|---|---|---|---|---|
| **CatBoost** ⭐ | **0.7263** | **$1.93** | **$1.37** | Best overall, used by the API |
| LightGBM | 0.7171 | $1.96 | $1.41 | Fastest to train |
| Stacking Ensemble | 0.7161 | $1.97 | $1.41 | Most robust |
| Gradient Boosting | 0.6948 | $2.04 | $1.46 | Stable |
| Random Forest | 0.6929 | $2.05 | $1.48 | Interpretable |
| XGBoost | 0.6923 | $2.05 | $1.47 | Industry standard |
| Linear Regression | 0.6488 | $2.19 | $1.56 | Baseline |
| K-Nearest Neighbors | 0.6341 | $2.23 | $1.60 | Simple |
| Decision Tree | 0.3419 | $2.99 | $2.11 | Overfits |

> [!NOTE]
> An R² of 0.726 means the model explains **72.6% of the variation in fares**. In practice, a typical $10 ride is predicted within about ±$1.40.

---

## 🚀 Quick start

### Prerequisites

- **Python 3.12+**
- `make` (Linux / macOS, or WSL on Windows)

### One command

```bash
git clone https://github.com/yassinemaazaoui114-git/uber-fare-prediction.git
cd uber-fare-prediction
make full-start
```

This sets up the environment, trains all models, starts the API server and opens the web app in your browser.

### Step by step

```bash
make setup          # Create venv and install dependencies
make train          # Train all 9 models (~5 minutes)
make server-start   # Start the API at http://127.0.0.1:8000
make open-browser   # Open the web app
```

> [!IMPORTANT]
> Trained models (`.joblib`) aren't included in the repo because of their size. Run `make train` before starting the server.

<details>
<summary><b>Manual setup without make</b></summary>

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python3 main.py                 # Train all models
uvicorn api.app:app --port 8000 # Start the API
```

</details>

---

## 🌐 Web app & API

Once the server is running:

| Page | URL |
|---|---|
| 🏠 Home | http://127.0.0.1:8000/ |
| 🔮 Predict a fare | http://127.0.0.1:8000/predict-page |
| 🔁 Retrain a model | http://127.0.0.1:8000/retrain-page |
| 📚 Interactive API docs | http://127.0.0.1:8000/docs |

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server and model status |
| `POST` | `/predict` | Predict a fare with the best model |
| `POST` | `/retrain` | Retrain a model with custom hyperparameters |
| `GET` | `/models/hyperparameters/{model_type}` | Default hyperparameters for a model |

### Example prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_datetime": "2026-03-16T08:30:00",
    "pickup_longitude": -73.985,
    "pickup_latitude": 40.748,
    "dropoff_longitude": -73.974,
    "dropoff_latitude": 40.750,
    "passenger_count": 1
  }'
```

```json
{
  "predicted_fare": 7.85,
  "trip_distance_km": 0.95,
  "is_rush_hour": true,
  "is_weekend": false,
  "is_late_night": false
}
```

<sub>Response values are illustrative; your numbers depend on the trained model.</sub>

The API computes the distance and every engineered feature itself, so you only send the raw ride details.

---

## 🧠 Feature engineering

Raw inputs are turned into **13 features**:

<table>
<tr>
<td width="50%" valign="top">

### Base features (8)
| Feature | Description |
|---|---|
| `hour` | Pickup hour (0–23) |
| `day_of_week` | 0 = Monday … 6 = Sunday |
| `pickup_longitude` / `latitude` | Pickup GPS |
| `dropoff_longitude` / `latitude` | Dropoff GPS |
| `passenger_count` | Passengers (1–6) |
| `trip_km` | Distance (Haversine) |

</td>
<td width="50%" valign="top">

### Engineered features (5)
| Feature | Description |
|---|---|
| `is_rush_hour` | Weekdays 7–9 am, 5–7 pm |
| `is_weekend` | Saturday or Sunday |
| `is_late_night` | 10 pm – 4 am |
| `distance_category` | 1 = short … 4 = very long |
| `hour_distance_interaction` | hour × distance |

</td>
</tr>
</table>

---

## 🔬 Methodology

**Preprocessing**
1. **Cleaning:** remove duplicates and missing values
2. **Outlier removal:** IQR method (Q1 − 1.5×IQR, Q3 + 1.5×IQR)
3. **Feature engineering:** time-based and interaction features
4. **Split:** 80% train / 20% test (`random_state=42`)
5. **Scaling:** `StandardScaler` on numeric features

**Training**
1. Train 8 individual algorithms
2. Optional tuning with `RandomizedSearchCV` and 5-fold CV (`python3 main.py --tune`)
3. Stack the top 3 models (LightGBM, Gradient Boosting, Random Forest) with a Ridge meta-learner
4. Evaluate with R², RMSE and MAE
5. Save models with `joblib`

---

## 🛠 Make commands

| Command | Description |
|---|---|
| `make help` | Show all commands |
| `make full-start` | Setup + train + start server + open browser |
| `make setup` | Create venv and install dependencies |
| `make train` | Train all models |
| `make server-start` / `server-stop` / `server-restart` | Manage the API server |
| `make server-status` | Check if the server is running |
| `make logs` | Follow server logs |
| `make test` | Run tests with pytest |
| `make lint` | Check style with flake8 |
| `make format` | Format code with Black |
| `make security` | Security scan with Bandit |
| `make clean` | Remove cache files |
| `make clean-all` | Remove venv, models, logs and cache |
| `make info` | Project status and URLs |

### Training CLI

```bash
python3 main.py              # Train all models (fast)
python3 main.py --tune       # Train with hyperparameter tuning (slower, better)
python3 main.py --evaluate   # Evaluate existing models
```

---

## 📂 Project structure

<details>
<summary><b>Click to expand</b></summary>

```text
uber-fare-prediction/
├── main.py                 # Training pipeline + CLI
├── Makefile                # Automation commands
├── requirements.txt
├── pyproject.toml          # Black config
├── .flake8                 # Lint config
├── bandit.yml              # Security scan config
│
├── api/
│   └── app.py              # FastAPI backend (predict, retrain, health)
├── static/                 # Web interface
│   ├── index.html
│   ├── predict.html
│   └── retrain.html
├── pipelines/              # One file per model
│   ├── preprocessing.py
│   ├── catboost_pipeline.py
│   ├── lightgbm_pipeline.py
│   ├── xgboost_pipeline.py
│   ├── stacking_pipeline.py
│   └── ...
├── utils/
│   └── data_loader.py      # Loading, cleaning, feature engineering
├── data/
│   └── datauber.csv        # 44,377 rides
├── models/                 # Trained models (generated, git-ignored)
└── tests/
    └── test_models.py      # 6 tests
```

</details>

---

## 🧪 Quality checks

| Tool | Purpose |
|---|---|
| **pytest** | 6 tests: directories, model files, loading, prediction shape |
| **flake8** | Linting |
| **Black** | Formatting (100-char lines) |
| **Bandit** | Security scanning |

---

## 🗺 Roadmap

- [x] Model development and comparison
- [x] Automation with Makefile, tests and quality checks
- [x] Web interface and FastAPI backend
- [ ] Cloud deployment
- [ ] Extra data (traffic, weather, surge pricing), aiming for R² of 0.80–0.85

---

## 👨‍💻 Author

**Yassine Maazaoui**, machine learning student focused on data science and predictive analytics.

## 📝 License

Created for educational purposes.

---

<div align="center">

Built with scikit-learn · XGBoost · LightGBM · CatBoost · FastAPI

</div>
