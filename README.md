# Machine Learning Based Raw Material Quality Prediction

A Flask-based machine learning web application that demonstrates raw material quality prediction across three domains: **Food Processing**, **Textile**, and **Cosmetics**.

The project generates reproducible synthetic datasets, trains and compares 10 classification algorithms, stores the trained artifacts, and exposes a browser-based interface for predictions and model analytics.

> **Important:** This is an academic/portfolio project. The datasets are synthetically generated for experimentation and demonstration; the application is not intended to replace laboratory testing, regulatory checks, or industrial quality-control procedures.

## Features

- Three industry-specific prediction workflows
- 10 machine learning classification algorithms
- Standardized feature preprocessing
- Label encoding for target classes
- Train/test evaluation and 5-fold cross-validation
- Random Forest feature-importance analysis
- Single-model and all-model prediction APIs
- Flask web interface with prediction and analytics pages

## Industries

- **Food Processing:** moisture, protein, fat, fiber, ash, pH, sugar, acidity, color index, storage temperature
- **Textile:** fiber length, strength, micronaire, uniformity, elongation, moisture regain, trash content, color grade, maturity ratio, short-fiber percentage
- **Cosmetics:** purity, viscosity, pH, particle size, moisture, peroxide value, saponification value, color degree, heavy metals, microbial count

## Machine Learning Models

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. AdaBoost
6. Extra Trees
7. Bagging
8. Support Vector Machine
9. K-Nearest Neighbors
10. Naive Bayes

## Project Structure

```text
raw-material-quality-prediction/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/              # generated locally after training
└── templates/
    ├── landing.html
    ├── predict.html
    ├── analytics.html
    └── about.html
```

## How It Works

1. `train_models.py` generates synthetic data for each industry.
2. Features are standardized with `StandardScaler` and labels are encoded with `LabelEncoder`.
3. Data is split into training and test sets.
4. Ten classification models are trained and evaluated.
5. Five-fold cross-validation is calculated for each model.
6. Models and preprocessing objects are serialized into the `models/` directory.
7. `app.py` loads those artifacts and serves the web interface and prediction APIs.

## Setup

```bash
git clone https://github.com/avulasailakshmi/raw-material-quality-prediction.git
cd raw-material-quality-prediction
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Generate the datasets and model artifacts:

```bash
python train_models.py
```

Start the application:

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## API Endpoints

- `GET /api/summary` — model performance, feature importance, classes and training summary
- `POST /api/predict` — prediction using one selected model
- `POST /api/predict_all` — predictions from all trained models

## Current Experimental Results

| Industry | Best Model | Test Accuracy |
|---|---|---:|
| Food Processing | Support Vector Machine | 67.50% |
| Textile | Logistic Regression | 98.75% |
| Cosmetics | Logistic Regression | 95.00% |

These results are based on synthetic datasets and demonstrate the ML workflow rather than real-world industrial performance.

## Skills Demonstrated

Python, Flask, NumPy, pandas, scikit-learn, data preprocessing, classification, cross-validation, feature importance, model serialization, REST-style API development, and frontend/backend integration.

## Possible Improvements

- Replace synthetic datasets with validated real-world datasets
- Add class-balancing and stratified evaluation
- Add precision, recall, F1-score, ROC-AUC and confusion matrices
- Add automated API tests and stronger input validation
- Containerize and deploy the application

## Portfolio Summary

**Machine Learning Based Raw Material Quality Prediction** is an end-to-end ML application covering dataset creation, preprocessing, multi-model training and evaluation, model persistence, Flask API development, and a browser-based prediction interface across three raw-material domains.
