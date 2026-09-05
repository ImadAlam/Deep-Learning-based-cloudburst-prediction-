# Advanced Cloudburst Prediction System

## A Research-Driven Early Warning Project for Babusar Top

A deep learning research project for predicting extreme cloudburst risk in the Babusar Top region of Kaghan Valley, Khyber Pakhtunkhwa, Pakistan. The system combines ERA5 atmospheric reanalysis, engineered time-series features, and a Bidirectional Long Short-Term Memory (BiLSTM) neural network with a Flask-based web application and prediction API.

> **Research focus:** Short-horizon, data-driven cloudburst prediction for complex mountainous terrain.
>
> **Forecast horizon:** Up to 2 hours ahead
>
> **Study location:** Babusar Top, approximately 35.6 N, 73.6 E

## About the Research Project

Cloudbursts are intense, localized rainfall events that can trigger flash floods, landslides, road closures, and severe risks to communities and travelers. Mountain environments are especially difficult to forecast because terrain, moisture transport, convection, and rapidly changing atmospheric conditions interact at small spatial and temporal scales.

This project investigates whether a sequence model can learn useful warning signals from hourly atmospheric conditions. The trained model is served through a lightweight Flask application so that research outputs can be explored through a browser dashboard or integrated into another monitoring system.

## Objectives

- Develop a BiLSTM model for short-term cloudburst probability prediction.
- Use ERA5 atmospheric reanalysis and total precipitation data for a mountainous study region.
- Represent temporal behavior through six-hour input sequences.
- Predict cloudburst conditions two hours in advance.
- Address severe class imbalance using focal loss and balanced class weights.
- Evaluate the model with precision-recall metrics rather than accuracy alone.
- Deliver the trained model through a practical Flask dashboard and REST API.
- Provide a foundation for future live data integration, alerting, and higher-resolution forecasting.

## Methodology

### Data and Features

The research uses hourly ERA5 reanalysis data covering 2014-2023, combined with total precipitation. The feature set contains 26 variables, including atmospheric conditions at 500, 700, and 850 hPa, surface meteorological variables, precipitation, and cyclical time features for seasonal behavior.

The documented preprocessing workflow includes:

1. Pivoting pressure-level variables into a wide feature table.
2. Merging surface variables and total precipitation by time and location.
3. Converting precipitation from metres to millimetres per hour.
4. Defining a cloudburst as precipitation above 20 mm/hour.
5. Creating six-hour sliding sequences.
6. Scaling features with a training-only `StandardScaler`.
7. Predicting the event two hours after the final observed timestep.

### Model Architecture

The final model uses a Bidirectional LSTM to learn temporal patterns from both directions within each six-hour sequence:

- Input: 6 timesteps x 26 features
- Bidirectional LSTM: 128 units per direction
- Dense layer: 48 ReLU units
- Dropout: 50 percent
- Output: sigmoid cloudburst probability
- Training objective: focal loss with balanced class weighting

### Validation Strategy

Because weather observations are time-dependent, the project uses temporal rolling-window validation instead of random cross-validation. The final model was evaluated on a held-out 2023 test period after training on earlier available years. This design reduces leakage and more closely represents operational forecasting.

## Findings

The documented dataset contains 61,998 cleaned hourly records and 61,984 six-hour sequences. Cloudburst events represent approximately 1.23 percent of the labelled observations, making the problem highly imbalanced.

The current evaluation on the held-out 2023 test set uses the fixed operational threshold of 0.5, matching the deployed Flask application and current research report:

| Metric | Result |
| --- | ---: |
| Accuracy | 98.2% |
| Precision | 0.72 |
| Recall | 0.68 |
| F1 score | 0.70 |
| ROC-AUC | 0.9751 |
| PR-AUC | 0.4250 |
| Operational threshold | 0.5 |

Accuracy must be interpreted carefully because a model can appear accurate by predicting the majority class. Recall, precision, F1 score, and PR-AUC are more informative for this rare-event warning problem. The model-selection result was a rolling-validation PR-AUC of 0.4375, while 0.4250 is the current held-out 2023 test PR-AUC. The results indicate strong ranking ability and useful detection potential, while also showing that some events remain difficult to detect.

## Reports and Research Material

The repository can be kept focused on the runnable application while the following material remains available as supporting research documentation:

- `project_explanation.md` - detailed dataset, architecture, training, evaluation, and limitations.
- `DATABASE_GUIDE.md` - prediction storage and alert database design.
- `PRODUCTION_README.md` - deployment, API, and operational details.
- `paper_submission/` - manuscript source and publication materials.

Generated thesis, presentation, and temporary office files are intentionally separate from the core software repository. This keeps the GitHub project easier to understand, clone, and maintain.

## How to Use

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Check the model files

Place the trained model and preprocessing artifacts in the project root:

```text
cloudburst_final_bilstm_only.keras
scaler_final.pkl
feature_cols.pkl
```

### 3. Configure API authentication

For protected endpoints, set an API key in the environment:

```bash
# Windows PowerShell
$env:API_KEY = "your-secret-key"

# Linux or macOS
export API_KEY="your-secret-key"
```

### 4. Start the application

```bash
python app.py
```

Open the dashboard at:

```text
http://127.0.0.1:5000
```

### 5. Use the API

A quick prediction is available at `GET /api/prediction`. Feature-based predictions are available at `POST /api/predict`, and live ERA5-style input is accepted at `POST /api/predict_live`.

Example request structure:

```json
{
  "era5": {
    "rows": [
      {"t2m": 290.0, "d2m": 285.0}
    ],
    "timesteps": 6
  }
}
```

The complete endpoint details and request examples are documented in `README.md` and `PRODUCTION_README.md`.

## Limitations and Future Work

- ERA5 has approximately 0.25 degree spatial resolution, which may miss highly local convective processes.
- The test results contain false positives and missed cloudburst events.
- The training record has a documented gap for 2018-2019.
- Operational use requires independent real-time data validation and alert governance.
- Future work can evaluate higher-resolution weather data, additional locations, uncertainty estimates, and continuously updated monitoring.

## Conclusion

This project demonstrates a practical research pipeline for rare-event weather prediction: physically meaningful atmospheric features, temporal deep learning, imbalance-aware training, time-aware validation, and a deployable API. Its strongest contribution is not accuracy alone, but the connection between a reproducible forecasting methodology and an accessible early-warning interface for a vulnerable mountainous region.

The system is intended for research, experimentation, and decision-support development. It should not replace official meteorological warnings or emergency-management procedures.

## Author

**Imad Alam**

## Stay Updated and Join the Community

For research updates, collaboration, and discussion:

- Email: [alam1122imad@gmail.com](mailto:alam1122imad@gmail.com)
- LinkedIn: [Imad Alam](https://www.linkedin.com/in/imad-alam-85b4aa25a)

Contributions, research feedback, reproducibility improvements, and ideas for higher-resolution mountain forecasting are welcome.
