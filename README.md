# Police Station-Level Crime Count Prediction Dashboard

A Streamlit-based dashboard for exploring, evaluating, and interpreting machine-learning models for police station-level quarterly crime-count prediction. The application supports temporally aware model comparison, encoding-scenario evaluation, future crime-count forecasting, and interpretable model outputs using SHAP and LIME.

## Project Overview

This application forms part of a research project on interpretable machine learning for police station-level crime-count prediction. The focus is not to claim certainty about future crime, but to support evidence-informed and probabilistic forecasting decisions using historical quarterly crime-count data.

The dashboard allows users to compare how different machine-learning regression models behave when trained on historical police station-level crime-count data and tested on future held-out periods. It also includes a Seasonal Naïve benchmark to test whether more complex models improve on simple temporal persistence.

## Main Features

- Police station-level quarterly crime-count prediction
- Temporal held-out model evaluation
- Rolling-origin validation support
- Seasonal Naïve benchmark comparison
- Multiple machine-learning model comparison
- Alternative categorical encoding scenarios
- 2026 and 2027 future crime-count forecasts
- Prediction interval display
- Relative hotspot identification using top-decile ranking
- Global and local model explanation using SHAP
- Local prediction explanation using LIME
- User and data scientist dashboard views
- Model performance visualisation
- Forecast and prediction-result exploration

## Research Focus

The dashboard supports the following research objective:

> To evaluate how selected machine-learning regression models behave under temporal held-out testing when applied to police station-level quarterly crime-count prediction across alternative categorical encoding scenarios, using Seasonal Naïve forecasting as a transparent benchmark.

The system is designed to separate prediction outcome from modelling-decision quality. A forecast may be judged by its closeness to the observed count, but the quality of the modelling decision also depends on whether the method uses only historically available data, is tested on unseen future periods, is compared against a transparent benchmark, and produces interpretable signals that analysts can examine.

## Machine-Learning Models

The application supports comparison of the following regression models:

| Abbreviation | Model |
|---|---|
| RFM | Random Forest Model |
| XGBR | Extreme Gradient Boosting Regressor |
| KNNR | K-Nearest Neighbor Regressor |
| SVR | Support Vector Regression |
| MLPR | Multi-Layer Perceptron Regressor |
| SNaïve | Seasonal Naïve benchmark |

## Encoding Scenarios

The application evaluates model behaviour across alternative categorical encoding scenarios.

| Scenario | Police Station Encoding | Quarter Encoding |
|---|---|---|
| Scenario 1 | Label encoding | Label encoding |
| Scenario 2 | One-hot encoding | One-hot encoding |
| Scenario 3 | Label encoding | One-hot encoding |
| Scenario 4 | One-hot encoding | Label encoding |

Crime category is retained as a categorical modelling feature and encoded consistently for model training and interpretation.

## Dataset Scope

The dashboard uses quarterly police station-level crime-count data. The modelling design is based on historical data arranged by:

- Police station
- Crime category
- Year
- Quarter
- Historical lag values
- Recent count differences
- Recent percentage changes
- Historical summary statistics

The system uses a time-aware evaluation structure to avoid random leakage from future periods into training data.

## Temporal Evaluation Design

The modelling workflow follows a temporal structure:

| Period | Purpose |
|---|---|
| 2019-2023 | Model development and rolling-origin training |
| 2024 | Validation and prediction interval calibration |
| 2025 | Final held-out testing |
| 2026 | One-step-ahead future forecast |
| 2027 | Recursive future forecast |

This design tests whether a model trained on past data can make reasonable predictions for later unseen periods.

## Evaluation Metrics

The dashboard reports several regression evaluation metrics:

| Metric | Purpose |
|---|---|
| MAE | Measures average absolute prediction error |
| MSE | Measures average squared prediction error |
| RMSE | Main model-ranking metric; penalises larger errors |
| R² | Measures explained variance |
| Adjusted R² | Adjusts R² for feature count |
| MAPE | Measures percentage prediction error |
| sMAPE | Symmetric percentage prediction error |
| Bias | Indicates systematic overprediction or underprediction |
| Prediction interval coverage | Evaluates uncertainty calibration |

## Interpretability

The dashboard includes post-hoc model explanation techniques.

### SHAP

SHAP is used to explain global and local feature contributions. It helps identify which features contribute most strongly to model predictions, such as historical lag values, recent percentage change, target year, crime category indicators, and police station indicators.

### LIME

LIME is used to explain individual predictions locally. It helps show how nearby feature values influence a specific forecast.

Interpretability outputs should be treated as model-behaviour explanations, not as causal explanations of crime.

## Forecasting and Hotspot Interpretation

The application generates future crime-count predictions and supports relative hotspot identification.

A hotspot in this system refers to a police station that falls within the top-decile forecasted count for a specific:

- Year
- Quarter
- Crime category
- Police station comparison group

The hotspot output is therefore a relative ranking based on forecasted counts. It should not be interpreted as a precise incident-level risk prediction or a causal claim.

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend dashboard | Streamlit |
| Backend API | FastAPI |
| Database | Microsoft SQL Server |
| Machine learning | scikit-learn, XGBoost |
| Explainability | SHAP, LIME |
| Data processing | pandas, NumPy |
| Visualisation | matplotlib, Streamlit charts |

## Suggested Project Structure

```text
crime-count-prediction-dashboard/
│
├── app/
│   ├── main.py
│   ├── pages/
│   │   ├── 1_User_View.py
│   │   ├── 2_Data_Scientist_View.py
│   │   ├── 3_Model_Performance.py
│   │   ├── 4_Forecasts.py
│   │   └── 5_Interpretability.py
│   │
│   ├── components/
│   │   ├── charts.py
│   │   ├── filters.py
│   │   └── tables.py
│   │
│   └── utils/
│       ├── config.py
│       ├── database.py
│       ├── preprocessing.py
│       ├── metrics.py
│       └── explainability.py
│
├── api/
│   ├── main.py
│   ├── routes/
│   ├── services/
│   └── models/
│
├── models/
│   ├── trained_models/
│   ├── encoders/
│   └── scalers/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
│
├── notebooks/
│   ├── model_training.ipynb
│   ├── evaluation.ipynb
│   └── interpretability.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/crime-count-prediction-dashboard.git
cd crime-count-prediction-dashboard
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Example Requirements

Your `requirements.txt` may include:

```text
streamlit
pandas
numpy
scikit-learn
xgboost
shap
lime
matplotlib
plotly
sqlalchemy
pyodbc
fastapi
uvicorn
python-dotenv
joblib
```

## Configuration

Create a `.env` file in the project root.

```env
DB_SERVER=your_sql_server
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password
API_BASE_URL=http://localhost:8000
```

Do not commit the `.env` file to GitHub.

## Running the FastAPI Backend

Start the API service:

```bash
uvicorn api.main:app --reload
```

The API should be available at:

```text
http://localhost:8000
```

The Swagger documentation should be available at:

```text
http://localhost:8000/docs
```

## Running the Streamlit Dashboard

Start the Streamlit application:

```bash
streamlit run app/main.py
```

The dashboard should open in your browser at:

```text
http://localhost:8501
```

## Dashboard Views

### User View

The user view is designed for exploring predictions and forecasts. It allows users to filter by:

- Year
- Quarter
- Police station
- Crime category
- Prediction period
- Hotspot ranking

### Data Scientist View

The data scientist view is designed for model evaluation and diagnostics. It includes:

- Model comparison
- Scenario comparison
- Error metrics
- Rolling-origin results
- Held-out test results
- SHAP explanations
- LIME explanations
- Prediction interval coverage

## Model Workflow

The application follows this modelling workflow:

1. Load historical quarterly crime-count data.
2. Validate the schema and clean missing or inconsistent values.
3. Create lag-based and historical summary features.
4. Apply scenario-specific categorical encoding.
5. Train selected machine-learning models.
6. Compare models using rolling-origin validation.
7. Validate and calibrate using 2024 data.
8. Test final model behaviour on held-out 2025 data.
9. Generate 2026 and 2027 forecasts.
10. Explain selected predictions using SHAP and LIME.
11. Display results in the Streamlit dashboard.

## Important Interpretation Notes

This application predicts quarterly crime counts, not individual criminal events.

The predictions should be interpreted as decision-support outputs, not as deterministic statements about future crime. A model may indicate that a count is likely to increase or decrease based on historical patterns, but the prediction remains uncertain and should be considered together with operational knowledge, policing context, and domain expertise.

The model does not establish the causes of crime. SHAP and LIME explain model behaviour, not causal relationships.

## Current Key Finding

In the research evaluation, the Seasonal Naïve benchmark produced the strongest overall held-out performance. Among trained machine-learning models, XGBR under Scenario 4 produced the strongest temporal held-out performance, with RFM Scenario 4 remaining a close alternative.

This finding suggests that annual persistence is a strong signal in police station-level quarterly crime-count data, and that complex machine-learning models should be evaluated against simple temporal benchmarks before being used for future forecasting.

## Limitations

- The system uses aggregated quarterly police station-level crime counts.
- The predictions do not represent incident-level risk.
- Model outputs depend on the quality and consistency of the historical data.
- Forecasts are probabilistic and should not be treated as certain.
- Interpretability methods explain model behaviour, not causal drivers of crime.
- Future crime counts may be affected by social, economic, policing, and reporting changes not captured in the dataset.

## Future Improvements

Possible improvements include:

- Adding more external contextual variables
- Improving prediction interval calibration
- Extending station-level diagnostic reporting
- Adding model drift monitoring
- Adding automated retraining workflows
- Adding user feedback loops
- Improving hotspot visualisation
- Adding downloadable reports
- Deploying the system using Docker and cloud hosting

## Ethical Use

This system should be used responsibly. Forecasts should support planning, analysis, and resource-awareness discussions, but should not be used as the sole basis for enforcement decisions. Any operational use should include human review, transparency, fairness considerations, and ongoing evaluation.

## Author

Charles Nolte  
Cape Peninsula University of Technology  
Email: cnoltehj@yahoo.com

## License

This project is intended for academic and research purposes. Add a formal license file if the repository will be shared publicly.

Suggested license options:

- MIT License for open-source reuse
- Creative Commons license for research documentation
- Private repository license if the code is not intended for public use

## Acknowledgements

This project forms part of postgraduate research on temporal machine-learning evaluation and interpretable police station-level crime-count prediction.
