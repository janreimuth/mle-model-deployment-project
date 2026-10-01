## Florida Red Tide Early Warning

A harmful algal bloom dataset for Karenia brevis monitoring, geospatial analysis, time-series exploration, and red tide classification.

### About the Dataset
This dataset contains historical laboratory observations of Karenia brevis, the marine microorganism associated with Florida red tide and other harmful algal bloom events in the Gulf of Mexico.

It includes 211,834 observations collected between 1953 and 2024 along the coasts of Florida, Texas, Alabama, and Mississippi. The original observations come from the NOAA Harmful Algal BloomS Observing System (HABSOS).

In addition to the cleaned observations, the dataset provides an ML-ready benchmark for predicting whether Karenia brevis concentrations will reach bloom level within the next seven days.

Source: https://www.kaggle.com/datasets/samoilovmikhail/florida-red-tide-early-warning-noaa-habsos/data

### Import Dataset

```
uv pip install kagglehub # kagglehub is not preinstalled in virtual environment
```

```
import os
from pathlib import Path

# Set KaggleHub cache to ./data
data_dir = Path.cwd() / "data/raw"
data_dir.mkdir(parents=True, exist_ok=True)

os.environ["KAGGLEHUB_CACHE"] = str(data_dir)

import kagglehub

# Download latest version
path = kagglehub.dataset_download("samoilovmikhail/florida-red-tide-early-warning-noaa-habsos")

print("Path to dataset files:", path)
```


# HAB Early Warning Model

 ## Project Overview

 This project builds a machine-learning early-warning system for harmful algal bloom (HAB) escalation. 
 The original project workflow was designed around taxi trip-duration prediction. For this project, the same machine-learning deployment workflow is applied to the supplied HAB early-warning benchmark.
 The model predicts the probability that a monitoring location will experience a bloom during the following 7 days.

 ## Dataset

 The project uses the supplied HAB benchmark data:

 - `early_warning_train.csv`
- `early_warning_validation.csv`
- `early_warning_test.csv`
- `early_warning_benchmark.csv`
- `early_warning_benchmark.parquet`
- `feature_columns.txt`
- `habsos_observations.csv`
- `habsos_observations.parquet`

 The training and validation datasets are provided as separate benchmark splits.

 Dataset sizes used in this project:

 - Training: 32,304 rows
- Validation: 8,668 rows
- Features: 49

 ## Target Variable

 The target variable is:

```
target_bloom_next_7d
```

 It indicates whether a bloom occurs during the following 7 days. 
 This makes the task a binary classification problem.
 The model therefore uses a `RandomForestClassifier` rather than the `RandomForestRegressor` from the original taxi-duration example.

 ## Important Benchmark Consideration

 A missing observation does not mean that no bloom occurred.
 The benchmark is designed around observed follow-up samples. Locations without an appropriate future observation are not automatically treated as negative examples.
 This makes the task an early-warning problem rather than a contemporaneous bloom detector.

 The model predicts future bloom probability using information available at the prediction date.

 ## Feature Preparation

 The approved model features are defined in:

```
data/feature_columns.txt
```

 There are 49 model features.

 Feature preparation is implemented in:

```
src/features.py
```

 The same feature preparation module is used by the training workflow and the API.
 This avoids duplicating feature-selection logic and reduces the risk of differences between training and serving.

 ### Missing Values

 The dataset contains missing environmental measurements.
 The training data contains 119,093 missing feature values and the validation data contains 29,590.

 Missing values are handled with:

```
SimpleImputer(strategy="median")
```

 The imputer is fitted only on the training data and then applied to validation and API data.
 The model also includes the supplied missingness indicator features such as:

 - `water_temp_missing`
- `salinity_missing`
- `wind_speed_missing`

 This allows the model to distinguish an imputed value from an originally observed measurement.

 ## Preventing Data Leakage

 Future target and audit columns are excluded from the model features.

 The following columns are not used as model inputs:

```
target_bloom_next_7d
target_severity_next_7d
target_max_cell_count_next_7d
target_log1p_max_cell_count_next_7d
target_observation_days_next_7d
days_to_next_observation
```

 Historical features are used because they represent information available before the prediction date.
 Future observation counts are used only for target evaluation and auditing.

 ## Model

 The baseline model is a scikit-learn pipeline consisting of:

```
SimpleImputer(strategy="median")
        ↓
RandomForestClassifier
```

 Random Forest parameters:

```
n_estimators = 200
random_state = 42
n_jobs = -1
```

 The model returns the probability of a bloom during the next 7 days:

```
model.predict_proba(X)[:, 1]
```

 The output is a probability rather than a hard yes/no classification.
 This allows an operational decision threshold to be selected separately from the model.

 ## Evaluation

 The model was evaluated on the supplied validation set using RMSE between the observed binary outcome and the predicted bloom probability.
 Final validation result:

```
RMSE = 0.1871
```

 The earlier baseline without the explicit imputation pipeline achieved:

```
RMSE = 0.1876
```

 The final deployed model is the pipeline with median imputation and Random Forest, with a validation RMSE of **0.1871**.

 ## MLflow

 The final training run is logged with MLflow.

 Logged parameters include:

 - model type
- number of estimators
- random state
- preprocessing method
- number of features

 The validation RMSE is logged as a metric.
 The trained pipeline is also logged as an MLflow model artifact.
 Because MLflow/skops performs type safety checks when serializing the sklearn model, the model logging configuration explicitly trusts the model types used by this self-created Random Forest pipeline:

```
skops_trusted_types=[
    "sklearn.tree._tree.Tree",
    "numpy.dtype",
]
```

 ## Model Persistence

 Reusable model functions are implemented in:

```
src/model.py
```

 The module provides functions to:

 - create the model pipeline
- save the model
- load the model

 The final model is saved as:

```
models/early_warning_model.joblib
```

 The saved object contains both the preprocessing and the Random Forest model.
 Therefore, the API does not need to reproduce the training-time imputation logic separately.

 ## API

 The prediction API is implemented with FastAPI:

```
api/main.py
```

 ### Health Check

```
GET /
```

 Example response:

```
{
  "status": "ok"
}
```

 ### Prediction

```
POST /predict
```

 The endpoint accepts the 49 model features and returns the predicted probability of bloom occurrence during the next 7 days.

 Example response:

```
{
  "bloom_probability": 0.095
}
```

 ## Running the API

 Activate the project virtual environment:

```
source .venv/bin/activate
```

 Start the API from the project root:

```
python -m uvicorn api.main:app --reload
```

 The API runs at:

```
http://127.0.0.1:8000
```

 Interactive API documentation is available at:

```
http://127.0.0.1:8000/docs
```

 ## API Verification

 A validation row was sent to the running API.

 The prediction from the trained model and the prediction returned by the API were:

```
Direct model: 0.095
API:          0.095
Difference:   0.0
```

 This confirms that the saved model and API produce the same prediction for the same input.

 ## Project Structure

```
mle-model-deployment-project/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   ├── early_warning_train.csv
│   ├── early_warning_validation.csv
│   ├── early_warning_test.csv
│   ├── early_warning_benchmark.csv
│   ├── early_warning_benchmark.parquet
│   ├── feature_columns.txt
│   ├── habsos_observations.csv
│   └── habsos_observations.parquet
│
├── models/
│   └── early_warning_model.joblib
│
├── notebooks/
│   └── ...
│
├── src/
│   ├── __init__.py
│   ├── features.py
│   └── model.py
│
├── README.md
└── requirements.txt


```
Benchmark data
      ↓
Feature preparation
      ↓
Median imputation
      ↓
Random Forest
      ↓
Validation RMSE: 0.1871
      ↓
MLflow
      ↓
Saved model
      ↓
FastAPI
      ↓
POST /predict
      ↓
Bloom probability
```