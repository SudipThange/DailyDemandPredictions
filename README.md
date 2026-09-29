# Supply Chain Daily Demand Prediction

## 1. Project Description

This project predicts **daily demand for an oil supply chain** using a structured, end-to-end machine-learning pipeline built entirely on the Databricks Lakehouse platform. The business problem is straightforward but high-impact: without accurate daily demand forecasts, supply chain operators risk either overstocking (tying up capital and increasing storage costs) or understocking (leading to missed deliveries and customer dissatisfaction). By leveraging historical transaction data, time-series feature engineering, and a Random Forest regression model, this project produces data-driven daily demand predictions that can inform inventory planning, logistics scheduling, and procurement decisions.

The overall workflow follows a **medallion data architecture** (Bronze → Silver → Gold), progresses through feature engineering and Feature Store registration, and culminates in model training with MLflow experiment tracking, model registration in Unity Catalog, and a dedicated prediction pipeline. Every stage is implemented as a Databricks notebook, making the pipeline reproducible and modular.

### Technologies & Platforms

| Category | Technology |
| --- | --- |
| Platform | Databricks (Serverless compute) |
| Data Governance | Unity Catalog |
| Storage Format | Delta Lake tables |
| Feature Management | Databricks Feature Store |
| Data Processing | PySpark |
| Machine Learning | Scikit-learn (Random Forest Regressor) |
| Experiment Tracking | MLflow |
| Model Registry | Unity Catalog Model Registry |

### High-Level Data Flow

```text
Raw Oil CSV
    → Bronze (raw ingestion)
    → Silver (cleaning & deduplication)
    → Gold (daily aggregation)
    → Feature Engineering (lag, moving average, calendar features)
    → Feature Store (feature & label tables)
    → Data Preparation (join, time-series split, encoding)
    → Model Training (Random Forest Regressor)
    → MLflow (experiment tracking & artifact logging)
    → Model Registry (best model registration in Unity Catalog)
    → Prediction (load model + encoder, generate & evaluate predictions)
```

---

## 2. How the Project Works

### 2.1 Raw Data

The project begins with an **oil supply chain CSV dataset** containing many columns — some relevant for analysis and prediction, others not. The raw file is uploaded to a Unity Catalog volume for reliable, governed storage.

A dedicated catalog named **`supply_chain_daily_demand`** is created in Unity Catalog. Inside this catalog, a **`raw`** schema holds the original CSV file in a volume, preserving the data exactly as received before any transformation.

### 2.2 Bronze Layer

The Bronze notebook (`01_bronze_ingestion`) is responsible for loading the raw CSV data from the volume into the lakehouse:

* Reads the original raw CSV file from the Unity Catalog volume.
* Loads the data into a PySpark DataFrame.
* Writes the raw copy as a **Delta table** in the Bronze schema.

**Bronze table:**

```
supply_chain_daily_demand.bronze.daily_demand_bronze
```

The Bronze layer maintains a **raw, near-source representation** of the data. No filtering, cleaning, or transformation is applied at this stage — the goal is to land the data as-is so that all downstream layers have a faithful baseline to work from.

### 2.3 Silver Layer

The Silver notebook (`02_silver_transformation`) takes the Bronze data and applies cleaning transformations:

* Loads the Bronze Delta table.
* Removes duplicate records to ensure data integrity.
* Writes the cleaned data as a Delta table in the Silver schema.

**Silver table:**

```
supply_chain_daily_demand.silver.daily_demand_transformed
```

The Silver layer serves as the **cleaned and transformed** version of the Bronze data. It enforces basic data quality (deduplication) and prepares the dataset for business-level aggregation in the Gold layer.

### 2.4 Gold Layer

The Gold notebook (`03_gold_aggregations`) aggregates the cleaned Silver data into a **business-level daily demand dataset** suitable for analytics and machine learning.

**Group-by columns:**

* `transaction_date`
* `product_name`
* `destination_city`

**Aggregation logic:**

| Source Column | Aggregation | Output Column |
| --- | --- | --- |
| `total_demand` | `SUM()` | `total_demand` |
| `avg_unit_price` | `AVG()` | `avg_unit_price` |
| `total_inventory` | `SUM()` | `total_inventory` |

**Gold table:**

```
supply_chain_daily_demand.gold.aggregated_daily_demand
```

The Gold layer contains **business-ready, aggregated data** at the daily grain. By collapsing raw transaction records into daily summaries keyed by product and city, the Gold table provides a clean, compact foundation for feature engineering and model training.

### 2.5 Feature Engineering

Using the Gold table `aggregated_daily_demand`, a feature table is created with additional **time-series features** that help the model learn temporal demand patterns:

| Feature | Description | Why It's Useful |
| --- | --- | --- |
| `demand_lag_1` | Demand value from the previous day | Captures short-term momentum and day-to-day correlation in demand |
| `demand_lag_7` | Demand value from 7 days ago | Captures weekly seasonality (e.g., same weekday last week) |
| `moving_avg_7` | 7-day rolling average of demand | Smooths noise and reveals the underlying trend over the past week |
| `day_of_week` | Day of the week (0–6) | Captures day-of-week seasonality (e.g., weekday vs. weekend patterns) |
| `month` | Month number (1–12) | Captures monthly or seasonal cycles throughout the year |

**Resulting table:**

```
features_daily_demand
```

These features give the Random Forest model both recent temporal context (lags, moving average) and calendar awareness (day of week, month), which are essential for accurate daily demand forecasting.

---

## 3. Feature Store Workflow

The feature table `features_daily_demand` is loaded into the **Databricks Feature Store** to make features discoverable, reusable, and properly governed for machine learning.

A **primary key** is added to uniquely identify each feature record. The dataset is then divided into two logical tables, both stored in the **`features`** schema:

### Feature Table

* **Primary key** — unique identifier for each row
* **All model input features** — `demand_lag_1`, `demand_lag_7`, `moving_avg_7`, `day_of_week`, `month`, and other engineered columns

### Label Table

* **Primary key** — same key as the feature table, enabling a join
* **Target variable (`y`)** — the actual daily demand value the model learns to predict

### Why Separate Features and Labels?

Separating features and labels follows Feature Store best practices:

* **Reusability** — features can be shared across multiple models or experiments without duplicating the label.
* **Governance** — access to the target variable can be restricted independently from features.
* **Clean joins** — the primary key acts as a join column, allowing the training dataset to be assembled deterministically by joining the two tables at training time.

---

## 4. Data Engineering / Model Training Workflow

### 4.1 Joining Features and Labels

The feature table and label table are loaded and **joined using the primary key** to create a single, unified training dataset. This joined dataset contains every feature column alongside the target variable, ready for model preparation.

### 4.2 Time-Series Data Split

Because this is **time-series data**, the records must maintain **chronological order**. A random train-test split would leak future information into the training set and destroy the temporal structure the model needs to learn. Instead, the data is split sequentially by time:

* The **latest 20%** of the data is reserved for final testing.
* The **historical 80%** is further divided into training and validation.

The final split is:

| Split | Percentage | Purpose |
| --- | --- | --- |
| `validation_training` | 64% | Used to train the model |
| `validation_testing` | 16% | Used to validate model performance during development |
| `testing_set` | 20% | Latest observations — kept for final, unbiased testing |

```text
Historical Data
├── 64% → Training (validation_training)
├── 16% → Validation (validation_testing)
└── 20% → Final Testing (testing_set)
```

This approach ensures the model is evaluated on data it has never seen, while preserving the temporal order that is critical for demand forecasting.

### 4.3 Categorical Encoding

Categorical columns are converted into numerical representations using **One-Hot Encoding**.

Crucially, the encoder is fitted **only on the training data** (`validation_training`):

| Dataset | Encoder Operation |
| --- | --- |
| `validation_training` | `fit_transform()` — the encoder learns the categories and transforms the data |
| `validation_testing` | `transform()` — the fitted encoder is applied without re-learning |
| `testing_set` | `transform()` — same fitted encoder is reused |

**Why fit only on training data?** Fitting the encoder on the full dataset (including validation or test data) would constitute **data leakage** — the model would indirectly see information from the validation/test sets through the encoder's vocabulary. By fitting only on `validation_training`, we ensure that the encoding step respects the train/validation/test boundary.

### 4.4 X and Y Preparation

After encoding, each dataset is divided into features (`X`) and target (`Y`):

| Dataset | Features | Target |
| --- | --- | --- |
| Training | `X_train` | `Y_train` |
| Validation | `X_validation` | `Y_validation` |
| Testing | `X_test` (kept separate) | `Y_test` (kept separate) |

The testing set is **never used during model training or hyperparameter tuning**. It is reserved exclusively for the final prediction and evaluation workflow.

---

## 5. Model Training and MLflow

### Model Training

The model is trained using `X_train` and `Y_train`, and evaluated using `X_validation` and `Y_validation`.

The project uses a **Random Forest Regressor** from scikit-learn.

**Key model parameters:**

| Parameter | Purpose |
| --- | --- |
| `n_estimators` | Number of decision trees in the forest. More trees generally improve accuracy but increase training time. |
| `random_state` | Seed for reproducibility. Ensures the same training data produces the same model every time. |
| `n_jobs` | Number of CPU cores to use for parallel training. `-1` uses all available cores, speeding up training. |

### MLflow Experiment Tracking

**MLflow** is used to track the complete experiment lifecycle. For each training run, the following are logged:

* **Parameters** — model hyperparameters (`n_estimators`, `random_state`, `n_jobs`, etc.)
* **Metrics** — validation metrics (e.g., RMSE, MAE, R²) used to compare runs
* **Model** — the trained Random Forest model artifact
* **Encoder artifact** — the fitted One-Hot Encoder, saved for later use in the prediction pipeline
* **Other relevant artifacts** — any additional objects needed to reproduce the run

MLflow provides a centralized, versioned record of every experiment, making it possible to compare runs, reproduce results, and select the best-performing model with confidence.

### Best Model Selection

After multiple MLflow runs, the **best-performing model** is identified based on the recorded validation metrics. The selected model is **registered in the Unity Catalog Model Registry** under the `ml_model` catalog/schema structure, making it versioned, discoverable, and ready for deployment.

Additionally, the **three datasets** (`validation_training`, `validation_testing`, `testing_set`) are stored for reference in their **original, non-encoded form**. This allows the prediction pipeline to apply the saved encoder at inference time rather than relying on pre-encoded data.

---

## 6. Model Prediction Workflow

The prediction pipeline is a **separate workflow** from model training. It reuses artifacts saved during training to generate and evaluate predictions on the held-out testing set.

### Prediction Steps

1. **Load the saved encoder** — the One-Hot Encoder artifact is loaded from the MLflow run where the best model was trained.
2. **Load the non-encoded dataset** — the original (pre-encoding) testing set is loaded from storage.
3. **Apply the saved encoder** — categorical columns are passed through the **same fitted encoder** from training.
4. **Generate encoded feature columns** — the encoder produces the numerical representations needed by the model.
5. **Separate into `X` and `Y`** — the encoded dataset is split into features (`X`) and target (`Y`).
6. **Load the registered model** — the best model is loaded from the Unity Catalog Model Registry via MLflow.
7. **Generate predictions** — `model.predict(X)` produces demand forecasts.
8. **Compare predictions against `Y`** — predicted values are compared to actual demand values.
9. **Calculate evaluation metrics** — metrics such as RMSE, MAE, and R² are computed to assess model performance.

### Critical: Reuse the Training Encoder

The prediction pipeline **must use the same encoder that was fitted during model training**. It must **never fit a new encoder** on the prediction data. Fitting a new encoder would:

* Produce a different feature space than the one the model was trained on, causing mismatched input dimensions.
* Introduce data leakage by learning category mappings from the test set.
* Invalidate any performance metrics, as the model would be evaluated on inconsistently encoded data.

---

## 7. Project Folder / File Structure

```text
supply_chain_project/
│
├── README.md                          ← This file
│
├── data_preparation/                  ← Medallion architecture + feature engineering
│   ├── 00_data_preparation            ← Runs other files 
│   ├── bronze/
│   │   └── 01_bronze_ingestion         ← Load raw CSV → Bronze Delta table
│   ├── silver/
│   │   └── 02_silver_transformation    ← Clean & deduplicate → Silver Delta table
│   ├── gold/
│   │   └── 03_gold_aggregations        ← Aggregate to daily grain → Gold Delta table
│   └── feature_store/
│       └── 04_feature_store            ← Feature engineering + Feature Store registration
│
├── model_training/
│   └── 05_model_training               ← Join, split, encode, train, MLflow logging
│
├── model_predictions/
    └── 06_model_predictions            ← Load model + encoder, predict, evaluate

```

### Notebook Execution Order

| Step | Notebook | Description |
| --- | --- | --- |
| 0 | `00_data_preparation` | Create Unity Catalog, schemas, and volumes; upload raw CSV |
| 1 | `01_bronze_ingestion` | Ingest raw CSV into Bronze Delta table |
| 2 | `02_silver_transformation` | Clean and deduplicate into Silver Delta table |
| 3 | `03_gold_aggregations` | Aggregate to daily demand in Gold Delta table |
| 4 | `04_feature_store` | Engineer features and register in Feature Store |
| 5 | `05_model_training` | Prepare data, train Random Forest, log to MLflow |
| 6 | `06_model_predictions` | Load best model, generate and evaluate predictions |

---

## Unity Catalog Table Summary

| Layer | Schema | Table Name |
| --- | --- | --- |
| Raw | `raw` | *(CSV in volume)* |
| Bronze | `bronze` | `daily_demand_bronze` |
| Silver | `silver` | `daily_demand_transformed` |
| Gold | `gold` | `aggregated_daily_demand` |
| Features | `features` | Feature table + Label table |
| Model Registry | `ml_model` | Registered best model |

All tables are under the catalog: **`supply_chain_daily_demand`**
