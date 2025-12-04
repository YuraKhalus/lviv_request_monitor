# Project Roadmap: Lviv Request Monitor

This document outlines the architecture, file structure, and implementation plan for the "Lviv Request Monitor" project. The primary goal is to build a containerized microservices application to predict the resolution time for civic appeals in Lviv.

## 1. Project Structure

The project will be organized into separate directories for each service to maintain a clean and scalable microservices architecture.

```
lviv_request_monitor/
├── data/
│   └── hotline_1580.csv           # Raw dataset
├── db/
│   ├── Dockerfile                 # PostgreSQL Dockerfile
│   └── init.sql                   # DB initialization script (creates table)
├── model_service/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI application entrypoint
│   │   ├── schemas.py             # Pydantic schemas for requests/responses
│   │   ├── models.py              # ML model training, prediction, and evaluation logic
│   │   └── utils.py               # Data processing and memory optimization functions
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_api.py            # API endpoint tests
│   ├── Dockerfile                 # Dockerfile for the model service
│   └── requirements.txt           # Python dependencies (FastAPI, scikit-learn, etc.)
├── interface_service/
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py                # Streamlit application entrypoint
│   ├── Dockerfile                 # Dockerfile for the interface service
│   └── requirements.txt           # Python dependencies (Streamlit, pandas, etc.)
├── .dockerignore                  # Files/directories to ignore in Docker builds
├── docker-compose.yml             # Main Docker Compose file to orchestrate all services
└── project_roadmap.md             # This file
```

## 2. Step-by-Step Implementation Plan

### Phase 1: Data Analysis & Memory Optimization

**Goal:** Prepare the raw dataset for use in the database and for model training, with a strong focus on memory efficiency.

1.  **Initial Analysis:**
    *   Load `hotline_1580.csv` into a Pandas DataFrame.
    *   Perform an initial `df.info(memory_usage='deep')` to establish a baseline memory footprint.
    .  Analyze columns for data types, missing values, and cardinality (number of unique values).
2.  **Cleaning Logic (`model_service/app/utils.py`):**
    *   Handle missing values (e.g., fill with a placeholder like 'unknown' or drop rows if appropriate).
    *   Parse date columns (`creation_date`, `registration_date`, `completion_date`) into `datetime` objects.
    *   Create the target variable: `days_to_resolve = (completion_date - creation_date).dt.days`.
3.  **Memory Optimization (`model_service/app/utils.py`):**
    *   **Categorical Data:** Convert high-cardinality string columns (e.g., `category`, `district`, `source`) to the `category` dtype. This is the most critical step for reducing memory usage.
    *   **Numerical Data:** Downcast numerical columns to the smallest possible integer or float subtypes (e.g., `pd.to_numeric(df['column'], downcast='integer')`).
    *   Verify memory reduction with `df.info(memory_usage='deep')`.
4.  **Feature Selection:**
    *   Identify the features most likely to influence `days_to_resolve`. Initially, we'll use `district`, `category`, and potentially the day of the week or month from `creation_date`.

### Phase 2: Database Setup

**Goal:** Create a PostgreSQL service that automatically initializes and loads the cleaned data.

1.  **PostgreSQL Dockerfile (`db/Dockerfile`):**
    *   Use the official `postgres` base image.
    *   The primary role of this Dockerfile is to copy the `init.sql` script into the `docker-entrypoint-initdb.d` directory. This ensures the script runs automatically on the first container startup.
2.  **Initialization Script (`db/init.sql`):**
    *   Define the `appeals` table structure (see Data Schema below).
    *   Use the `COPY` command to efficiently load the `hotline_1580.csv` data into the `appeals` table. This is much faster than `INSERT` statements for bulk data.
3.  **Docker Compose Integration:**
    *   Define the `db` service in `docker-compose.yml`.
    *   Map a volume to persist the PostgreSQL data (`pgdata`).
    *   Set environment variables for the database user, password, and name.

### Phase 3: API Development (Model Service)

**Goal:** Build a FastAPI service to handle model training and predictions.

1.  **Pydantic Schemas (`model_service/app/schemas.py`):**
    *   `AppealInput`: Defines the structure for input data to the `/predict` endpoint (e.g., `district`, `category`).
    *   `PredictionOutput`: Defines the response structure, including predictions from all three models.
    *   `MetricsOutput`: Defines the structure for the `/metrics` endpoint response (RMSE, MAE for each model).
2.  **Model Logic (`model_service/app/models.py`):**
    *   **Training Function:**
        *   Connects to the PostgreSQL database to fetch the training data.
        *   Applies the memory-optimized data processing steps from Phase 1.
        *   Trains three lightweight models:
            *   `LinearRegression`
            *   `RandomForestRegressor(n_estimators=50, max_depth=10)`
            *   `XGBRegressor(n_estimators=50, max_depth=5)`
        *   Serializes and saves the trained models (e.g., using `joblib`).
    *   **Prediction Function:**
        *   Loads the pre-trained models.
        *   Takes the `AppealInput` data, preprocesses it (one-hot encoding), and returns predictions.
    *   **Evaluation Function:**
        *   Calculates RMSE and MAE for each model on a test set.
3.  **API Endpoints (`model_service/app/main.py`):**
    *   `POST /train`: Triggers the model training function. This should be an asynchronous task.
    *   `POST /predict`: Takes JSON input, passes it to the prediction function, and returns the results.
    *   `GET /metrics`: Returns the latest evaluation metrics.
4.  **Dockerfile & Dependencies (`model_service/`):**
    *   Create `requirements.txt` with `fastapi`, `uvicorn`, `scikit-learn`, `xgboost`, `pandas`, `pydantic`, `psycopg2-binary`.
    *   Write `model_service/Dockerfile` to install dependencies and run the FastAPI app with Uvicorn.

### Phase 4: Frontend Implementation (Interface Service)

**Goal:** Develop a simple Streamlit UI for users to interact with the model service.

1.  **UI Components (`interface_service/app/main.py`):**
    *   Use `st.selectbox` for `district` and `category` selection. These options should be fetched dynamically from the model API or a shared configuration.
    *   A "Predict" button to trigger the API call.
    *   Display the prediction results in a clear, user-friendly format (e.g., using `st.metric`).
    *   Add a section to display model performance metrics (from the `/metrics` endpoint) using charts or tables.
2.  **API Integration:**
    *   Use the `requests` library to make HTTP calls to the `model_service` API.
    *   `POST /predict`: Send user input from the widgets.
    *   `GET /metrics`: Fetch and display model performance.
3.  **Dockerfile & Dependencies (`interface_service/`):**
    *   Create `requirements.txt` with `streamlit`, `requests`, `pandas`.
    *   Write `interface_service/Dockerfile` to install dependencies and run the Streamlit app.

### Phase 5: Docker Composition & Orchestration

**Goal:** Configure `docker-compose.yml` to build and run all three services together.

1.  **Service Definitions:**
    *   **`db` service:** Build from `./db`, expose port `5432`, and set up environment variables and volumes.
    *   **`model-api` service:** Build from `./model_service`, expose port `8000`, and set `depends_on: [db]` to ensure the database is ready before the API starts.
    *   **`interface-app` service:** Build from `./interface_service`, expose port `8501`, and set `depends_on: [model-api]`.
2.  **Networking:** Docker Compose will automatically create a default network, allowing services to communicate using their service names (e.g., `http://model-api:8000`).
3.  **Memory & CPU Limits:**
    *   Crucially, add resource limits to each service to prevent overloading the host machine.
    ```yaml
    services:
      model-api:
        # ...
        deploy:
          resources:
            limits:
              cpus: '1.0'
              memory: '2G'
      # ... similar limits for other services
    ```
4.  **Final Commands:** The entire system will be launched with `docker-compose up --build`.

## 3. Data Schema

A simple schema for the `appeals` table in PostgreSQL. Data types are chosen to be efficient.

```sql
CREATE TABLE appeals (
    id SERIAL PRIMARY KEY,
    creation_date TIMESTAMP,
    registration_date TIMESTAMP,
    completion_date TIMESTAMP,
    district VARCHAR(100),
    category VARCHAR(255),
    source VARCHAR(100),
    content TEXT
);
```
