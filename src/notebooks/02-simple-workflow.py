# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-pycharm
#     notebook_metadata_filter: all,-pycharm
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.10.16
# ---

# %% [markdown]
# # Simple Workflow with MLFlow and S3

# %% [markdown]
# ## Step 1: Set up environment variables for AWS S3 and MLFlow
#

# %% ExecuteTime={"end_time": "2025-05-11T00:18:07.455756Z", "start_time": "2025-05-11T00:18:07.447900Z"}
from IPython.core.display_functions import clear_output
# %env AWS_ACCESS_KEY_ID=admin
# %env AWS_SECRET_ACCESS_KEY=admin123
# %env MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

# %% [markdown]
# ## Step 2: Import necessary libraries and set up MLFlow

# %% ExecuteTime={"end_time": "2025-05-11T00:18:07.585247Z", "start_time": "2025-05-11T00:18:07.477380Z"}
import random

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from IPython.display import clear_output

experiment_name = "Simple Workflow"
model_name = "diabetes-model"
mlflow.set_tracking_uri(uri="http://localhost:5000")
experiment = mlflow.set_experiment(experiment_name)

print(f"Experiment Name: {experiment.name} with id: {experiment.experiment_id}")

latest_run = None
latest_metrics = {
    "test_rmse": None,
    "test_mae": None,
    "test_r2": None
}
res = []
client = MlflowClient()


# %% [markdown]
# ### Load latest metrics and display them

# %% ExecuteTime={"end_time": "2025-05-11T00:18:07.663005Z", "start_time": "2025-05-11T00:18:07.600137Z"}

if experiment:
    res = client.search_runs(
        experiment.experiment_id,
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

if len(res) > 0:
    latest_run = res[0]
    # Get metrics from latest run (handle missing metrics gracefully)
    latest_metrics["test_rmse"] = latest_run.data.metrics.get("test_rmse")
    latest_metrics["test_mae"] = latest_run.data.metrics.get("test_mae")
    latest_metrics["test_r2"] = latest_run.data.metrics.get("test_r2")

    print(f"Comparing next run with latest run: {latest_metrics}")

# %% [markdown]
# ### Load diabetes dataset

# %% ExecuteTime={"end_time": "2025-05-11T00:18:07.688139Z", "start_time": "2025-05-11T00:18:07.679177Z"}
dataset = datasets.load_diabetes()

# %% [markdown]
# ### Define helper functions

# %% ExecuteTime={"end_time": "2025-05-11T00:18:07.708862Z", "start_time": "2025-05-11T00:18:07.702544Z"}
from typing import Dict


def compare_metrics(client: MlflowClient, current_run_id: str, baseline_run_id: str, metrics_to_compare: Dict[str, str]):
    """
    Compare the performance of two runs based on given metrics.

    Args:
        client (object): Client object to interact with the system.
        current_run_id (str): ID of the current run.
        baseline_run_id (str): ID of the baseline run.
        metrics_to_compare (dict): Dictionary of metrics to compare, where each key is a metric name and its value is a string indicating whether "higher" or "lower" performance is better.

    Returns:
        dict: A dictionary containing boolean values for each metric in `metrics_to_compare`, indicating whether the current run performed better than the baseline run.
    """

    # Get the metrics data from the runs
    current_run = client.get_run(current_run_id)
    baseline_run = client.get_run(baseline_run_id)

    current_metrics = current_run.data.metrics
    baseline_metrics = baseline_run.data.metrics

    # Initialize a dictionary to store the comparison results
    improvement = {metric: False for metric in metrics_to_compare}

    # Compare each metric
    for metric, direction in metrics_to_compare.items():
        if direction.lower() == "higher":
            current_improvement = current_metrics.get(metric) > baseline_metrics.get(metric)
        elif direction.lower() == "lower":
            current_improvement = current_metrics.get(metric) < baseline_metrics.get(metric)
        else:
            raise ValueError(f"Invalid comparison direction for metric '{metric}'. Use 'higher' or 'lower'.")

        improvement[metric] = current_improvement

    return improvement


def generate_random_run_name():
    """
    Generates a random (semi) aviation-related name in slug form.

    Returns:
        str: A string with the name in lowercase, containing an adjective and noun, followed by a 3-digit suffix.
    """
    adjectives = [
        "Transonic", "Hypersonic", "Afterburning", "Turbocharged",
        "Supersonic", "Machbreaking", "Scramjet", "Thrustvectored",
        "Stratospheric", "Tropospheric", "Cloudpiercing", "Jetstreamed",
        "Contrailswept", "Headwinded", "Tailwinded", "Crosswinded",
        "Flybywire", "Autothrottle", "Glasscockpit", "Headup",
        "Skybound", "Runwaylit", "Aileronrolled", "Flapsdown",
        "Chocksaway", "Clearedfortakeoff", "Finalapproach", "Goaround",
        "Quantum", "Neural", "Plasma", "Gravitic", "Singularity",
        "Nanotech", "Exo", "Hyperspace", "Photon", "Cloaking",
        "Tachyon", "Warp", "Zero-G", "Cybernetic", "Holographic",
        "Ion", "Antimatter", "Bioengineered", "Psi", "Chronojump"
    ]

    nouns = [
        "Turbofan", "Tailfin", "Flaps", "Ailerons", "Elevator", "Rudder",
        "Spoilers", "Slats", "Throttle", "Yawdamper", "Stick", "Pedals",
        "Tarmac", "Hangar", "Airstrip", "Runway", "Taxiway", "Apron",
        "Jetbridge", "Windsock", "Glideslope", "Localizer", "Gliderail",
        "Flightdeck", "Blackbox", "Transponder", "Squawkbox",
        "Takeoff", "Landing", "Approach", "Holdingpattern",
        "Jumpgate", "Thruster", "Pulsejet", "Shield", "Wormhole",
        "Drone", "Neurohelm", "Gravcoil", "Phasewings", "Starfighter",
        "Titanium", "Voidship", "Lasercannon", "AI", "Stasis",
        "Dyson", "Warpcore", "Omnitool", "Singularity", "Hoverpad"
    ]

    random_adjective = random.choice(adjectives).lower()
    random_noun = random.choice(nouns).lower()

    # Generate a 3-digit suffix
    suffix = str(random.randint(100, 999))

    return f"{random_adjective}-{random_noun}-{suffix}"


# %% [markdown]
# ## Step 3: Randomize Hyperparameters for Model Training

# %% ExecuteTime={"end_time": "2025-05-11T00:18:15.849454Z", "start_time": "2025-05-11T00:18:07.721250Z"}
from mlflow.entities import RunStatus

# Define the hyperparameter ranges
n_estimators_range = (10, 500)
max_depth_range = (5, 20)
max_features_range = (2, 10)
min_samples_leaf_range = (1, 50)

max_search_attempts = 5

# Create a parent run for all the attempts
with mlflow.start_run(
        run_name=generate_random_run_name(),
) as parent_run:
    # Enable MLflow's automatic experiment tracking for scikit-learn
    mlflow.sklearn.autolog()

    print(f"Starting hyperparameter search under parent run {parent_run.info.run_id}")

    best_run = None
    improvement_found = False
    for attempt in range(1, max_search_attempts + 1):
        print(f"\n=== Attempt {attempt}/{max_search_attempts} ===")

        # Randomize the hyperparameters
        n_estimators = random.randint(*n_estimators_range)
        max_depth = random.randint(*max_depth_range)
        max_features = random.randint(*max_features_range)
        min_samples_leaf = random.randint(*min_samples_leaf_range)
        random_seed = random.randint(0, 1000)

        with mlflow.start_run(run_name=f"{parent_run.info.run_name}-attempt-{attempt}", nested=True) as child_run:
            print(f"Starting nested run {child_run.info.run_id}")

            # Load the training dataset
            X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=random_seed)

            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("max_features", max_features)
            mlflow.log_param("min_samples_leaf", min_samples_leaf)

            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                random_state=random_seed
            )

            # MLflow triggers logging automatically upon model fitting
            rf.fit(X_train, y_train)
            # Make predictions
            y_pred = rf.predict(X_test)

            # Calculate and log regression metrics
            current_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            current_mae = mean_absolute_error(y_test, y_pred)
            current_r2 = r2_score(y_test, y_pred)

            current_metrics = {
                "test_rmse": current_rmse,
                "test_mae": current_mae,
                "test_r2": current_r2
            }

            mlflow.log_metrics(current_metrics, run_id=child_run.info.run_id)
            mlflow.log_metrics(current_metrics, run_id=parent_run.info.run_id, step=attempt)

            if latest_run is None:
                print("✅ No previous run found, this will be our baseline")
                best_run = child_run
                improvement_found = True
                break

          # Compare metrics
            improvement = compare_metrics(
                client,
                child_run.info.run_id,
                latest_run.info.run_id,
                {
                    "test_r2": "higher",
                    "test_rmse": "lower",
                    "test_mae": "lower",
                }
            )

            # Print improvement status
            for metric, result in improvement.items():
                print(f"Improvement on {metric}: {'Yes' if result else 'No'}")

            if all(improvement.values()):
                print("✅ Found better performing model!")
                best_run = child_run
                improvement_found = True
                break


    # After all attempts (or early exit)
    if improvement_found:
        print(f"\n🎉 Found improved model after {attempt} attempts")
        model_uri = f"runs:/{best_run.info.run_id}/model"
        model_version = mlflow.register_model(model_uri, model_name)
        client.set_registered_model_alias(name=model_name, alias="dev", version=model_version.version)
        latest_run = best_run
        print(f"Model registered as '{model_name}' version {model_version.version}")
    else:
        print(f"\n🔴 No improvement found after {max_search_attempts} attempts")
        # Close the experiment as failure if no improvements found
        mlflow.end_run('FAILED')
        if latest_run:
            print("Keeping the previous best model")
        else:
            print("No model was registered (no baseline found)")

# %% [markdown]
# ## Step 5: Load and Use the Model

# %% ExecuteTime={"end_time": "2025-05-11T00:18:15.991967Z", "start_time": "2025-05-11T00:18:15.863213Z"}
# Get the model version (correct approach)

latest_version_info = client.get_model_version_by_alias(model_name, "dev")
print(f"Loaded model version: {latest_version_info.version}. Alias: {latest_version_info.aliases}")

latest_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest_version_info.version}")

# Convert diabetes data to a Pandas DataFrame
X_test = pd.DataFrame(dataset.data, columns=dataset.feature_names)

# Make predictions on the diabetes dataset
predictions = latest_model.predict(X_test)

# Add the predictions to a DataFrame
diabetes_result = pd.DataFrame(X_test, columns=dataset.feature_names)
# Since we don't have actual classes for the diabetes dataset, we can't add them
# diabetes_result["actual_class"] = y_test (commented out as not applicable)

# Add the model predictions to the DataFrame
diabetes_result["predicted_value"] = predictions

diabetes_result[:4]

# %% ExecuteTime={"end_time": "2025-05-11T00:18:16.118956Z", "start_time": "2025-05-11T00:18:16.022759Z"}

model = mlflow.pyfunc.load_model("models:/diabetes-model@dev")
print(model)

# %% [markdown]
# ### Invoke model server with CURL

# %% ExecuteTime={"end_time": "2025-05-11T00:18:16.410762Z", "start_time": "2025-05-11T00:18:16.156001Z"} language="bash"
# curl -X POST -s http://localhost:8080/v2/models/diabetes-model/infer \
#      -H "Content-Type: application/json" \
#      -d '{
#            "inputs": [
#              {
#                "name": "input-0",
#                "shape": [1, 10],
#                "datatype": "FP64",
#                "data": [0.038076, 0.050680, 0.061696, 0.021872, -0.044223, -0.034821, -0.043401, -0.002592, 0.019907, -0.017646]
#              }
#            ]
#          }' | jq
#

# %% ExecuteTime={"end_time": "2025-05-11T00:19:15.210137Z", "start_time": "2025-05-11T00:19:15.129042Z"} language="bash"
# curl -X GET -s http://localhost:8080/v2/models/diabetes-model | jq
#

# %% ExecuteTime={"end_time": "2025-05-11T00:18:16.489392Z", "start_time": "2025-05-11T00:18:16.432671Z"} language="bash"
#
#
# curl -s -X POST http://localhost:7000/invocations \
#  -H "Content-Type: application/json" \
#   -d '{"dataframe_split": {"columns": ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"], "data": [[0.038076, 0.050680, 0.061696, 0.021872, -0.044223, -0.034821, -0.043401, -0.002592, 0.019907, -0.017646]]}}' \
#   | jq
#

# %% ExecuteTime={"end_time": "2025-05-11T01:02:14.476104Z", "start_time": "2025-05-11T01:02:14.266723Z"} language="bash"
# curl -s -X POST http://localhost:8080/invocations \
#  -H "Content-Type: application/json" \
#   -d '{"dataframe_split": {"columns": ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"], "data": [[0.038076, 0.050680, 0.061696, 0.021872, -0.044223, -0.034821, -0.043401, -0.002592, 0.019907, -0.017646]]}}' \
#   | jq
#
#

# %% ExecuteTime={"end_time": "2025-05-12T08:46:30.702132Z", "start_time": "2025-05-12T08:46:29.854802Z"}
import requests

inference_request = {
    "dataframe_split": {
        "columns": ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"],
        "data": [[0.138076, 0.050680, 0.061696, 0.021872, -0.044223, -0.034821, -0.043401, -0.002592, 0.019907, -0.017646]]
    }
}

endpoint = "http://localhost:8080/invocations"
response = requests.post(endpoint, json=inference_request)

response.json()
