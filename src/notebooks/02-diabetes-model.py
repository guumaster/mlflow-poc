# ---
# jupyter:
#   execution:
#     iopub_status: {}
#   jupytext:
#     cell_metadata_filter: all,-pycharm
#     formats: ipynb,py:percent
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
# ## Setup

# %% [markdown]
# #### Set up environment
#

# %% [markdown]
# Import all libraries and extensions

# %%
import sys

sys.path.append("./extensions")

# %load_ext skip_kernel_extension

from typing import Dict
import os
import json

import random
import requests

import mlflow
import numpy as np
import pandas as pd

from mlflow import MlflowClient
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# %% [markdown]
# Set default values for env variables. These will be overwritten when running inside docker container.

# %% tags=["parameters"]
# Defaults to be changed when run inside docker
os.environ.setdefault("AWS_ACCESS_KEY_ID", "admin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "admin123")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("SKIP_INFERENCE", "false")

# Set variables
aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
mlflow_s3_endpoint_url = os.environ["MLFLOW_S3_ENDPOINT_URL"]
mlflow_tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
skip_inference = os.environ["SKIP_INFERENCE"].lower() == "true"

# For inference
os.environ.setdefault("MODEL_SERVER_URL", "http://localhost:8080")
model_server_url = os.environ["MODEL_SERVER_URL"]


# %% [markdown]
# MLflow configuration

# %%
experiment_name = "Diabetes Model"
model_name = "diabetes-model"

mlflow_client = MlflowClient(
    tracking_uri=os.getenv("MLFLOW_TRACKING_URI"), registry_uri=os.getenv("MLFLOW_TRACKING_URI")
)

mlflow.set_tracking_uri(uri=os.getenv("MLFLOW_TRACKING_URI"))
experiment = mlflow.set_experiment(experiment_name)


# %% [markdown]
# ##### Add some helper functions


# %%
def compare_metrics(
    client: MlflowClient, current_run_id: str, baseline_run_id: str, metrics_to_compare: Dict[str, str]
):
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
        "Transonic",
        "Hypersonic",
        "Afterburning",
        "Turbocharged",
        "Supersonic",
        "Machbreaking",
        "Scramjet",
        "Thrustvectored",
        "Stratospheric",
        "Tropospheric",
        "Cloudpiercing",
        "Jetstreamed",
        "Contrailswept",
        "Headwinded",
        "Tailwinded",
        "Crosswinded",
        "Flybywire",
        "Autothrottle",
        "Glasscockpit",
        "Headup",
        "Skybound",
        "Runwaylit",
        "Aileronrolled",
        "Flapsdown",
        "Chocksaway",
        "Clearedfortakeoff",
        "Finalapproach",
        "Goaround",
        "Quantum",
        "Neural",
        "Plasma",
        "Gravitic",
        "Singularity",
        "Nanotech",
        "Exo",
        "Hyperspace",
        "Photon",
        "Cloaking",
        "Tachyon",
        "Warp",
        "Zero-G",
        "Cybernetic",
        "Holographic",
        "Ion",
        "Antimatter",
        "Bioengineered",
        "Psi",
        "Chronojump",
    ]

    nouns = [
        "Turbofan",
        "Tailfin",
        "Flaps",
        "Ailerons",
        "Elevator",
        "Rudder",
        "Spoilers",
        "Slats",
        "Throttle",
        "Yawdamper",
        "Stick",
        "Pedals",
        "Tarmac",
        "Hangar",
        "Airstrip",
        "Runway",
        "Taxiway",
        "Apron",
        "Jetbridge",
        "Windsock",
        "Glideslope",
        "Localizer",
        "Gliderail",
        "Flightdeck",
        "Blackbox",
        "Transponder",
        "Squawkbox",
        "Takeoff",
        "Landing",
        "Approach",
        "Holdingpattern",
        "Jumpgate",
        "Thruster",
        "Pulsejet",
        "Shield",
        "Wormhole",
        "Drone",
        "Neurohelm",
        "Gravcoil",
        "Phasewings",
        "Starfighter",
        "Titanium",
        "Voidship",
        "Lasercannon",
        "AI",
        "Stasis",
        "Dyson",
        "Warpcore",
        "Omnitool",
        "Singularity",
        "Hoverpad",
    ]

    random_adjective = random.choice(adjectives).lower()
    random_noun = random.choice(nouns).lower()

    # Generate a 3-digit suffix
    suffix = str(random.randint(100, 999))

    return f"{random_adjective}-{random_noun}-{suffix}"


# %% [markdown]
# #### Load datatest and previous metrics

# %%
dataset = datasets.load_diabetes()

# %%
latest_run = None
latest_metrics = {"test_rmse": None, "test_mae": None, "test_r2": None}
res = []

if experiment:
    res = mlflow_client.search_runs(experiment.experiment_id, order_by=["attributes.start_time DESC"], max_results=1)
    if len(res) > 0:
        latest_run = res[0]
        latest_metrics = latest_run.data.metrics


# %% [markdown]
# ## Model Training with MLflow

# %%
# Define the hyperparameter ranges
n_estimators_range = (10, 600)
max_depth_range = (5, 30)
max_features_range = (2, 20)
min_samples_leaf_range = (1, 100)

max_search_attempts = 5
import scrapbook as sb

# Create a parent run for all the attempts
with mlflow.start_run(run_name=generate_random_run_name()) as parent_run:

    # Enable MLflow's automatic experiment tracking for scikit-learn
    mlflow.sklearn.autolog(log_models=False)  # Model is logged separately below

    print(f"Starting hyperparameter search under parent run {parent_run.info.run_id}")
    # Record the run ID using scrapbook
    sb.glue("mlflow_run_id", parent_run.info.run_id)

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
                random_state=random_seed,
            )

            # MLflow triggers logging automatically upon model fitting
            rf.fit(X_train, y_train)
            # Make predictions
            y_pred = rf.predict(X_test)

            # Calculate and log regression metrics
            current_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            current_mae = mean_absolute_error(y_test, y_pred)
            current_r2 = r2_score(y_test, y_pred)

            current_metrics = {"test_rmse": current_rmse, "test_mae": current_mae, "test_r2": current_r2}

            mlflow.log_metrics(current_metrics, run_id=child_run.info.run_id)
            mlflow.log_metrics(current_metrics, run_id=parent_run.info.run_id, step=attempt)

            # Log the model with dependencies and metadata
            mlflow.sklearn.log_model(
                sk_model=rf,
                artifact_path="model",
                # registered_model_name=model_name,
                extra_pip_requirements=["boto3==1.38.16"],
                input_example=X_train[:5],  # Example input for schema inference
                signature=mlflow.models.infer_signature(X_train, y_pred),  # Model signature
            )

            if latest_run is None:
                print("✅ No previous run found, this will be our baseline")
                best_run = child_run
                improvement_found = True
                break

            # Compare metrics
            improvement = compare_metrics(
                mlflow_client,
                child_run.info.run_id,
                latest_run.info.run_id,
                {"test_r2": "higher", "test_rmse": "lower", "test_mae": "lower"},
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
        mlflow_client.set_registered_model_alias(name=model_name, alias="dev", version=model_version.version)
        latest_run = best_run
        print(f"Model registered as '{model_name}' version {model_version.version}")
    else:
        print(f"\n🔴 No improvement found after {max_search_attempts} attempts")
        # Close the experiment as failure if no improvements found
        mlflow.end_run("FAILED")
        if latest_run:
            print("Keeping the previous best model")
        else:
            print("No model was registered (no baseline found)")

# %% [markdown]
# ## Predictions

# %% [markdown]
# ##### Skip inference cells
#
# This checkbox can be marked to skip all subsequent cells during Dagster run.

# %%
from ipywidgets import Checkbox

skip_inference = Checkbox(
    value=os.environ["SKIP_INFERENCE"].lower() == "true", description="Skip inference", disabled=False, indent=False
)

skip_inference

# %% [markdown]
# #### Load the model

# %% [markdown]
# ##### Load the model with MLflowClient

# %%
# %%skip $skip_inference.value

latest_version_info = mlflow_client.get_model_version_by_alias(model_name, "dev")
print(f"Latest model version: {latest_version_info.version}. Alias: {latest_version_info.aliases}")

# model_uri ="models:/diabetes-model@dev"
model_uri = f"models:/{model_name}/{latest_version_info.version}"

latest_model = mlflow.pyfunc.load_model(model_uri)

latest_model

# %% [markdown]
# #### Make predictions for all dataset

# %%
# %%skip $skip_inference.value

# Convert diabetes data to a Pandas DataFrame
X_test = pd.DataFrame(dataset.data, columns=dataset.feature_names)

# Make predictions on the diabetes dataset
predictions = latest_model.predict(X_test)

# Add the predictions to a DataFrame
diabetes_result = pd.DataFrame(X_test, columns=dataset.feature_names)
# Since we don't have actual classes for the diabetes dataset, we can't add them
# diabetes_result["actual_class"] = y_test (commented out as not applicable)
diabetes_result_with_predictions = diabetes_result.copy()

# Add the model predictions to the DataFrame
diabetes_result_with_predictions["predicted_value"] = predictions

print("Diabetes result shape:", diabetes_result_with_predictions.shape)

diabetes_result_with_predictions.head()

# %% [markdown]
# #### Use model server

# %% [markdown]
# Ensure mlserver container is running in docker

# %% language="bash"
# [ "$SKIP_INFERENCE" = true ] && echo "SKIP: $SKIP_INFERENCE" && exit 0
#
# docker-compose up -d mlserver
#

# %% [markdown]
# ##### Make predictions through model server

# %% [markdown]
# Make a single prediction

# %%
# %%skip $skip_inference.value

# select random row
row = diabetes_result.sample().iloc[0].to_list()  # Select a random row from the dataset

response = requests.post(
    f"{model_server_url}/invocations",
    json={"dataframe_split": {"columns": diabetes_result.columns.to_list(), "data": [row]}},
)

print(json.dumps(response.json(), indent=4))


# %%
# %%skip $skip_inference.value

# select first/last rows
first_last = pd.concat([diabetes_result.iloc[[0]], diabetes_result.iloc[[-1]]])

response = requests.post(
    f"{model_server_url}/invocations",
    json={"dataframe_split": {"columns": diabetes_result.columns.to_list(), "data": first_last.values.tolist()}},
)

print(json.dumps(response.json(), indent=4))


# %% [markdown]
# Make prediction for all rows in a dataframe

# %%
# %%skip $skip_inference.value

response = requests.post(
    f"{model_server_url}/invocations",
    json={"dataframe_split": {"columns": diabetes_result.columns.to_list(), "data": diabetes_result.values.tolist()}},
)
response_data = response.json()
# print(json.dumps(response_data, indent=4))

diabetes_result_with_predictions = diabetes_result.copy()
diabetes_result_with_predictions["predictions_response"] = response_data["predictions"]

diabetes_result_with_predictions

# %% [markdown]
# Check model server status

# %%
# %%skip $skip_inference.value

response = requests.post(f"{model_server_url}/v2/repository/index", json={})
pretty_json = json.dumps(response.json(), indent=4)
print(pretty_json)

# %% [markdown]
# Force a model reload with latest version

# %% [markdown]
# Force a model reload on model server

# %%
# %%skip $skip_inference.value

response = requests.post(
    f"{model_server_url}/v2/repository/models/diabetes-model/load",
    headers={"Content-Type": "application/json"},
    timeout=10,
)
response.raise_for_status()
