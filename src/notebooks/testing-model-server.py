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
#     version: 3.12.9
# ---

# %% [markdown]
# ## Calls to Dagster

# %%
import time

from dagster_graphql import DagsterGraphQLClient, DagsterGraphQLClientError

from dagster import DagsterRunStatus


# Function to wait for completion
def wait_for_job_completion(run_id, timeout=3600, poll_interval=5):
    start_time = time.time()

    while True:
        # Check if timeout reached
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Job {run_id} did not complete within {timeout} seconds")

        try:
            status = client.get_run_status(run_id)
            if status == DagsterRunStatus.SUCCESS:
                return True, status
            if status in [DagsterRunStatus.FAILURE, DagsterRunStatus.CANCELED]:
                return False, status
            print(f"Job status: {status}. Waiting...")
            time.sleep(poll_interval)

        except DagsterGraphQLClientError as exc:
            print(f"Error checking job status: {exc}")
            raise


def wait_until_model_ready(model_server_url, model_name, max_retries=10, timeout=5, initial_delay=5) -> None:
    """Wait until the model is ready."""

    # Initial delay to allow model server to start
    print(f"Waiting for {initial_delay} seconds for model server to start...")
    time.sleep(initial_delay)

    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.get(f"{model_server_url}/v2/models/{model_name}/ready")
            response.raise_for_status()  # Raise an exception for bad status codes
            print("Ready")
            break
        except requests.exceptions.RequestException as e:
            if retry_count == max_retries - 1:
                print(f"Model is not ready after {max_retries} retries: {e}")
            else:
                print(f"Not ready, retrying in {timeout} seconds: {e}")
        time.sleep(timeout)
        retry_count += 1


# %%
client = DagsterGraphQLClient("localhost", port_number=3000)

# Wait for the job to complete
try:
    new_run_id: str = client.submit_job_execution(
        "train_diabetes_model_job",  # Your job name
        repository_location_name="src.dagster_job.repository",  # Location from your error
        run_config={},
    )

    print(f"Job submitted with run_id: {new_run_id}")

    success, status = wait_for_job_completion(new_run_id)
    if success:
        print("Job completed successfully!")
    else:
        print(f"Job failed with status: {status}")

except Exception as e:
    print(f"Error while waiting for job completion: {e}")
    raise

# %% language="bash"
#
# export MODEL_VERSION=$(curl -s -X GET "http://localhost:5000/api/2.0/mlflow/registered-models/alias?name=diabetes-model&alias=dev" \
#     | jq -r '.model_version.version')
#
# echo "Model version: $MODEL_VERSION"
#
# echo "Restarting docker with latest version"
#
# docker-compose --profile model-server up -d  --force-recreate mlflow-diabetes-model
#

# %% [markdown]
# ## Testing python calls

# %%
import pandas as pd
import requests
from sklearn import datasets

# Manual "manual-diabetes-model" build with "dev" model.
model_server_url = "http://localhost:7001"
model_name = "mlflow-model"

# Build by MLflow "mlflow-diabetes-model"
model_server_url = "http://localhost:7000"
model_name = "diabetes-model"

# %%
dataset = datasets.load_diabetes()

diabetes_result = pd.DataFrame(dataset["data"], columns=dataset["feature_names"])

# %% [markdown]
# Make prediction for all rows in a dataframe

# %%
wait_until_model_ready(model_server_url, model_name)

# %%
response = requests.post(
    f"{model_server_url}/invocations",
    json={
        "dataframe_split": {"columns": diabetes_result.columns.to_list(), "data": diabetes_result.to_numpy().tolist()},
    },
    timeout=5,
)
response_data = response.json()
# print(json.dumps(response_data, indent=4))

diabetes_result_with_predictions = diabetes_result.copy()
diabetes_result_with_predictions["predictions_response"] = response_data["predictions"]

diabetes_result_with_predictions

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Testing API calls to model server

# %% [markdown]
# #### Setup for Dagster built MLflow container

# %%
# Config for diabetes-model-server
# %env MODEL_SERVER_URL=http://localhost:7000
# %env MODEL_NAME=diabetes-model

# %% [markdown]
# ### Setup for Seldom mlserver

# %%
# Config for mlserver (Seldon.IO container)
# %env MODEL_SERVER_URL=http://localhost:8080
# %env MODEL_NAME=diabetes-model

# %% language="bash"
#
# docker-compose up -d mlserver
#

# %% [markdown]
# ### Calls to server

# %% language="bash"
#
# curl -f -s  "${MODEL_SERVER_URL}/v2/models/diabetes-model/ready" && echo "Ready" || echo "Not ready"
#

# %% language="bash"
# curl -s -X POST ${MODEL_SERVER_URL}/invocations \
#  -H "Content-Type: application/json" \
#   -d '{
#         "dataframe_split": {
#             "columns": ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"],
#             "data": [
#                 [0.038076, 0.050680, 0.061696, 0.021872, -0.044223, -0.034821, -0.043401, -0.002592, 0.019907, -0.017646],
#                 [-0.001882, 0.352598, -0.170647, 0.190409, 0.273711, -0.024980, -0.091299, 0.042257, -0.049390, -0.092804]
#             ]
#         }
#       }' \
#   | jq
#
#

# %% language="bash"
#
# curl -X POST -s "${MODEL_SERVER_URL}/v2/models/${MODEL_NAME}/infer" \
#      -H "Content-Type: application/json" \
#      -d '{
#            "inputs": [
#              {
#                "name": "input-0",
#                "shape": [2, 10],
#                "datatype": "FP64",
#                "data": [
#                     [0.038076, 0.050680, 0.061696, 0.021872, -0.044223, -0.034821, -0.043401, -0.002592, 0.019907, -0.017646],
#                     [0.041708, 0.059182, 0.063738, 0.022681, -0.042640, -0.034450, -0.042857, -0.002639, 0.020058, -0.017646]
#                ]
#              }
#            ]
#          }' | jq
#

# %% language="bash"
# curl -X GET -s "${MODEL_SERVER_URL}/v2/models/${MODEL_NAME}" | jq
#

# %% [markdown]
# ##### Force model unload/load with curl

# %% language="bash"
#
# curl -s  -X POST "${MODEL_SERVER_URL}/v2/repository/models/${MODEL_NAME}/unload"
#

# %% language="bash"
#
# curl -s -X POST "${MODEL_SERVER_URL}/v2/repository/models/${MODEL_NAME}/load" | jq
#

# %% [markdown]
# Check model status with repository index call

# %% language="bash"
#
# curl -s -X POST "${MODEL_SERVER_URL}/v2/repository/index" \
#   -H "Content-Type: application/json" \
#   -d "{}" | jq

# %% [markdown]
# #### Using Custom docker container model-server

# %% language="bash"
#
# curl -s -X POST "${MODEL_SERVER_URL}/invocations" \
#  -H "Content-Type: application/json" \
#   -d '{
#         "dataframe_split": {
#             "columns": ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"],
#             "data": [
#                 [0.038076, 0.050680, 0.061696, 0.021872, -0.044223, -0.034821, -0.043401, -0.002592, 0.019907, -0.017646],
#                 [-0.001882, -0.044642, -0.118358, -0.034689, -0.045946, -0.034157, -0.072402, 0.042324, -0.009637, 0.034509]
#             ]
#         }
#       }' \
#   | jq
#

# %% language="bash"
#
# curl -s -X POST "${MODEL_SERVER_URL}/v2/repository/index" \
#   -H "Content-Type: application/json" \
#   -d "{}" | jq
#
