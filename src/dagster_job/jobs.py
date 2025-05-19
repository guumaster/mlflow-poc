import os
from pathlib import Path

import papermill as pm
import scrapbook as sb
from mlflow import MlflowClient
from mlflow.models import build_docker

from dagster import (
    DefaultScheduleStatus,
    MetadataValue,
    Out,
    Output,
    job,
    op,
    repository,
    schedule,
)


@op(out={"mlflow_parent_run_id": Out(str), "mlflow_model_run_id": Out(str)})
def train_diabetes_model(context):
    # Define notebook paths
    input_notebook = "/usr/src/app/notebooks/02-diabetes-model.ipynb"
    output_dir = "/tmp"  # "/usr/src/app/notebooks/outputs"

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_notebook = f"{output_dir}/02-simple-workflow_{context.run_id}.ipynb"

    try:
        # Execute the notebook with papermill
        pm.execute_notebook(
            input_notebook,
            output_notebook,
            parameters={
                "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
                "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
                "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                "mlflow_s3_endpoint_url": os.getenv("MLFLOW_S3_ENDPOINT_URL"),
                "skip_inference": True,
            },
            log_output=True,
            progress_bar=False,
            report_mode=True,
            strip_output=True,  # This removes output cells entirely
        )

        scrapbook_data = sb.read_notebook(output_notebook)
        mlflow_parent_run_id = scrapbook_data.scraps.get("mlflow_parent_run_id").data
        mlflow_model_run_id = scrapbook_data.scraps.get("mlflow_model_run_id").data

        if not mlflow_parent_run_id or not mlflow_model_run_id:
            raise ValueError("MLflow run IDs not found in notebook execution")

        context.log.info(
            f"Notebook executed successfully. "
            f"MLflow parent run ID: {mlflow_parent_run_id}, "
            f"MLflow model run ID: {mlflow_model_run_id}"
        )

        yield Output(mlflow_parent_run_id, "mlflow_parent_run_id")
        yield Output(mlflow_model_run_id, "mlflow_model_run_id")

    except pm.PapermillExecutionError as e:
        context.log.error(f"Notebook execution failed: {str(e)}")
        raise


@op(out={"verified_mlflow_run_id": Out(str)})
def check_mlflow_status(context, mlflow_model_run_id: str):
    try:
        client = MlflowClient(os.getenv("MLFLOW_TRACKING_URI"))
        run = client.get_run(mlflow_model_run_id)

        if run.info.status == "FAILED":
            raise ValueError(f"MLflow run {mlflow_model_run_id} failed with status: {run.info.status}")

        context.log.info(f"MLflow run {mlflow_model_run_id} completed successfully with status: {run.info.status}")

        yield Output(mlflow_model_run_id, "verified_mlflow_run_id")

    except Exception as e:
        context.log.error(f"Error checking MLflow status: {str(e)}")
        raise


@op
def build_model_docker_image(context, verified_mlflow_run_id: str):
    """
    Builds Docker image for the newly registered MLflow model
    """
    try:
        mlflow_client = MlflowClient()

        # Find the registered model version from the run
        model_name = "diabetes-model"
        model_versions = mlflow_client.search_model_versions(f"run_id='{verified_mlflow_run_id}'")

        if not model_versions:
            raise ValueError(f"No model versions found for run {verified_mlflow_run_id}")

        model_version = model_versions[0]

        context.log.info(f"Found model version {model_version.version} for run {verified_mlflow_run_id}")

        # Build Docker image for this specific version
        image_name = f"mlflow-diabetes-model:{model_version.version}"
        context.log.info(f"Building Docker image '{image_name}'...")

        # Programmatic Docker build
        build_docker(
            model_uri=f"models:/{model_name}/{model_version.version}",
            name=image_name,
            # env_manager="conda",  # or "virtualenv"
            # env_manager="virtualenv",  # or "virtualenv"
            env_manager="",
            install_mlflow=True,
            enable_mlserver=True,
            install_java=False,
            base_image="python:3.12-slim",
        )

        context.log.info(f"Successfully built Docker image: {image_name}")

        url = "{}/experiments/1/runs/{}".format(os.getenv("MLFLOW_TRACKING_URI"), verified_mlflow_run_id)
        context.add_output_metadata(
            {
                "mlflow_run": MetadataValue.url(url),
                "mlflow_run_id": verified_mlflow_run_id,
                "model_version": model_version.version,
                "image_name": image_name,
            }
        )

        return image_name

    except Exception as e:
        context.log.error(f"Failed to build Docker image: {str(e)}")
        raise


@job
def train_diabetes_model_job():
    _, mlflow_model_run_id = train_diabetes_model()
    verified_mlflow_model_run_id = check_mlflow_status(mlflow_model_run_id)  # This must complete first
    build_model_docker_image(verified_mlflow_model_run_id)


@schedule(
    cron_schedule="0 0 * * *",
    job=train_diabetes_model_job,
    execution_timezone="UTC",
    default_status=DefaultScheduleStatus.RUNNING,
)  # Daily at midnight UTC
def daily_training_schedule(_context):
    return {}


@repository()
def mlflow_repo():
    return [train_diabetes_model_job, daily_training_schedule]
