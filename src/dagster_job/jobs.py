import os
from pathlib import Path

import papermill as pm
from mlflow import MlflowClient

from dagster import DefaultScheduleStatus, Out, Output, job, op, repository, schedule


@op(out={"mlflow_run_id": Out(str)})
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

        # Read the scrapbook data
        import scrapbook as sb

        scrapbook_data = sb.read_notebook(output_notebook)
        mlflow_run_id = scrapbook_data.scraps.get("mlflow_run_id").data

        if not mlflow_run_id:
            raise ValueError("MLflow run ID not found in notebook execution")

        context.log.info(f"Notebook executed successfully. MLflow run ID: {mlflow_run_id}")
        return Output(mlflow_run_id, "mlflow_run_id")

    except pm.PapermillExecutionError as e:
        context.log.error(f"Notebook execution failed: {str(e)}")
        raise


@op
def check_mlflow_status(context, mlflow_run_id: str):
    try:
        client = MlflowClient(os.getenv("MLFLOW_TRACKING_URI"))
        run = client.get_run(mlflow_run_id)

        if run.info.status == "FAILED":
            raise ValueError(f"MLflow run {mlflow_run_id} failed with status: {run.info.status}")

        context.log.info(f"MLflow run {mlflow_run_id} completed successfully with status: {run.info.status}")

    except Exception as e:
        context.log.error(f"Error checking MLflow status: {str(e)}")
        raise


@job
def train_diabetes_model_job():
    mlflow_run_id = train_diabetes_model()
    check_mlflow_status(mlflow_run_id)


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
