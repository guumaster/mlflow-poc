import os
from pathlib import Path

import papermill as pm

from dagster import DefaultScheduleStatus, job, op, repository, schedule


@op
def train_diabetes_model(context):
    # Define notebook paths
    input_notebook = "/usr/src/app/notebooks/02-diabetes-model.ipynb"
    output_dir = "/usr/src/app/notebooks/outputs"

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_notebook = f"{output_dir}/02-simple-workflow_{context.run_id}.ipynb"

    try:
        # Execute the notebook with papermill
        pm.execute_notebook(
            input_notebook,
            None,  # output_notebook,
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
        )

        context.log.info(f"Notebook executed successfully. Output saved to {output_notebook}")

    except pm.PapermillExecutionError as e:
        context.log.error(f"Notebook execution failed: {str(e)}")
        raise
    except Exception as e:
        context.log.error(f"Unexpected error: {str(e)}")
        raise


@job
def train_diabetes_model_job():
    train_diabetes_model()


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
