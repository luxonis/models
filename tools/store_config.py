from luxonis_train.utils.config import Config
import mlflow
import argparse
from dotenv import load_dotenv
import boto3
import os
import json

if __name__ == "__main__":
    """
    Stores config as MLFlow artifact and prints run_id
    Prerequisites:
        - MLFlow parameters configured in configs under `logger`
        - .env file with AWS and MLFlow variables
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="Configuration file to use"
    )
    parser.add_argument(
        "--override", default=None, type=str, help="Manually override config parameter"
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="luxonis-mlflow",
        help="S3 bucket name for config upload",
    )
    args = parser.parse_args()
    args_dict = vars(args)

    load_dotenv()

    cfg = Config(args.config)
    if args.override:
        cfg.override_config(args.override)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    if cfg.get("logger.project_id") is not None:
        cfg.override_config("logger.project_name null")
    project_id = cfg.get("logger.project_id")
    mlflow.set_experiment(
        experiment_name=cfg.get("logger.project_name"),
        experiment_id=str(project_id) if project_id is not None else None,
    )

    with mlflow.start_run() as run:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            endpoint_url=os.getenv("AWS_S3_ENDPOINT_URL"),
        )
        tmp_path = "config.json"
        with open(tmp_path, "w+") as f:
            json.dump(cfg.get_data(), f, indent=4)

        key = run.info.artifact_uri.split(args.bucket + "/")[-1] + "/config.json"
        s3_client.upload_file(Filename="config.json", Bucket=args.bucket, Key=key)
        os.remove(tmp_path)  # delete temporary file

        run_id = run.info.run_id
        print(f"Config saved as MLFlow artifact. Run id: {run_id}")
