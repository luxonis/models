from luxonis_train.utils.config import Config
import mlflow
import argparse
from dotenv import load_dotenv

if __name__ == "__main__":
    """
        Stores config as MLFlow artifact and prints run_id
        Prerequisites: 
            - MLFlow parameters configured in configs under `logger`
            - .env file with AWS and MLFlow variables 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, required=True, help="Configuration file to use")
    parser.add_argument("--override", default=None, type=str, help="Manually override config parameter")
    args = parser.parse_args()
    args_dict = vars(args)

    load_dotenv()

    cfg = Config(args.config)
    if args.override:
        cfg.override_config(args.override)
    
    mlflow.set_tracking_uri(cfg.get("logger.mlflow_tracking_uri"))
    mlflow.set_experiment(cfg.get("logger.project_name"))
    with mlflow.start_run() as run:
        mlflow.log_dict(cfg.get_data(), "config.json")
        run_id = run.info.run_id
        print(f"Config saved as MLFlow artifact. Run id: {run_id}")