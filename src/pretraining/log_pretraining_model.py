from src.common import read_params
import argparse
import mlflow
import pandas as pd
# import shutil
import boto3


def log_production_model(config_path):
    config = read_params(config_path)

    server_uri = config["mlflow_pretraining"]["server_uri"]
    experiment_name = config["mlflow_pretraining"]["experiment_name"]
    s3_mlruns_bucket = config["mlflow_pretraining"]["s3_mlruns_bucket"]

    mlflow.set_tracking_uri(server_uri)

    # get experiment id
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # get best model
    df = pd.DataFrame(mlflow.search_runs(experiment_ids=experiment_id))


    run_id = None #"77b98144caa84e34b16a413045736d97"

    if run_id is not None:
        run = df[df['run_id'] == run_id]
        
    else:
        df = df[df["status"] == "FINISHED"]
        run = df[df["metrics.Val Loss"] == df["metrics.Val Loss"].min()]
        
    model_src = run['artifact_uri'].values[0].split(s3_mlruns_bucket)[1] + "/GPT/data/model.pth"
    model_src = model_src[1:]
    
    model_dest = config["log_pretrained_model"]["model_dir"]

    # Download file
    s3 = boto3.client('s3')
    s3.download_file(s3_mlruns_bucket, model_src, model_dest)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    log_production_model(config_path=parsed_args.config)
