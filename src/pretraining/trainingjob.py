from src.common import read_params
import argparse
import sagemaker
from sagemaker.pytorch import PyTorch
from datetime import datetime


def setup_training(config_path):
    config = read_params(config_path)

    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    
    # Environment variables for training script inside the container pulled from params.yaml
    environment = {
        "VOCAB_SIZE": str(config["hyperparameters"]["vocab_size"]),
        "N_EMBD": str(config["hyperparameters"]["n_embd"]),
        "BLOCK_SIZE": str(config["hyperparameters"]["block_size"]),
        "N_HEAD": str(config["hyperparameters"]["n_head"]),
        "N_LAYER": str(config["hyperparameters"]["n_layer"]),
        "DROPOUT": str(config["hyperparameters"]["dropout"]),
        "EPOCHS": str(config["hyperparameters"]["epochs"]),
        "MAX_ITERS": str(config["hyperparameters"]["max_iters"]),
        "BATCH_SIZE": str(config["hyperparameters"]["batch_size"]),
        "MINI_BATCH_SIZE": str(config["hyperparameters"]["mini_batch_size"]),
        "LEARNING_RATE": str(config["hyperparameters"]["learning_rate"]),
        "EVAL_ITERS": str(config["hyperparameters"]["eval_iters"]),
        "EVAL_INTERVAL": str(config["hyperparameters"]["eval_interval"]),

        "EXPERIMENT_NAME": config['mlflow_pretraining']['experiment_name'],
        "RUN_NAME": config['mlflow_pretraining']['run_name'],
        "REGISTERED_MODEL_NAME": config['mlflow_pretraining']['registered_model_name'],
        "SERVER_URI": config['mlflow_pretraining']['server_uri'],
        "S3_MLRUNS_BUCKET": config['mlflow_pretraining']['s3_mlruns_bucket'],

        "MLFLOW_TRACKING_USERNAME": config['mlflow_pretraining']['tracking_username'],
        "MLFLOW_TRACKING_PASSWORD": config['mlflow_pretraining']['tracking_password'],
    }

    # Configure PyTorch estimator
    estimator = PyTorch(
        entry_point=config['pytorch_estimator']['entry_point'],
        source_dir=config['pytorch_estimator']['source_dir'],
        role=config['pytorch_estimator']['role'],
        framework_version=config['pytorch_estimator']['framework_version'],
        py_version=config['pytorch_estimator']['py_version'],
        instance_count=config['pytorch_estimator']['instance_count'],
        instance_type=config['pytorch_estimator']['instance_type'],
        use_spot_instances=config['pytorch_estimator']['use_spot_instances'],
        max_wait=config['pytorch_estimator']['max_wait'],
        max_run=config['pytorch_estimator']['max_run'],
        environment=environment,
        distribution={
            "deepspeed": {
                "enabled": True
            }
        }
    )

    # Define data channels
    data = {
        'train': config['pytorch_estimator']['s3_train_data'],
    }
    
    # Start training with less verbose logs
    estimator.fit(inputs=data)
    
    # update the training completion time so that DVC can use that file to check whether the training step is complete
    with open("training_completion.txt", "w") as file:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        file.write("Training Completed at " + formatted_datetime)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    setup_training(config_path=parsed_args.config)