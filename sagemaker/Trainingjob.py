from sagemaker.inputs import TrainingInput
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator

role = "arn:aws:iam::975050169307:role/training-jobs-role"

estimator = Estimator(
    image_uri="975050169307.dkr.ecr.us-west-2.amazonaws.com/gpt",
    role=role,
    instance_type="ml.g4dn.xlarge",  # Choose your instance type
    instance_count=1,
    base_job_name="custom-ecs-training",
    use_spot_instances=True,  # Enable spot instances
    max_wait=3600,  # Maximum wait time (in seconds)
    max_run=3600,   # Maximum run time (in seconds)
    input_mode="File",
    output_dir="/opt/ml/model/",
    output_path="s3://text-summarization-gpt-aniket/",
    environment={  # Override environment variables here
        "vocab_size": "50257",
        "n_embd": "768",
        "block_size": "128",
        "n_head": "12",
        "n_layer": "12",
        "dropout": "0.1",
        "epochs": "1",
        "max_iters": "2500",
        "batch_size": "32",
        "mini_batch_size": "32",
        "learning_rate": "2e-4",
        "eval_iters": "50",
        "eval_interval": "100",
    },
)

# Define the input data location (could be S3 or other)
s3_input_train = TrainingInput(
    s3_data="s3://text-summarization-gpt-aniket/pretraining_train.pkl"
)

s3_input_validation = TrainingInput(
    s3_data="s3://text-summarization-gpt-aniket/pretraining_val.pkl"
)

# Start the training job
estimator.fit({"train": s3_input_train, "validation": s3_input_validation})