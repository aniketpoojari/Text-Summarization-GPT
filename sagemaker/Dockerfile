# Use the official NVIDIA CUDA base image with Python 3.8 and minimal dependencies
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Set the environment variable to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools, Python 3.8, pip, curl, git, and other necessary libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3-pip curl git build-essential \
    ca-certificates make gcc libssl-dev && \
    rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Install Rust using rustup (official installation method)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add Rust binaries to PATH permanently for the build process
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Python dependencies in a single step to reduce the number of layers
RUN pip3 install --no-cache-dir \
    transformers \
    torch \
    sagemaker

# Clean up pip cache to reduce image size
RUN rm -rf /root/.cache/pip

# Set environment variables for working with SageMaker and other ML tools
ENV PATH="/opt/ml/code:/usr/local/bin:$PATH" \
    PYTHONUNBUFFERED=TRUE \
    PYTHONPATH="/opt/ml/code"

# Set the working directory inside the container to /opt/ml/code (SageMaker default)
WORKDIR /opt/ml/code

# Copy your application code (train.py, model.py, dataloader.py) into the container
COPY train.py model.py dataloader.py ./

# Ensure the Python executable is available
RUN python3 --version
RUN which python3

# Define the container's entry point to run the training script (train.py)
ENTRYPOINT ["python3", "train.py"]

# Mount directories for input data and model output (SageMaker expected paths)
VOLUME ["/opt/ml/input", "/opt/ml/model"]

# Optional: Expose a port (useful for inference endpoints or monitoring)
# EXPOSE 8080

# Clean up unnecessary files and reduce image size
RUN apt-get clean
