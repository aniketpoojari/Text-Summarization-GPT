# Use a smaller base Python image (slim version)
FROM python:3.10-slim

# Set the environment variable to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY saved_models/requirements.txt /app/requirements.txt

# Install system dependencies, clean up to reduce image size, and install Python dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    rm -rf /root/.cache/pip

# Set the working directory

# Copy necessary files
COPY saved_models/summary.pth /app/summary.pth
COPY saved_models/website.py /app/website.py
COPY saved_models/model.py /app/model.py


# Set the default command to run the website
CMD ["streamlit", "run", "website.py"]




