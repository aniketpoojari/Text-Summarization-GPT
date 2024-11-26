# 🚀 Project Name

Text Summarization GPT

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Requirements](#requirements)
- [Data](#data)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Docker](#docker)

## 📄 Introduction

Text summarization is the task of generating a concise and coherent summary from a longer text while preserving the essential information. In this project, I developed a text summarization model using a GPT-based architecture, trained on the XSum dataset for both pretraining and summary generation. The XSum dataset contains news articles paired with single-sentence summaries, which provides an ideal foundation for training an abstractive summarization model.

The model has a context length of 128 tokens, allowing it to process a substantial portion of text while generating concise summaries. It consists of 125 million parameters, enabling the model to capture a rich understanding of the input text and generate high-quality, fluent summaries.

The model was trained using the teacher forcing method for the summary generation task, where the correct previous token is provided as input during training to help the model learn more effectively. This approach ensures that the model can generate high-quality summaries by learning to predict the next word in the sequence while receiving the ground truth token during training.

To make the model accessible and user-friendly, I deployed it through a Streamlit app, which allows users to input text and receive generated summaries. The entire app was containerized using a Dockerfile to ensure a consistent and reproducible environment across different systems.

For efficient management of data and reproducibility, I used DVC (Data Version Control) to create a pipeline that handles data and model versions, ensuring a smooth and organized workflow. Additionally, I tracked the entire training process, hyperparameters, and model performance using MLflow, which provides valuable insights into the training process and experiment results.

The goal of this project is to create a model that generates high-quality, concise, and informative summaries across a variety of text domains while providing an easy-to-use interface for users through the Streamlit app.

## 🌟 Features

- [Data Loader](src/data_loader.py)
- [Model](src/model.py)
- [Clean Pretraining Data](src/clean_pretraining_data.py)
- [Split Pretraining Data](src/create_pretraining_split.py)
- [Pretraining](src/pretraining.py)
- [Best Pretrained Model Selection](src/log_pretraining_model.py)
- [Create Summary Data](src/create_summary_data.py)
- [Split Summary Data](src/create_summary_split.py)
- [Summary Finetuning](src/summary_generation.py)
- [Best Summary Model Selection](src/log_summary_model.py)
- [Streamlit App](saved_models/website.py)

## 🚚 Installation

```bash
# Clone the repository
git clone https://github.com/aniketpoojari/Monocular-Depth-Estimation.git

# Change directory
cd Monocular-Depth-Estimation

# Install dependencies
pip install -r requirements.txt
```

## 🛠️ Requirements

- argparse
- mlflow
- torch
- transformers
- pandas
- scikit-learn
- tqdm
- rouge-score
- PyYAML
- numpy
- shutil

## 📊 Data

- Downloaded from the Xsum Dataset - https://huggingface.co/datasets/sentence-transformers/xsum
- save the train.src.txt and train.tgt.txt files in the data/raw/summary folder
- track both the files using DVC - dvc add path/to/directory/

## 🤖 Model Training

```bash
# Run mlflow server in the background before running the pipeline
mlflow server --backend-store-uri sqlite:///mlflow.db --host localhost

# Change values in the params.yaml file for testing different parameters
# Train the model using this command
dvc repro
```

## 📈 Evaluation

- RMSE score is used to evaluate the pretrained model
- ROUGE-L score is used to evaluate the summary model

## 🎉 Results

- Got a ROUGE-L score of 0.54 on the validation set for the summary model
- Go to MLFLOW server to look at all the results.
- saved_models folder will contain the final model after the pipeline is executed using MLFlow

## 🚀 Usage

```bash
# Run the Streamlit app
streamlit run saved_models/website.py
```

## Docker

```bash
# Build the docker image
docker build -t summary .

# Run the docker image
docker run -p 8501:8501 summary
```
