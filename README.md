# Text Summarization GPT (From Scratch)

A custom 125-million parameter GPT-based language model built from scratch in PyTorch to perform abstractive text summarization. 

This project trains an autoregressive language model on the **XSum dataset** (news articles paired with single-sentence summaries). It features a two-phase training pipeline: an unsupervised causal language modeling pretraining phase, followed by a supervised fine-tuning phase specifically for summarization using teacher forcing.

## 🌟 Key Features

- **Custom GPT Architecture**: A causal transformer decoder trained from scratch.
- **Two-Phase Training**: 
  - *Pretraining*: Learns language structure by predicting the next token.
  - *Fine-tuning*: Learns to summarize using Teacher Forcing on the target summaries.
- **128-Token Context Window**: Optimized for short to medium-length news articles.
- **MLOps Orchestration**: Full reproducibility using **DVC** pipelines and **MLflow** experiment tracking.
- **Interactive UI**: Deployed via a containerized **Streamlit** application.

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, HuggingFace Transformers (for tokenization)
- **MLOps**: DVC, MLflow
- **Evaluation**: ROUGE-L, Cross-Entropy
- **Deployment**: Streamlit, Docker

## 🚚 Installation

```bash
# Clone the repository
git clone https://github.com/aniketpoojari/Text-Summarization-GPT.git
cd Text-Summarization-GPT

# Install dependencies
pip install -r requirements.txt
```

## 📊 Data Preparation

1. Download the dataset from [Xsum on Hugging Face](https://huggingface.co/datasets/sentence-transformers/xsum).
2. Save `train.src.txt` and `train.tgt.txt` in `data/raw/summary/`.
3. Track the files using DVC: `dvc add data/raw/summary/`

## 🤖 Model Training

The entire pretraining and fine-tuning pipeline is automated via DVC.

```bash
# Start the MLflow server in the background
mlflow server --backend-store-uri sqlite:///mlflow.db --host localhost

# Run the DVC training pipeline
dvc repro
```

## 📈 Evaluation & Results

- **Pretraining**: Evaluated using Cross-Entropy Loss.
- **Summarization**: Evaluated using the **ROUGE-L** metric. The fine-tuned model achieved a **ROUGE-L score of 0.54** on the validation set.
- The best-performing model artifacts are automatically tracked and conditionally saved to `saved_models/`.

## 🚀 Usage (Streamlit App)

You can run the interactive Streamlit UI to test the summarization model locally:

```bash
streamlit run saved_models/website.py
```

### Run with Docker

```bash
# Build the Docker image
docker build -t text-summarizer .

# Run the container
docker run -p 8501:8501 text-summarizer
```
