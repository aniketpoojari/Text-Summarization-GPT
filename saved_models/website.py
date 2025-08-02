import streamlit as st
import torch
import numpy as np
import onnxruntime as ort
import tiktoken
import time

# --- CONFIG ---
BLOCK_SIZE = 128
ONNX_MODEL_PATH = "onnx_models/summarizer.onnx"
PYTORCH_MODEL_PATH = "summary.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = tiktoken.get_encoding("gpt2")

# --- Model Loaders ---
@st.cache_resource
def load_pytorch_model():
    model = torch.load(PYTORCH_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.eval()
    return model

@st.cache_resource
def load_onnx_model():
    providers = ["CUDAExecutionProvider"] if DEVICE == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
    return session

# --- Inference Functions ---
def generate_pytorch(model, article, tokenizer, temperature=0.7, top_k=50, max_length=BLOCK_SIZE):
    prompt = f"Summarize the following article.\n\n### Article:\n{article}\n\n### Summary:\n"
    l = len(prompt)
    prompt_ids = tokenizer.encode(prompt, allowed_special="all")
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    while prompt_tensor.shape[1] <= max_length:
        out = model(prompt_tensor)
        logits = out[:, -1, :]
        if top_k > 0:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)
        if temperature > 0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next.item() == 50256:
            break
        prompt_tensor = torch.cat((prompt_tensor, idx_next), dim=1)
    return tokenizer.decode(prompt_tensor[0].tolist())[l:]

def generate_onnx(session, article, tokenizer, temperature=0.7, top_k=50, max_length=BLOCK_SIZE):
    prompt = f"Summarize the following article.\n\n### Article:\n{article}\n\n### Summary:\n"
    l = len(prompt)
    input_ids = tokenizer.encode(prompt, allowed_special="all")
    generated_ids = list(input_ids)
    for _ in range(max_length - len(input_ids)):
        input_array = np.array([generated_ids[-max_length:]], dtype=np.int64)
        outputs = session.run(None, {"input_ids": input_array})
        logits = outputs[0][:, -1, :]  # last token logits
        if top_k > 0:
            top_k_indices = np.argpartition(-logits, top_k, axis=-1)[:, :top_k]
            top_k_logits = np.take_along_axis(logits, top_k_indices, axis=-1)
            min_val = np.min(top_k_logits, axis=-1, keepdims=True)
            logits = np.where(logits < min_val, -np.inf, logits)
        if temperature > 0:
            logits = logits / temperature
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            idx_next = int(np.random.choice(logits.shape[-1], p=probs[0]))
        else:
            idx_next = int(np.argmax(logits, axis=-1)[0])
        if idx_next == 50256:
            break
        generated_ids.append(idx_next)
    return tokenizer.decode(generated_ids)[l:]

# --- Streamlit UI ---
st.title("Text Summarizer")

# Model selection
model_choice = st.selectbox(
    "Select inference backend:",
    ("PyTorch", "ONNX"),
    index=1  # default to ONNX
)

user_input = st.text_area("Enter text to summarize:", height=200)
col1, col2 = st.columns(2)
temperature = col1.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
top_k = col2.slider("Top-k", 0, 100, 50, 1)
model = load_pytorch_model()
session = load_onnx_model()

if st.button("Generate Summary"):
    if user_input:
        with st.spinner("Generating summary..."):
            if model_choice == "PyTorch":
                start_time = time.time()
                output_text = generate_pytorch(model, user_input, tokenizer, temperature=temperature, top_k=top_k)
                end_time = time.time()
            else:
                start_time = time.time()
                output_text = generate_onnx(session, user_input, tokenizer, temperature=temperature, top_k=top_k)
                end_time = time.time()
            execution_time = end_time - start_time
            st.write(f"Execution time: {execution_time:.2f} seconds")
            st.subheader("Generated Summary:")
            st.write(output_text)