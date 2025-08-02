import torch
import tiktoken
import os

# === Settings ===
MODEL_PATH = "summary.pth"
ONNX_DIR = "onnx_models"
ONNX_PATH = os.path.join(ONNX_DIR, "summarizer.onnx")
BLOCK_SIZE = 128  # Model's block size

# === Ensure output directory exists ===
os.makedirs(ONNX_DIR, exist_ok=True)

# === Load model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model = model.to(device).eval()

# === Prepare dummy input ===
tokenizer = tiktoken.get_encoding("gpt2")
dummy_text = "Summarize the following article.\n\n### Article:\nDummy article\n\n### Summary:\n"
input_ids = tokenizer.encode(dummy_text, allowed_special="all")

# Pad or truncate to BLOCK_SIZE
if len(input_ids) < BLOCK_SIZE:
    input_ids += [27156] * (BLOCK_SIZE - len(input_ids))
else:
    input_ids = input_ids[:BLOCK_SIZE]

input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)  # Shape: (1, 128)

# === Export to ONNX ===
torch.onnx.export(
    model,
    input_tensor,  # e.g., shape (1, 128)
    "onnx_models/summarizer.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "logits": {0: "batch_size", 1: "seq_length"}
    },
    opset_version=17,
    do_constant_folding=True
)

print(f"✅ Summarizer model exported to {ONNX_PATH}")
