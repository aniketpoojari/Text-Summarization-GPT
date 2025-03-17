import streamlit as st
import torch
from transformers import GPT2Tokenizer
from torch.nn import functional as F


@st.cache_resource
def load_model():
    model_path = "summary.pth"
    model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    return model


model = load_model()

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Add special tokens
special_tokens_dict = {"pad_token": "<PAD>", "sep_token": "<SEP>"}
tokenizer.add_special_tokens(special_tokens_dict)


def generate_text(
    model, context, device, max_length=128, stop_token_id=50257, top_k=50
):
    """
    Generates text from the given context using a pre-trained transformer model.

    Args:
        context (tensor): The initial input context (e.g., a prompt or seed text).
        model (torch.nn.Module): The pre-trained transformer model.
        device (torch.device): The device (CPU or GPU) to run the model on.
        max_length (int): The maximum length of the generated sequence.
        stop_token_id (int): The token ID that signals the end of the generation.
        top_k (int): The number of top probable tokens to consider for each generation step.

    Returns:
        tensor: The generated text sequence as a tensor of token IDs.
    """

    # Ensure the input context is a tensor and move it to the appropriate device
    context = torch.tensor(context, dtype=torch.long, device=device)

    # Add batch dimension (batch size of 1)
    context = context.unsqueeze(0)

    # Start with the initial context as the generated sequence
    generated = context

    # Set the model to evaluation mode (no gradients needed)
    model.eval()

    # Generate tokens until the desired length is reached or stop token is encountered
    with torch.no_grad():
        while generated.shape[1] < max_length:
            # Pass the current generated sequence through the model to get logits
            logits = model(generated)

            # Focus on the logits of the last token generated
            logits = logits[:, -1, :]

            # Convert logits to probabilities using softmax
            probs = F.softmax(logits, dim=-1)

            # Get the top-k probable tokens and their corresponding probabilities
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)

            # Sample one token from the top-k options using multinomial sampling
            sampled_token_ix = torch.multinomial(topk_probs, 1)  # (batch_size, 1)

            # Gather the corresponding token ID based on the sampled index
            next_token = torch.gather(
                topk_indices, -1, sampled_token_ix
            )  # (batch_size, 1)

            # Check if the stop token is generated (e.g., end-of-sequence token)
            if next_token == stop_token_id:
                break

            # Append the sampled token to the generated sequence
            generated = torch.cat((generated, next_token), dim=1)

    # Return the generated sequence of token IDs
    return generated


# Title of the app
st.title("Text Summarizer")

user_input = st.text_input("Enter some text:")

if st.button("Start Prediction"):

    if user_input:

        input_text = tokenizer.encode(user_input) + [
            tokenizer.sep_token_id
        ]  # enoded + sep

        output_text = generate_text(model, input_text, "cpu")

        output_text = tokenizer.decode(output_text.tolist()[0])

        summary = output_text.split("<SEP>")[1]

        st.write("Summary: ", summary)
