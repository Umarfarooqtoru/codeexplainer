import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import math

# Available models for code explanation
MODELS = {
    "CodeT5-base-sum": "Salesforce/codet5-base-multi-sum",
    "CodeT5-small-sum": "Salesforce/codet5-small",
    "CodeBERTa": "huggingface/CodeBERTa-small-v1",
    "StarCoder": "bigcode/starcoderbase"
}

@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # Use summarization for CodeT5, text2text-generation for others
    if "codet5" in model_name.lower():
        return pipeline("summarization", model=model, tokenizer=tokenizer)
    else:
        return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def chunk_code(code, tokenizer, max_length=512):
    tokens = tokenizer.encode(code)
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk_tokens = tokens[i:i+max_length]
        chunk = tokenizer.decode(chunk_tokens)
        chunks.append(chunk)
    return chunks

st.title("CodeExplainerBot")

# Model selection
model_choice = st.selectbox("Select a model", list(MODELS.keys()))
model_id = MODELS[model_choice]
explainer = load_model(model_id)

# File uploader
uploaded_files = st.file_uploader("Upload code files", type=["py", "js", "java", "cpp", "txt"], accept_multiple_files=True)

if uploaded_files:
    explanations = {}
    for file in uploaded_files:
        code = file.read().decode("utf-8")
        st.subheader(f"File: {file.name}")
        st.code(code, language="python")  # Adjust language as needed

        # Generate line-by-line explanation
        with st.spinner("Explaining line by line..."):
            lines = code.splitlines()
            line_explanations = []
            for idx, line in enumerate(lines):
                if not line.strip():
                    continue  # Skip empty lines
                prompt = f"Explain this line of code:\n{line}"
                # Calculate a reasonable max_length for the output (shorter than input for summarization)
                input_length = len(line.split())
                max_length = max(4, min(16, input_length // 2 + 2))
                if "codet5" in model_id.lower():
                    result = explainer(line, max_length=max_length, do_sample=False)[0].get('summary_text', '')
                else:
                    result = explainer(prompt, max_length=max_length, do_sample=False)[0].get('generated_text', '')
                line_explanations.append(f"Line {idx+1}: {line}\nExplanation: {result.strip()}\n")
            full_explanation = "\n".join(line_explanations)
            explanations[file.name] = full_explanation
            st.success("Line-by-line Explanation:")
            st.write(full_explanation)

    # Comparison feature (up to 4 files)
    if len(explanations) > 1:
        st.subheader("Compare Explanations")
        files = list(explanations.keys())
        num_cols = min(len(files), 4)
        cols = st.columns(num_cols)
        selected_files = []
        for i, col in enumerate(cols):
            with col:
                file_key = f"file{i+1}"
                file_selected = st.selectbox(f"File {i+1}", files, key=file_key)
