import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import math

# Available models for code explanation
MODELS = {
    "CodeT5-base-sum": "Salesforce/codet5-base-multi-sum"
}

@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("summarization", model=model, tokenizer=tokenizer)

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

        # Generate explanation
        with st.spinner("Explaining..."):
            code_chunks = chunk_code(code, explainer.tokenizer, max_length=512)
            explanations_list = []
            for idx, chunk in enumerate(code_chunks):
                if not chunk.strip():
                    continue  # Skip empty chunks
                # For summarization, just use the code chunk
                result = explainer(chunk, max_length=128, do_sample=False)[0]['summary_text']
                explanations_list.append(f"Chunk {idx+1}:\n" + result.strip())
            full_explanation = "\n\n".join(explanations_list)
            explanations[file.name] = full_explanation
            st.success("Explanation:")
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
                selected_files.append(file_selected)
                st.write(explanations[file_selected])
