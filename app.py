import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base")
    return tokenizer, model

tokenizer, model = load_model()

# Function to explain code
def explain_code(code):
    inputs = tokenizer("summarize: " + code, return_tensors="pt", truncation=True, max_length=512)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=2)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit UI
st.title("🤖 CodeExplainerBot")
st.write("Upload multiple code files and get explanations using an AI model.")

uploaded_files = st.file_uploader("Upload Code Files", type=["py", "js", "java", "cpp"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        code = file.read().decode("utf-8")
        st.subheader(f"📄 {file.name}")
        with st.expander("🔍 View Code"):
            st.code(code, language="python")
        with st.spinner("Generating explanation..."):
            explanation = explain_code(code)
        st.success("Explanation:")
        st.write(explanation)
