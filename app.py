import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Available models for code explanation
MODELS = {
    "CodeT5-base": "Salesforce/codet5-base",
    "StarCoder": "bigcode/starcoderbase"
}

@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

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
            prompt = f"Explain the following code:\n{code}"
            result = explainer(prompt, max_length=256, do_sample=False)[0]['generated_text']
            explanations[file.name] = result
            st.success("Explanation:")
            st.write(result)

    # Comparison feature
    if len(explanations) > 1:
        st.subheader("Compare Explanations")
        files = list(explanations.keys())
        col1, col2 = st.columns(2)
        with col1:
            file1 = st.selectbox("File 1", files, key="file1")
            st.write(explanations[file1])
        with col2:
            file2 = st.selectbox("File 2", files, key="file2")
            st.write(explanations[file2])
