import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import os
from typing import List, Dict
import time

# Configuration
MODELS = {
    "CodeT5 (small)": "Salesforce/codet5-small",
    "CodeT5 (base)": "Salesforce/codet5-base",
    "CodeBERT": "microsoft/codebert-base",
    "PLBART": "uclanlp/plbart-multi_task-en_XX",
}

DEFAULT_MODEL = "Salesforce/codet5-base"

# Initialize session state
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = None
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "explanations" not in st.session_state:
    st.session_state.explanations = {}

def load_model(model_name: str):
    """Load the selected model from Hugging Face"""
    if st.session_state.current_model == model_name:
        return
    
    st.session_state.current_model = model_name
    model_path = MODELS[model_name]
    
    with st.spinner(f"Loading {model_name} (this may take a minute)..."):
        try:
            if "codet5" in model_path.lower():
                # CodeT5 is a sequence-to-sequence model
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                st.session_state.model_loaded = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer
                )
            elif "codebert" in model_path.lower():
                # CodeBERT is better for embeddings and classification
                st.session_state.model_loaded = pipeline(
                    "feature-extraction",
                    model=model_path
                )
            elif "plbart" in model_path.lower():
                # PLBART is a multilingual model
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                st.session_state.model_loaded = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer
                )
            st.success(f"{model_name} loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            st.session_state.model_loaded = None

def generate_explanation(code: str, model_name: str) -> str:
    """Generate explanation for the given code using the loaded model"""
    if not st.session_state.model_loaded:
        return "Model not loaded. Please select and load a model first."
    
    try:
        if "codet5" in model_name.lower() or "plbart" in model_name.lower():
            # For sequence-to-sequence models
            prompt = f"Explain the following code:\n{code}"
            result = st.session_state.model_loaded(
                prompt,
                max_length=512,
                num_return_sequences=1
            )
            return result[0]['generated_text']
        elif "codebert" in model_name.lower():
            # For CodeBERT, we'll use a simpler approach since it's not seq2seq
            return "CodeBERT is better for embeddings. Consider using CodeT5 for explanations."
        else:
            return "Model type not recognized for explanation generation."
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

def display_explanation(file_name: str, explanation: str):
    """Display the explanation in an organized way"""
    with st.expander(f"Explanation for {file_name}"):
        st.code(explanation, language="text")

def save_uploaded_files(uploaded_files) -> List[Dict]:
    """Save uploaded files temporarily and return their content"""
    files_data = []
    for uploaded_file in uploaded_files:
        file_content = uploaded_file.read().decode("utf-8")
        files_data.append({
            "name": uploaded_file.name,
            "content": file_content
        })
    return files_data

def main():
    st.title("📝 CodeExplainerBot")
    st.markdown("""
    Upload your code files and get explanations using open-source models from Hugging Face.
    """)
    
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    selected_model = st.sidebar.selectbox(
        "Choose a model",
        list(MODELS.keys()),
        index=1  # Default to CodeT5 base
    )
    
    if st.sidebar.button("Load Model"):
        load_model(selected_model)
    
    # Main content area
    st.header("Upload Code Files")
    uploaded_files = st.file_uploader(
        "Choose code files",
        type=["py", "js", "java", "cpp", "go", "rs", "sh", "html", "css"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        files_data = save_uploaded_files(uploaded_files)
        
        if st.button("Explain All Files"):
            if not st.session_state.model_loaded:
                st.warning("Please load a model first!")
                return
            
            progress_bar = st.progress(0)
            total_files = len(files_data)
            
            for i, file_data in enumerate(files_data):
                progress_bar.progress((i + 1) / total_files)
                with st.spinner(f"Analyzing {file_data['name']}..."):
                    explanation = generate_explanation(
                        file_data["content"],
                        selected_model
                    )
                    st.session_state.explanations[file_data["name"]] = explanation
                    display_explanation(file_data["name"], explanation)
                time.sleep(0.5)  # To avoid rate limiting
            
            st.success("Analysis complete!")
    
    # Show previous explanations if available
    if st.session_state.explanations:
        st.header("Previous Explanations")
        for file_name, explanation in st.session_state.explanations.items():
            display_explanation(file_name, explanation)

if __name__ == "__main__":
    main()
