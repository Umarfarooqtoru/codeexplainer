import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
from typing import List, Dict
import time
import platform

# Configuration
MODELS = {
    "CodeT5 (small)": "Salesforce/codet5-small",
    "CodeT5 (base)": "Salesforce/codet5-base",
    "CodeLlama (7B)": "codellama/CodeLlama-7b-hf",
    "StarCoder (base)": "bigcode/starcoderbase",
}

# Reduce logging noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize session state
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = None
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "explanations" not in st.session_state:
    st.session_state.explanations = {}
if "device" not in st.session_state:
    st.session_state.device = "cuda" if torch.cuda.is_available() else "cpu"

def get_system_info():
    """Return system information for debugging"""
    return {
        "system": platform.system(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": st.session_state.device,
        "cuda_available": torch.cuda.is_available(),
    }

def load_model(model_name: str):
    """Load the selected model from Hugging Face with optimizations"""
    if st.session_state.current_model == model_name and st.session_state.model_loaded:
        return
    
    st.session_state.current_model = model_name
    model_path = MODELS[model_name]
    
    with st.spinner(f"Loading {model_name} (this may take several minutes)..."):
        try:
            start_time = time.time()
            
            # Common loading parameters for all models
            common_params = {
                "device_map": "auto",
                "torch_dtype": torch.float16 if st.session_state.device == "cuda" else torch.float32,
                "low_cpu_mem_usage": True
            }
            
            if "codet5" in model_path.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **common_params)
                st.session_state.model_loaded = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=st.session_state.device
                )
            elif "llama" in model_path.lower() or "starcoder" in model_path.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **common_params)
                st.session_state.model_loaded = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=st.session_state.device
                )
            
            load_time = time.time() - start_time
            st.success(f"{model_name} loaded successfully in {load_time:.2f} seconds!")
            st.json(get_system_info())
            
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            st.session_state.model_loaded = None

def generate_explanation(code: str, model_name: str) -> str:
    """Generate explanation for the given code using the loaded model"""
    if not st.session_state.model_loaded:
        return "Model not loaded. Please select and load a model first."
    
    try:
        if "codet5" in model_name.lower():
            prompt = f"Explain the following code in detail:\n```\n{code}\n```"
            result = st.session_state.model_loaded(
                prompt,
                max_length=512,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            return result[0]['generated_text']
        elif "llama" in model_name.lower() or "starcoder" in model_name.lower():
            prompt = f"""Below is a code snippet. Please provide a detailed explanation of what it does.

Code: Explanation:"""
            result = st.session_state.model_loaded(
                prompt,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            return result[0]['generated_text']
        else:
            return "Model type not recognized for explanation generation."
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

def display_explanation(file_name: str, explanation: str):
    """Display the explanation in an organized way"""
    with st.expander(f"📄 {file_name}"):
        st.markdown("### Code Explanation")
        st.write(explanation)
        st.markdown("---")

def save_uploaded_files(uploaded_files) -> List[Dict]:
    """Save uploaded files temporarily and return their content"""
    files_data = []
    for uploaded_file in uploaded_files:
        try:
            file_content = uploaded_file.read().decode("utf-8")
            files_data.append({
                "name": uploaded_file.name,
                "content": file_content,
                "size": len(file_content)
            })
        except Exception as e:
            st.warning(f"Could not read file {uploaded_file.name}: {str(e)}")
    return files_data

def main():
    st.set_page_config(
        page_title="CodeExplainerBot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 CodeExplainerBot")
    st.markdown("""
    Upload your code files and get detailed explanations using state-of-the-art AI models.
    Supports multiple programming languages and large code models.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        selected_model = st.selectbox(
            "Choose a model",
            list(MODELS.keys()),
            index=1
        )
        
        st.markdown(f"**Current Device:** `{st.session_state.device.upper()}`")
        
        if st.button("Load Model", type="primary"):
            load_model(selected_model)
        
        st.markdown("---")
        st.markdown("### System Info")
        st.json(get_system_info())
    
    # Main content area
    tab1, tab2 = st.tabs(["📁 Upload Files", "📚 Documentation"])
    
    with tab1:
        st.header("Upload Your Code Files")
        uploaded_files = st.file_uploader(
            "Choose files to analyze",
            type=["py", "js", "java", "cpp", "go", "rs", "sh", "html", "css", "php", "rb", "swift"],
            accept_multiple_files=True,
            help="Select one or more code files to analyze"
        )
        
        if uploaded_files:
            files_data = save_uploaded_files(uploaded_files)
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("Analyze All Files", type="primary"):
                    if not st.session_state.model_loaded:
                        st.warning("Please load a model first!")
                        st.stop()
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    total_files = len(files_data)
                    
                    for i, file_data in enumerate(files_data):
                        progress = (i + 1) / total_files
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {i+1}/{total_files}: {file_data['name']}")
                        
                        with st.spinner(f"Analyzing {file_data['name']}..."):
                            explanation = generate_explanation(
                                file_data["content"],
                                selected_model
                            )
                            st.session_state.explanations[file_data["name"]] = {
                                "explanation": explanation,
                                "model": selected_model,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            display_explanation(file_data["name"], explanation)
                        
                        time.sleep(0.5)  # Rate limiting
                    
                    progress_bar.empty()
                    status_text.success("Analysis complete!")
            
            with col2:
                if st.session_state.explanations:
                    st.header("Analysis History")
                    for file_name, data in st.session_state.explanations.items():
                        with st.expander(f"🕒 {data['timestamp']} - {file_name}"):
                            st.markdown(f"**Model used:** {data['model']}")
                            st.write(data["explanation"])
    
    with tab2:
        st.header("How to Use CodeExplainerBot")
        st.markdown("""
        ### Step-by-Step Guide
        
        1. **Select a Model**: Choose from available models in the sidebar
        - CodeT5: Good balance of speed and accuracy
        - CodeLlama: More powerful but requires more resources
        - StarCoder: Excellent for larger codebases
        
        2. **Load the Model**: Click "Load Model" in the sidebar
        
        3. **Upload Files**: Select one or more code files
        
        4. **Analyze**: Click "Analyze All Files" to get explanations
        
        ### Tips for Best Results
        - For large files (>500 lines), consider splitting into smaller chunks
        - The first model load may take several minutes (models are cached)
        - GPU acceleration is automatically used if available
        """)

if __name__ == "__main__":
    main()
