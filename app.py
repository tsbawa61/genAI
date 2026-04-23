import streamlit as st
from transformers import pipeline
from datetime import datetime

# --- Setup Hugging Face Pipeline ---
# Use a small, efficient model for fast inference
generator = pipeline("text-generation", model="distilgpt2")

# --- Streamlit App ---
st.set_page_config(page_title="LLM Inference Dashboard", layout="wide")
st.title("🧠 LLM Inference Dashboard (Transformers & Streamlit)")

# Sidebar for Configuration
st.sidebar.header("Configuration")
selected_model = st.sidebar.selectbox("Select LLM Model", ["distilgpt2"], index=0)

# Initialize logs
if 'logs' not in st.session_state:
    st.session_state.logs = []

# Main App Interface
st.header("LLM Inference")

user_prompt = st.text_area("Enter your prompt:", "What is MLOps for LLMs?")
if st.button("Generate Response"):
    if user_prompt:
        with st.spinner("Generating response..."):
            try:
                result = generator(user_prompt, max_length=100, num_return_sequences=1)
                full_response = result[0]["generated_text"]

                response_container = st.empty()
                response_container.markdown(full_response)

                # Log interaction
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.logs.append({
                    "timestamp": timestamp,
                    "model": selected_model,
                    "prompt": user_prompt,
                    "response_length": len(full_response.split()),
                    "status": "success"
                })
                st.success("Response generated.")
            except Exception as e:
                st.error(f"Error generating response: {e}")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.logs.append({
                    "timestamp": timestamp,
                    "model": selected_model,
                    "prompt": user_prompt,
                    "response_length": 0,
                    "status": f"error: {str(e)[:50]}..."
                })
    else:
        st.warning("Please enter a prompt.")

# MLOps Monitoring Section
st.markdown("--- ")
st.header("MLOps Monitoring & Logs")

if st.session_state.logs:
    st.dataframe(st.session_state.logs, use_container_width=True)
    if st.button("Clear Logs"):
        st.session_state.logs = []
        st.rerun()
else:
    st.info("No LLM interactions logged yet.")

st.caption("This Streamlit app showcases basic LLM inference and interaction logging, using Hugging Face Transformers.")
