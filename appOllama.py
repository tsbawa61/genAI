import streamlit as st
import ollama
import subprocess
import time
import os
import requests
from datetime import datetime

# --- MLOps Setup Functions (reusing from previous examples) ---

def check_ollama_api_health(api_url='http://localhost:11434'):
    """Checks if the Ollama API endpoint is responsive."""
    health_endpoint = f"{api_url}/api/tags" # /api/tags is a common endpoint to list models
    try:
        response = requests.get(health_endpoint, timeout=2)
        response.raise_for_status() # Raise an exception for HTTP errors
        return True
    except requests.exceptions.RequestException:
        return False

def setup_ollama_environment(model_name="llama3", ollama_host='0.0.0.0:11434', start_server=True):
    if start_server:
        # Kill any existing Ollama processes to ensure a clean start
        subprocess.run(['pkill', '-9', 'ollama'], capture_output=True, text=True, check=False)
        time.sleep(1) # Give processes time to terminate

        # Start the Ollama server in the background
        os.environ['OLLAMA_HOST'] = ollama_host
        subprocess.Popen(["nohup", "/usr/local/bin/ollama", "serve"], stdout=open("ollama_streamlit.log", "a"), stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        time.sleep(5) # Give Ollama server enough time to start

    # Check Ollama server health
    api_url = f"http://{ollama_host.split(':')[0]}:11434"
    if not check_ollama_api_health(api_url):
        st.error("Ollama server is not healthy. Please check logs for errors.")
        return False

    # Check if the specified model is pulled
    try:
        list_models_output = subprocess.run(['/usr/local/bin/ollama', 'list'], capture_output=True, text=True, check=True)
        if model_name not in list_models_output.stdout:
            st.info(f"'{model_name}' model not found. Attempting to pull it now. This may take some time...")
            subprocess.run(['/usr/local/bin/ollama', 'pull', model_name], capture_output=True, text=True, check=True)
            st.success(f"'{model_name}' model pulled successfully.")
    except Exception as e:
        st.error(f"Error checking/pulling Ollama model: {e}")
        return False

    return True

# --- Streamlit App --- #
st.set_page_config(page_title="LLM MLOps with Streamlit", layout="wide")
st.title("🧠 LLM Inference Dashboard (Ollama & Streamlit)")

# Sidebar for Configuration
st.sidebar.header("Configuration")
selected_model = st.sidebar.selectbox("Select LLM Model", ["llama3"], index=0) # Add other models if pulled
start_ollama = st.sidebar.checkbox("Start Ollama Server (if not running)", value=True)

# Initialize Ollama environment
if 'ollama_ready' not in st.session_state:
    st.session_state.ollama_ready = False

if start_ollama and not st.session_state.ollama_ready:
    with st.spinner("Setting up Ollama environment and pulling model..."):
        st.session_state.ollama_ready = setup_ollama_environment(model_name=selected_model, start_server=True)
    if st.session_state.ollama_ready:
        st.sidebar.success("Ollama is ready!")
    else:
        st.sidebar.error("Ollama setup failed.")

elif not start_ollama and not st.session_state.ollama_ready:
    st.sidebar.warning("Ollama server is not configured to start. Please ensure it's running manually or check the box.")

# Main App Interface
st.header("LLM Inference")

if st.session_state.ollama_ready:
    user_prompt = st.text_area("Enter your prompt:", "What is MLOps for LLMs?")
    if st.button("Generate Response"):
        if user_prompt:
            with st.spinner("Generating response..."):
                try:
                    response_stream = ollama.generate(model=selected_model, prompt=user_prompt, stream=True)
                    full_response = ""
                    response_container = st.empty()
                    for chunk in response_stream:
                        if 'response' in chunk:
                            full_response += chunk['response']
                            response_container.markdown(full_response + "▌") # Add blinking cursor
                    response_container.markdown(full_response) # Final response without cursor

                    # Log interaction (basic MLOps monitoring)
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
else:
    st.warning("Ollama environment is not ready. Please ensure the server is running and the model is pulled.")


# MLOps Monitoring Section
st.markdown("--- ")
st.header("MLOps Monitoring & Logs")

if 'logs' not in st.session_state:
    st.session_state.logs = []

if st.session_state.logs:
    st.dataframe(st.session_state.logs, use_container_width=True)
    if st.button("Clear Logs"):
        st.session_state.logs = []
        st.experimental_rerun()
else:
    st.info("No LLM interactions logged yet.")

st.caption("This Streamlit app showcases basic LLM inference and interaction logging, which are foundational for MLOps practices.")
