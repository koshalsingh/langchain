import streamlit as st
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.llms import HuggingFaceHub  # Use Hugging Face API

# Load environment variables
load_dotenv()

# Set API key for Hugging Face
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # For LangSmith tracing    

# Check API Key
if not HUGGINGFACE_API_KEY:
    st.error("Please set your Hugging Face API key in the environment variables.")
    st.stop()

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user query."),
    ("human", "Question: {question}")
])

# Streamlit Framework
st.title("Langchain Chatbot DEMO with Llama 3 (Hugging Face API)")
input_text = st.text_input("Enter your question here")

# Initialize Llama 3 via Hugging Face
llm = HuggingFaceHub(
    repo_id="meta-llama/Meta-Llama-3-8B",  # Updated to Llama 3
    model_kwargs={"temperature": 0.7, "max_length": 512},
    huggingfacehub_api_token=HUGGINGFACE_API_KEY
)

# Process user input
if input_text:
    response = llm.invoke(prompt.format(question=input_text))
    st.write("Response:", response)
