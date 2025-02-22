import streamlit as st
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.llms import HuggingFaceHub, Ollama

# Load environment variables
load_dotenv()

# Set API key for Hugging Face
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# deepseek-r1:1.5b

# Check API Key
if not HUGGINGFACE_API_KEY:
    st.error("Please set your Hugging Face API key in the environment variables.")
    st.stop()

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user query. Don't create a longer content. the answer should be concise."),
    ("human", "Question: {question}")
])

# Streamlit Framework
st.title("Langchain Chatbot DEMO with Llama 3 (Hugging Face API)")
input_text = st.text_input("Enter your question here")

# Initialize Llama 3 via Hugging Face
llm = Ollama(
    model="deepseek-r1:1.5b"  # local deepseek model
    # model_kwargs={"temperature": 0.7, "max_length": 512},
    # huggingfacehub_api_token=HUGGINGFACE_API_KEY
)

# Process user input
if input_text:
    response = llm.invoke(prompt.format(question=input_text))
    st.write("Response:", response)
