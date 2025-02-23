import streamlit as st
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.llms import Ollama

# Load environment variables
load_dotenv()

# Set API key for Hugging Face
# HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # For LangSmith tracing

# deepseek-r1:1.5b

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
)

# Process user input
if input_text:
    response = llm.invoke(prompt.format(question=input_text))
    st.write("Response:", response)
