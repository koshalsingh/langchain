import streamlit as st
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.llms import HuggingFaceHub  # Use Hugging Face API
import requests

# Set page configuration
st.set_page_config(page_title="Langchain Chatbot", page_icon="🤖", layout="wide")

# Sidebar content
with st.sidebar:
    st.title("🤖 Langchain Chatbot")
    st.markdown("""
    Welcome to the Langchain Chatbot DEMO with Llama 3 (Hugging Face API). This platform allows you to interact with a powerful language model to generate comprehensive responses.
    """)
    st.markdown("### ⚡ Powered By")
    st.markdown("- **LLAMA-3.3 70B Model**\n- **Groq's Ultra-fast Infrastructure**")

    st.markdown("### 🔑 Key Features")
    st.markdown("""
    - 📖 Comprehensive Responses
    - ⚡ Lightning-fast Processing
    - 🎨 Customizable Content
    - 📁 Multiple Export Formats
    - ⚙️ Advanced Configuration
    - 📚 Topic-focused Structure
    """)

    st.markdown("### 💡 Tips")
    st.markdown("Try the advanced mode for more control over your book generation!")

    st.markdown("### 👨‍💻 About Developer")
    st.markdown("""
    **Koshal Kumar**

    Data Scientist & AI Enthusiast
    """)

    st.markdown("""
    **Expertise**
    - 🤖 Prompt Engineering
    - 🌐 GenAI Development
    - 🧠 NLP & Machine Learning
    - 🌐 RESTful APIs
    - 🤖 Chatbot Architecture
    """)

    st.markdown("[LinkedIn](https://www.linkedin.com/in/koshalsingh) [GitHub](https://github.com/koshalsingh)")

# Main content
st.title("Langchain Chatbot DEMO with Llama 3 (Hugging Face API)")


# API Key input
api_key = st.text_input("Enter your Hugging Face API Key (api_xxx...)")

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user query."),
    ("human", "Question: {question}")
])

# Check API Key
if not api_key:
    st.warning("Please provide your Hugging Face API key to use this chatbot.")
    st.info("To create a Hugging Face API key, follow these steps:\n1. Go to [Hugging Face API Key Page](https://huggingface.co/settings/tokens)\n2. Click on 'Create new token'.\n3. Copy the token and paste it here.")
    st.stop()

input_text = st.text_input("Enter your question here")

# Initialize Llama 3 via Hugging Face
llm = HuggingFaceHub(
    repo_id="meta-llama/Meta-Llama-3-8B",  # Updated to Llama 3
    model_kwargs={"temperature": 0.7, "max_length": 100},
    huggingfacehub_api_token=api_key
)

# Process user input
if input_text:
    response = llm.invoke(prompt.format(question=input_text))
    st.write("Response:", response)

if st.button("Answer"):
    st.write("Thinking...")

# Footer
st.markdown("""
---
Created with ❤️ by Koshal Kumar
""")
