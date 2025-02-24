import streamlit as st
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq  # Use Groq API

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Langchain Chatbot", page_icon="🤖", layout="wide")

# Sidebar content
with st.sidebar:
    st.title("🤖 Langchain Chatbot")
    st.markdown("""
    Welcome to the Langchain Chatbot DEMO with Groq API. This platform allows you to interact with a powerful language model to generate comprehensive responses.
    """)
    st.markdown("### ⚡ Powered By")
    st.markdown("- **Groq's Ultra-fast Infrastructure**\n- **LLaMA or Mixtral Models**")

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
    st.markdown("Try different models for varied response styles!")

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
st.title("Langchain Chatbot DEMO with Groq API")

# API Key input
api_key = st.text_input("Enter your Groq API Key (gsk_xxx...)", type="password")

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user query."),
    ("human", "Question: {question}")
])

# Check API Key
if not api_key:
    st.warning("Please provide your Groq API key to use this chatbot.")
    st.info("To create a Groq API key, go to [Groq Console](https://console.groq.com/keys), sign up, and generate a key.")
    st.stop()

input_text = st.text_input("Enter your question here")

# Initialize Groq model
llm = ChatGroq(
    model="mixtral-8x7b-32768",  # Example Groq-supported model
    temperature=0.7,
    max_tokens=100,
    api_key=api_key  # Updated to use `api_key` as per latest langchain-groq
)

# Process user input with button
if st.button("Answer") and input_text:
    try:
        response = llm.invoke(prompt.format(question=input_text))
        st.write("Response:", response.content)  # Extract response content
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---\nCreated with ❤️ by Koshal Kumar")