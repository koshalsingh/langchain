import streamlit as st
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.llms import HuggingFaceHub  # Use Hugging Face API
import requests
# from streamlit_lottie import st_lottie

# Load environment variables
load_dotenv()

# Set API key for Hugging Face
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # For LangSmith tracing

# Function to load Lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Lottie animations
lottie_chatbot = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_kqel6cy2.json")

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

    Software Engineer & AI Enthusiast
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

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user query."),
    ("human", "Question: {question}")
])

input_text = st.text_input("Enter your question here")

# Initialize Llama 3 via Hugging Face
llm = HuggingFaceHub(
    repo_id="meta-llama/Meta-Llama-3-8B",  # Updated to Llama 3
    model_kwargs={"temperature": 0.7, "max_length": 100},
    huggingfacehub_api_token=HUGGINGFACE_API_KEY
)

# Process user input
if input_text:
    response = llm.invoke(prompt.format(question=input_text))
    st.write("Response:", response)

st.button("Answer")

# Footer
st.markdown("""
---
Created with ❤️ by Koshal Kumar
""")
