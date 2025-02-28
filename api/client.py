import requests
import streamlit as st

def get_deepseek_response(prompt):
    response = requests.post("http://localhost:8000/essay/invoke", 
                             json={"input": {'topic': prompt}})
    return response.json()['output'] 


def get_llama_response(prompt):
    response = requests.post("http://localhost:8000/poem/invoke", 
                             json={"input": {'topic': prompt}})
    return response.json()['output']    

st.title("Langchain Chatbot DEMO with Llama 2 API")
input_text_openai = st.text_input("Write an essay about ")
input_text_llama2 = st.text_input("Write a poem about ")

if input_text_openai:
    st.write("Response:", get_deepseek_response(input_text_openai))

if input_text_llama2:
    st.write("Response:", get_llama_response(input_text_llama2))

