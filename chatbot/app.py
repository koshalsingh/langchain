# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# # from langchain_core.output_parser import StrOutputParser
# from langchain_core.output_parsers.base import BaseLLMOutputParser
# # from langchain_core.output_parsers.openai import OpenAIOutputParser

# import streamlit as st
# import os
# from dotenv import load_dotenv

# load_dotenv()
# # os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# # os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# #for langsmith tracing
# # os.environ["LANGCHAIN_TRACING_V2"] = "true"

# #prompt Template
# prompt = ChatPromptTemplate.from_message([
#     ("System", "you are a helpful assistant. Please respond to user query"),
#     ("User", "question: { question}")
# ])

# ## Streamlit Framework

# st.title("Langchain Chatbot DEMO with OPENAI")
# input_text = st.text_input("Enter your question here")


# ## openAI llm
# llm=ChatOpenAI(model="gpt-3.5-turbo")
# output_parser = BaseLLMOutputParser()

# chain = prompt|llm|output_parser

# # if input_text:
# #     response = chain.run(question=input_text)
# #     st.write(response)

# if input_text:
#     response = chain.invoke({'question':input_text})
#     st.write(response)

import streamlit as st
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers.base import BaseLLMOutputParser

# Load environment variables
load_dotenv()

# Uncomment and ensure environment variables are set properly
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
print(os.getenv("OPENAI_API_KEY"))
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
print(os.getenv("LANGCHAIN_API_KEY"))
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # For LangSmith tracing

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user query."),
    ("human", "Question: {question}")
])

## Streamlit Framework
st.title("Langchain Chatbot DEMO with OpenAI")
input_text = st.text_input("Enter your question here")

## OpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")
class CustomOutputParser(BaseLLMOutputParser):
    def parse_result(self, result):
        return result

output_parser = CustomOutputParser()

# Process user input
if input_text:
    response = llm.invoke(prompt.format(question=input_text))
    st.write("Response:",response)