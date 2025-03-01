# %% [markdown]
# ### Retriever and Chain with LangChain

# %%
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("attention.pdf")
docs = loader.load()
docs[:3]


# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
text_splitter.split_documents(docs[0:5])

# %%
documents = text_splitter.split_documents(docs)
documents[:3]

# %%
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# %%
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

db = FAISS.from_documents(documents,HuggingFaceEmbeddings())

# %%
db

# %%
query = "An attention function can be described as mapping a query "
result = db.similarity_search(query)
result[0].page_content

# %%
from langchain_community.llms import Ollama

llm = Ollama(model="llama3.2")
llm

# %%
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>
Question: {input}""")

# %%
from langchain.chains.combine_documents import create_stuff_documents_chain

document_chain = create_stuff_documents_chain(llm, prompt)

# %%
retriever = db.as_retriever()
retriever

# %%
from langchain.chains import create_retrieval_chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# %%
response = retrieval_chain.invoke({"input":"Scaled Dot-Product Attention"})

# %%
response['answer']

# %%



