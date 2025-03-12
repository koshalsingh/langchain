# %%
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# %%
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
tool_wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
# tool_wiki.name

# %%
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()

embeddings = OllamaEmbeddings(model="llama3.2")

documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
vectordb = FAISS.from_documents(documents, embeddings)

retriver = vectordb.as_retriever()
# retriver

# %%
from langchain.tools.retriever import create_retriever_tool
tool_lc = create_retriever_tool(retriver,"langsmith_search",
                      "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!")

# tool_lc.name

# %%
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
tool_arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
# tool_arxiv.name

tools = [tool_wiki, tool_lc, tool_arxiv]
tools

from langchain_groq import ChatGroq

# Initialize the language model of Groq
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Bind tools to the language model
llm_with_tools = llm.bind_tools(tools)

print(hasattr(llm_with_tools, "bind_tools"))

from langchain import hub

prompt = hub.pull("hwchase17/openai-functions-agent")
print(prompt.messages)

from langchain.agents import AgentExecutor, create_tool_calling_agent, tool

# Bind tools to the language model
llm_with_tools = llm.bind_tools(tools)

# Create the agent
agent = create_tool_calling_agent(llm_with_tools, tools, prompt)

from langchain.agents import AgentExecutor
# Create an agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

from langchain.agents import AgentExecutor, AgentType, initialize_agent, load_tools

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# agent_executor
agent_executor.invoke({"input":"Tell me about LangSmith"})
agent_executor.invoke({"input":"What's the paper 1605.08386 about?"})
