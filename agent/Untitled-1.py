# %%
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# %%
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
tool_wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# %%
tool_wiki.name

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
retriver

# %%
from langchain.tools.retriever import create_retriever_tool
tool_lc = create_retriever_tool(retriver,"langsmith_search",
                      "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!")

tool_lc.name

# %%
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
tool_arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
tool_arxiv.name

# %%
tools = [tool_wiki, tool_lc, tool_arxiv]
tools

# %%
from langchain_community.llms import Ollama

r1_model = Ollama(model = "deepseek-r1:14b")
r15_model = Ollama(model = "deepseek-r1:1.5b")

# %%
from langchain import hub

prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages

# %%
## Agents
from langchain.agents import create_openai_tools_agent

agent = create_openai_tools_agent(r15_model, tools, prompt)
agent

# %%
# agent executor
from langchain.agents import AgentExecutor, AgentType, initialize_agent, load_tools

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor

# %%
agent_executor({"input": "Tell me about LangSmith"})
