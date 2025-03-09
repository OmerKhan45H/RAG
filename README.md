your task is to create a chatbot system for front end technologies use:
CSS, HTML, Javascript, Bootstrap. and for the backend use flask.
Your UI must be so modern and attractive with different animations and colour schemes. dark mod etc.
i have already created flask structure you just have to put files. make sure only create index file and app.py
for the text generation use this script:
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
import asyncio
import os

# Settings control global defaults
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Groq(model="llama3-70b-8192", api_key="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")

# Create a RAG tool using LlamaIndex
documents = SimpleDirectoryReader("/content/data").load_data()
index = VectorStoreIndex.from_documents(
    documents,
    # we can optionally override the embed_model here
    # embed_model=Settings.embed_model,
)
query_engine = index.as_query_engine(
    # we can optionally override the llm here
    # llm=Settings.llm,
)

response = query_engine.query("who is hasnain")
print(response)
