from langchain.chains import StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from prompt_templates import memory_prompt_template
import chromadb
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def create_llm(model_name=config['model_name']['name'], model_type=config['model_type'], model_config = config['model_config']):
    llm = Ollama(model=model_name)
    return llm

def create_embeddings(embedding_name=config['embeddings_name']):
    embeddings = HuggingFaceInstructEmbeddings(model_name = embedding_name)
    return embeddings

def create_chat_memory(chat_history, input_key):
    return ConversationBufferWindowMemory(
        memory_key="history",  # Keeps history consistent
        chat_memory=chat_history,
        input_key=input_key,  # Pass input key dynamically
        k=5,
    )

def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)

def create_llm_chain(llm, chat_prompt, memory):
    return LLMChain(llm=llm, prompt = chat_prompt, memory = memory)

def load_normal_chain(chat_history):
    return chatChain(chat_history)

def load_vectordb(embeddings):
    persistent_client = chromadb.PersistentClient("chroma_db")
    langchain_chroma = Chroma(
        client = persistent_client,
        collection_name="pdfs",
        embedding_function=embeddings,
    )

    return langchain_chroma

def load_pdf_chat_chain(chat_history):
    return pdfChatChain(chat_history)

def load_retrieval_chain(llm, memory, vector_db):
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})  # Ensure proper initialization
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,  # Pass the valid retriever
        return_source_documents=False
    )


class pdfChatChain:
    def __init__(self, chat_history):
        self.memory = create_chat_memory(chat_history, input_key="question")  # For PDF chat
        self.vector_db = load_vectordb(create_embeddings())
        llm = create_llm()
        self.llm_chain = load_retrieval_chain(llm, self.memory, self.vector_db)

    def run(self, inputs):
        inputs = {
            "question": inputs.get("question", ""),
            "chat_history": inputs.get("chat_history", []),
        }
        print(f"Final inputs for pdfChatChain: {inputs}")  # Debug final inputs
        return self.llm_chain.run(inputs)

class chatChain:
    def __init__(self, chat_history):
        self.memory = create_chat_memory(chat_history, input_key="human_input")  # For normal chat
        llm = create_llm()
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = create_llm_chain(llm, chat_prompt, self.memory)

    def run(self, inputs):
        inputs = {
            "human_input": inputs.get("human_input", ""),
            "chat_history": inputs.get("history", []),
        }
        print(f"Final inputs for chatChain: {inputs}")  # Debug final inputs
        return self.llm_chain.run(inputs)
