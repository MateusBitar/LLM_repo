from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()


chave_groq = os.getenv("GROQ_API_KEY")
chave_hf = os.getenv("HUGGINGFACEHUB_API_TOKEN")


@st.cache_resource
def configurar_motor_nuvem():
    print("☁️ Ligando o Motor de Nuvem: Groq (Llama 3 70B) + HF Embeddings...")
    

    # 1. Carregar documentos
    loader = DirectoryLoader('./base_conhecimento', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    docs = loader.load()

    # 2. Quebrar textos (Tamanho ideal para não cortar os links no final)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
    splits = text_splitter.split_documents(docs)

 # 3. Criar Banco Vetorial (Processamento de Embeddings via HuggingFace)
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
        
    # BANCO DE DADOS EM RAM
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    # ==========================================
    # 🧠 ALGORITMO MMR (Adeus textos repetidos!)
    # fetch_k=15: Ele lê 15 blocos nos bastidores.
    # k=5: E devolve apenas os 5 MAIS DIFERENTES entre si.
    # ==========================================
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 5, "fetch_k": 15}
    )
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)

# ==========================================
    # 5. Configuração do Prompt Multilíngue
    # ==========================================
    system_prompt = (
        "You are Mateus Bitar's official AI assistant. "
        "Base your answer ONLY on the <context> provided below. "
        "If asked about projects, always list 'Assistente de Portfólio' and 'Clipping Jurídico' using the info from the context, including their URLs.\n\n"
        "<context>\n{context}\n</context>\n\n"
        "<language_rule>\n"
        "1. Identify the language of the user's input.\n"
        "2. You MUST write your ENTIRE final response in that EXACT same language.\n"
        "3. If the user writes in English, reply ONLY in English.\n"
        "4. If the user writes in Spanish, reply ONLY in Spanish.\n"
        "5. If the user writes in Portuguese, reply ONLY in Portuguese.\n"
        "</language_rule>"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question: {input}\n\n[CRITICAL DIRECTIVE: You MUST evaluate the language of the Question above. Your ENTIRE reply MUST be translated to that exact same language. Do NOT use Portuguese if the question is in English.]")
    ])

    # O filtro vai pegar o objeto feio do Groq e devolver só o texto bonito
    chain = prompt | llm | StrOutputParser()

    return retriever, chain
