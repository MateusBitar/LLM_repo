from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


chave_groq = os.getenv("GROQ_API_KEY")
chave_hf = os.getenv("HUGGINGFACEHUB_API_TOKEN")



def configurar_motor_nuvem():
    print("☁️ Ligando o Motor de Nuvem: Groq (Llama 3 70B) + HF Embeddings...")
    

    # 1. Carregar documentos
    loader = DirectoryLoader('./base_conhecimento', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    docs = loader.load()

    # 2. Quebrar textos (Tamanho ideal para não cortar os links no final)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

 # 3. Criar Banco Vetorial (Processamento de Embeddings via HuggingFace)
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./banco_vetorial_v2" # <-- MUDE O NOME AQUI
    )
    

    # 4. Buscador e LLM (Groq Llama 3 70B com temperatura ZERO)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)

# ==========================================
    # 5. Configuração do Prompt Multilíngue
    # ==========================================
    system_prompt = (
        "You are Mateus Bitar's official AI assistant. "
        "CRITICAL DIRECTIVE: You MUST mirror the user's language EXACTLY. "
        "If the user prompt is in English (e.g., 'Hi', 'tell me'), you are FORBIDDEN from using Portuguese. Your ENTIRE response MUST be in English. "
        "If the user prompt is in Spanish, reply entirely in Spanish. "
        "STRICT CONTEXT: Base your answer ONLY on the context below. If asked about projects, summarize the projects found in the text (like Assistente de Portfólio and Clipping Jurídico). "
        "Always include their respective URLs. "
        "\n\nContext:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # O filtro vai pegar o objeto feio do Groq e devolver só o texto bonito
    chain = prompt | llm | StrOutputParser()

    return retriever, chain
