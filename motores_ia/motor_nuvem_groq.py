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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)

    # 5. Prompt Blindado
    system_prompt = (
        "Você é o assistente virtual do portfólio de Mateus Bitar. "
        "REGRA ABSOLUTA: RESPONDA APENAS USANDO O CONTEXTO FORNECIDO. "
        "Se a resposta não estiver no contexto, diga: 'Desculpe, não tenho essa informação. Confira o LinkedIn do Mateus.' "
        "REGRA DE PROJETOS: Liste sempre o 'Assistente de Portfólio' e o 'Sistema de Clipping Jurídico' quando perguntado sobre projetos atuais. "
        "REGRA DO NASCENTIA: Destaque que foi um TCC em grupo com contribuição ativa do Mateus. "
        "REGRA DE LINKS: SEMPRE inclua as URLs corretas dos projetos citados. Nunca misture os links. "
        "\n\n--- INÍCIO DO CONTEXTO ---\n{context}\n--- FIM DO CONTEXTO ---\n\n"
        "=== LANGUAGE AND TRANSLATION RULES (CRITICAL) ===\n"
        "1. Detect the language of the user's input.\n"
        "2. You MUST translate your final answer to MATCH the user's language EXACTLY.\n"
        "3. If the user asks in English, reply 100% in English.\n"
        "4. Si el usuario pregunta en Español, responde 100% en Español.\n"
        "5. Se o usuário perguntar em Português, responda em Português.\n"
        "NEVER reply in Portuguese if the user asked in English!"
    )
    

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # O filtro vai pegar o objeto feio do Groq e devolver só o texto bonito
    chain = prompt | llm | StrOutputParser()

    return retriever, chain
