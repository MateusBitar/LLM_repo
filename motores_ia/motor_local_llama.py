from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import os

def configurar_motor_local():
    print("⚙️ Ligando o Motor Local: Llama 3 (Ollama)...")
    
    # 1. Carregar documentos da pasta (Verifique se a pasta base_conhecimento existe!)
    loader = DirectoryLoader('./base_conhecimento', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    docs = loader.load()

    # Se não achar nenhum documento, avisa no terminal
    if not docs:
        print("⚠️ ALERTA: Nenhum arquivo .txt encontrado na pasta 'base_conhecimento'!")

    # 2. Quebrar textos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # 3. Criar Banco Vetorial (É AQUI QUE O PYTHON CRIA A PASTA SOZINHO)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory="./banco_vetorial_mateus"
    )

    # 4. Configurar Buscador e LLM (Temperatura ZERO = Zero Alucinação)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    llm = OllamaLLM(model="llama3", temperature=0.0)

       # 5. Criar o Prompt Blindado (Com Extração Inteligente de Links)
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

    # 6. Criar a Corrente (Chain)
    chain = prompt | llm

    return retriever, chain