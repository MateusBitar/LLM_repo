"""
Motor RAG opcional com Ollama (embeddings + Llama 3) e Chroma persistente.

Uso típico: desenvolvimento local quando não se deseja consumir API Groq.
Requer Ollama em execução com os modelos configurados abaixo.

Versão da aplicação: 1.0
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

def configurar_motor_local():
    """
    Carrega ``base_conhecimento``, persiste vetores em ``./banco_vetorial_mateus``
    e retorna retriever + chain com prompt restrito ao contexto.
    """
    loader = DirectoryLoader(
        "./base_conhecimento",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=50,
    )
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory="./banco_vetorial_mateus",
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    llm = OllamaLLM(model="llama3", temperature=0.0)

    system_prompt = (
        "Você é o assistente virtual do portfólio de Mateus Bitar. "
        "REGRA ABSOLUTA: RESPONDA APENAS USANDO O CONTEXTO FORNECIDO. "
        "Se a resposta não estiver no contexto, diga: 'Desculpe, não tenho essa informação. "
        "Confira o LinkedIn do Mateus.' "
        "REGRA DE PROJETOS: Liste sempre o 'Assistente de Portfólio' e o "
        "'Sistema de Clipping Jurídico' quando perguntado sobre projetos atuais. "
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

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm
    return retriever, chain
