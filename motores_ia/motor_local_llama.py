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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # 3. Criar Banco Vetorial (É AQUI QUE O PYTHON CRIA A PASTA SOZINHO)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory="./banco_vetorial_mateus"
    )

    # 4. Configurar Buscador e LLM (Temperatura ZERO = Zero Alucinação)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = OllamaLLM(model="llama3", temperature=0.0)

    # 5. Criar o Prompt Blindado (Com Regra de Links)
    system_prompt = (
        "Você é o assistente virtual do portfólio de Mateus Bitar. "
        "REGRA ABSOLUTA: VOCÊ SÓ PODE RESPONDER USANDO O CONTEXTO FORNECIDO ABAIXO. "
        "Se a resposta não estiver no contexto, diga EXATAMENTE: 'Desculpe, não tenho essa informação na minha base de dados atual. Por favor, confira o LinkedIn ou GitHub do Mateus.' "
        "NUNCA INVENTE projetos, tecnologias, diplomas ou anos de experiência. NUNCA deduza informações. "
        "REGRA DE LINKS: Sempre que você citar, explicar ou mencionar um projeto do Mateus, você DEVE obrigatoriamente adicionar ao final da sua resposta a frase: 'Confira mais em: [inserir o link do projeto que está no contexto]'. "
        "RESPONDA SEMPRE EM PORTUGUÊS DO BRASIL. "
        "\n\nContexto recuperado dos documentos:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 6. Criar a Corrente (Chain)
    chain = prompt | llm

    return retriever, chain