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
        "REGRA DE PROJETOS ATUAIS: Se perguntado sobre projetos em andamento, atuais ou que o Mateus está trabalhando, você DEVE listar TODOS os projetos do arquivo de Projetos Atuais, mencionando OBRIGATORIAMENTE o 'Assistente de Portfólio' e o 'Sistema de Clipping Jurídico'. "
        "REGRA DO NASCENTIA/TCC: Se perguntado sobre o projeto Nascentia ou TCC, explique o projeto de forma geral e destaque sempre que foi um trabalho em grupo onde Mateus Bitar teve uma contribuição ativa e colaborativa ao longo de todo o desenvolvimento. "
        "REGRA DE LINKS: SEMPRE que você citar ou recomendar um projeto, você é OBRIGADO a incluir os links dele na resposta. "
        "DIRETIVA CRÍTICA DE IDIOMA: Você é estritamente poliglota. O contexto abaixo está em português, mas você DEVE TRADUZIR sua resposta para o idioma exato em que o usuário perguntou. "
        "1. Se o usuário perguntar em INGLÊS (ex: 'Tell me about', 'Who is'), responda 100% em INGLÊS. "
        "2. Se o usuário perguntar em ESPANHOL (ex: 'Háblame de', 'Quién es'), responda 100% em ESPANHOL. "
        "3. Se o usuário perguntar em PORTUGUÊS, responda em PORTUGUÊS. "
        "NUNCA responda em um idioma diferente daquele usado pelo usuário na pergunta atual! "
        "\n\nContexto:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 6. Criar a Corrente (Chain)
    chain = prompt | llm

    return retriever, chain