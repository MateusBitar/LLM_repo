from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def configurar_motor_local():
    print("⚙️ Ligando o Motor Local: Llama 3 (Ollama)...")
    
    # 1. Carregar documentos da pasta
    loader = DirectoryLoader('./base_conhecimento', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    docs = loader.load()

    # 2. Quebrar textos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # 3. Criar/Atualizar Banco Vetorial
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory="./banco_vetorial_mateus"
    )

    # 4. Configurar Buscador e LLM
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model="llama3")

    # 5. Criar o Prompt Blindado (Forçando o Português)
# 5. Criar o Prompt Blindado (Anti-Alucinação e Foco em Resultados)
    system_prompt = (
        "Você é o assistente virtual oficial do portfólio de Mateus Bitar, focado em apoiar a contratação dele por recrutadores e gestores de TI. "
        "REGRA MÁXIMA 1: Use EXCLUSIVAMENTE as informações do contexto abaixo para responder. "
        "REGRA MÁXIMA 2: Se a resposta para a pergunta do usuário NÃO estiver no contexto, NÃO INVENTE NADA. Responda educadamente que você ainda não possui essa informação específica na sua base, mas convide o recrutador a acessar o GitHub ou o LinkedIn do Mateus para conferir mais detalhes. "
        "REGRA MÁXIMA 3: RESPONDA SEMPRE ESTRITAMENTE EM PORTUGUÊS DO BRASIL. NUNCA USE INGLÊS. "
        "Seu tom deve ser profissional, direto, articulado e persuasivo, sem ser bajulador. "
        "Quando perguntarem sobre projetos ou experiências presentes no contexto, disserte de forma estruturada abordando: "
        "1. O desafio ou a demanda inicial. "
        "2. O que o Mateus fez e quais tecnologias utilizou. "
        "3. As entregas, métricas e resultados de negócio alcançados. "
        "\n\nContexto:\n{context}"
    
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 6. Criar a Corrente (Chain)
    chain = prompt | llm

    return retriever, chain