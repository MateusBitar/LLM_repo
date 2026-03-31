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
        persist_directory="./banco_vetorial_mateus_nuvem" # Nome novo para evitar conflito
    )

    # 4. Buscador e LLM (Groq Llama 3 70B com temperatura ZERO)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)

    # 5. Prompt Blindado
    system_prompt = (
        "Você é o assistente virtual do portfólio de Mateus Bitar. "
        "REGRA ABSOLUTA: RESPONDA APENAS USANDO O CONTEXTO FORNECIDO. "
        "Se a resposta não estiver no contexto, diga: 'Desculpe, não tenho essa informação. Confira o LinkedIn do Mateus.' "
        "REGRA DE PROJETOS ATUAIS: Se perguntado sobre projetos em andamento, atuais ou que o Mateus está trabalhando, você DEVE listar TODOS os projetos do arquivo de Projetos Atuais, mencionando OBRIGATORIAMENTE o 'Assistente de Portfólio' e o 'Sistema de Clipping Jurídico'. "
        "REGRA DO NASCENTIA/TCC: Se perguntado sobre o projeto Nascentia ou TCC, explique o projeto de forma geral e destaque sempre que foi um trabalho em grupo onde Mateus Bitar teve uma contribuição ativa e colaborativa ao longo de todo o desenvolvimento. "
        "REGRA DE LINKS: SEMPRE que você citar ou recomendar um projeto, você é OBRIGADO a incluir os links dele na resposta. Procure os links no cabeçalho ou rodapé do projeto no contexto. MUITA ATENÇÃO: Nunca misture os links; certifique-se de que a URL pertence exatamente ao projeto que você está citando. "
        "RESPONDA SEMPRE EM PORTUGUÊS DO BRASIL de forma profissional e objetiva. "
        "\n\nContexto:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # O filtro vai pegar o objeto feio do Groq e devolver só o texto bonito
    chain = prompt | llm | StrOutputParser()

    return retriever, chain