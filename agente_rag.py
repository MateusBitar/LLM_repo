from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

print("🧠 Iniciando o cérebro do assistente...")

# 1. Carregar TODOS os arquivos de texto da pasta base_conhecimento
# Usamos o TextLoader por baixo dos panos para garantir que ele leia os acentos em pt-br (utf-8)
loader = DirectoryLoader('./base_conhecimento', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
docs = loader.load()

# 2. Quebrar os textos em pedaços menores (chunks)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 3. Criar e SALVAR o Banco de Dados Vetorial no disco
print("📚 Lendo a pasta e vetorizando o seu currículo...")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    persist_directory="./banco_vetorial_mateus" # <-- Isso salva a memória fisicamente no servidor!
)


# 4. Configurar o "Buscador" e o Llama 3
retriever = vectorstore.as_retriever()
llm = OllamaLLM(model="llama3")

# 5. Criar o Prompt (A personalidade do seu Agente)
system_prompt = (
    "Você é o assistente virtual oficial do portfólio de Mateus Bitar. "
    "Use APENAS as informações do contexto abaixo para responder as perguntas. "
    "Seja extremamente profissional, entusiasmado e tente 'vender' as habilidades do Mateus para os recrutadores. "
    "Se perguntarem algo que não está no contexto, diga gentilmente que você foca apenas em falar sobre a carreira de desenvolvedor e dados do Mateus. "
    "\n\n"
    "Contexto:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 6. Juntar tudo em uma Chain (Corrente de execução)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 7. O Teste de Fogo!
print("🤖 Assistente pronto! Fazendo a primeira pergunta...\n")
print("-" * 50)

pergunta = "Me fale sobre as habilidades técnicas do Mateus e me dê um exemplo de projeto que ele fez."
print(f"👤 Recrutador: {pergunta}")

resposta = rag_chain.invoke({"input": pergunta})

print(f"🤖 Agente Portfólio: {resposta['answer']}")
print("-" * 50)