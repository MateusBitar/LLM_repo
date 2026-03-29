from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import os

print("🧠 Iniciando o cérebro do assistente...")

# 1. Carregar documentos da pasta
loader = DirectoryLoader('./base_conhecimento', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
docs = loader.load()

# 2. Quebrar textos
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 3. Criar Banco Vetorial
print("📚 Lendo a pasta e vetorizando o seu histórico...")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    persist_directory="./banco_vetorial_mateus"
)

# 4. Configurar Buscador e Llama 3
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = OllamaLLM(model="llama3")

# 5. Criar o Prompt (A personalidade do Agente)
system_prompt = (
    "Você é o assistente virtual oficial do portfólio de Mateus Bitar. "
    "Use APENAS as informações do contexto abaixo para responder as perguntas. "
    "Seja extremamente profissional, entusiasmado e tente 'vender' as habilidades do Mateus para os recrutadores. "
    "\n\nContexto:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 6. O Bypass (Criando a Corrente de forma direta)
chain = prompt | llm

print("🤖 Assistente pronto! Fazendo a primeira pergunta...\n")
print("-" * 50)

pergunta = "Me fale sobre as habilidades técnicas do Mateus e me dê um exemplo de projeto que ele fez."
print(f"👤 Recrutador: {pergunta}")

# Passo A: O buscador vai na sua pasta e pega os trechos de texto mais relevantes
documentos_encontrados = retriever.invoke(pergunta)
textos_juntos = "\n\n".join([doc.page_content for doc in documentos_encontrados])

# Passo B: Mandamos o contexto real e a pergunta direto para o modelo responder
resposta = chain.invoke({
    "context": textos_juntos,
    "input": pergunta
})

print(f"\n🤖 Agente Portfólio:\n{resposta}")
print("-" * 50)