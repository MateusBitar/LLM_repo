from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

print("🧠 Iniciando o cérebro do assistente...")

# 1. Carregar o seu documento de perfil
loader = TextLoader("perfil_mateus.txt", encoding="utf-8")
docs = loader.load()

# 2. Quebrar o texto em pedaços menores (chunks) para a IA processar melhor
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 3. Criar o Banco de Dados Vetorial (A "Memória" com ChromaDB)
# Usando o nomic-embed-text que você baixou no servidor
print("📚 Vetorizando o seu currículo...")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OllamaEmbeddings(model="nomic-embed-text")
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