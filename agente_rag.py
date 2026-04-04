"""
Script de demonstração: pipeline RAG mínimo com Ollama + Chroma (sem Streamlit).

Execução: ``python agente_rag.py`` (a partir da raiz do repositório).

Versão da aplicação: 1.0
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main() -> None:
    loader = DirectoryLoader(
        "./base_conhecimento",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory="./banco_vetorial_mateus",
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model="llama3")

    system_prompt = (
        "Você é o assistente virtual oficial do portfólio de Mateus Bitar. "
        "Use APENAS as informações do contexto abaixo para responder as perguntas. "
        "Seja extremamente profissional, entusiasmado e tente 'vender' as habilidades "
        "do Mateus para os recrutadores.\n\nContexto:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm

    pergunta = (
        "Me fale sobre as habilidades técnicas do Mateus e me dê um exemplo de "
        "projeto que ele fez."
    )
    documentos = retriever.invoke(pergunta)
    textos_juntos = "\n\n".join(doc.page_content for doc in documentos)
    resposta = chain.invoke({"context": textos_juntos, "input": pergunta})
    print(resposta)


if __name__ == "__main__":
    main()
