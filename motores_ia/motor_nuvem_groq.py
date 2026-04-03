from collections import Counter
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

chave_groq = os.getenv("GROQ_API_KEY")
chave_hf = os.getenv("HUGGINGFACEHUB_API_TOKEN")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BASE_CONHECIMENTO = _REPO_ROOT / "base_conhecimento"


def configurar_motor_nuvem():
    print("☁️ Ligando o Motor de Nuvem: Groq (Llama 3 70B) + HF Embeddings...")

    loader = DirectoryLoader(
        str(_BASE_CONHECIMENTO),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    sources_loaded = sorted(
        {os.path.basename(d.metadata.get("source", "")) for d in docs if d.metadata.get("source")}
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
    splits = text_splitter.split_documents(docs)

    chunk_counts: Counter[str] = Counter()
    for s in splits:
        chunk_counts[os.path.basename(s.metadata.get("source", "?"))] += 1

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15},
    )

    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)

    system_prompt = (
        "You are Mateus Bitar's official AI assistant for his interactive portfolio. "
        "Base answers about Mateus ONLY on the <context> below. "
        "Never invent employers, grades, facts, or URLs about him. Use links exactly as they appear in the context when discussing his work.\n\n"
        "<scope>\n"
        "IN SCOPE: questions about Mateus Bitar, his professional profile, career, education, skills, projects, "
        "this portfolio assistant, how to contact him, or anything clearly answerable from the context.\n"
        "OUT OF SCOPE: general knowledge, other people (e.g. celebrities, CEOs), unrelated trivia, homework, "
        "or any topic not tied to Mateus or the portfolio purpose—even if you know the answer from training.\n"
        "If the user's message is OUT OF SCOPE, do NOT answer their factual question. Do NOT use general knowledge. "
        "Reply ONLY with a short professional deflection in the user's language, then the contact block below verbatim "
        "(translate the deflection sentence if needed; keep URLs and email/phone exactly as shown):\n"
        "\"I can only help with information about Mateus Bitar's career, projects, and professional background. "
        "Your question is outside that scope. For anything else, please contact Mateus directly:\"\n"
        "LinkedIn: https://linkedin.com/in/mateus-bitar\n"
        "GitHub: https://github.com/MateusBitar\n"
        "E-mail: mateusrbitar@gmail.com\n"
        "WhatsApp: +55 (61) 99559-7474\n"
        "Brief greetings (e.g. hi, thanks) may get a one-line polite reply inviting questions about Mateus—no full off-topic answers.\n"
        "</scope>\n\n"
        "<projects_rule>\n"
        "When the question is about projects, portfolio, skills demonstrated through work, or similar broad topics: "
        "draw from the VARIETY of projects in the context (e.g. churn prediction, recommendation SVD, Nascentia chatbot, "
        "portfolio assistant, legal clipping). Prefer a balanced mix—not every answer should center on the same two items.\n"
        "Mention Assistente de Portfólio and/or Clipping Jurídico when they are directly relevant "
        "(e.g. current focus, legal-tech, this RAG app, or when the user names them).\n"
        "Do not append unrelated project names or URLs at the end when the question is narrow or unrelated "
        "(e.g. a specific course grade, yes/no about a company, or a purely technical stack question that the context answers without needing extra plugs).\n"
        "</projects_rule>\n\n"
        "<status_rule>\n"
        "If materials describe a project as in development, WIP, or not yet in production, say that clearly when asked about production status. "
        "Do not guess beyond what is written.\n"
        "</status_rule>\n\n"
        "<context>\n{context}\n</context>\n\n"
        "<language>\n"
        "Answer in the same language as the user's message. "
        "Start directly with the answer—do not announce which language you will use.\n"
        "</language>\n\n"
        "<style>\n"
        "Sound like a professional human assistant at a portfolio or interview, not like a system narrating documents. "
        "Never use meta-phrases such as: 'according to the context', 'de acordo com o contexto', "
        "'based on the provided information', 'com base nas informações fornecidas', "
        "'the context does not say', 'o contexto não informa', or any equivalent. "
        "Do not expose RAG or internal limitations. "
        "If a detail is missing, answer with what is known in one natural sentence and add briefly that "
        "you don't have that specific detail (without mentioning documents or context).\n"
        "</style>"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    ingest_metrics = {
        "chunks_por_arquivo": dict(chunk_counts),
        "fontes_ingeridas": sources_loaded,
        "total_chunks": len(splits),
        "total_documentos_brutos": len(docs),
        "retriever_search_type": "mmr",
        "retriever_k": 5,
        "retriever_fetch_k": 15,
        "diretorio_base": str(_BASE_CONHECIMENTO),
    }

    return retriever, chain, ingest_metrics
