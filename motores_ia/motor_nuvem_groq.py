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
        "You are Mateus Bitar's official AI assistant. "
        "Base your answer ONLY on the <context> provided below. "
        "If asked about projects, always list 'Assistente de Portfólio' and 'Clipping Jurídico' using the info from the context, including their URLs.\n\n"
        "<context>\n{context}\n</context>\n\n"
        "<language_rule>\n"
        "1. Identify the language of the user's input.\n"
        "2. You MUST write your ENTIRE final response in that EXACT same language.\n"
        "3. If the user writes in English, reply ONLY in English.\n"
        "4. If the user writes in Spanish, reply ONLY in Spanish.\n"
        "5. If the user writes in Portuguese, reply ONLY in Portuguese.\n"
        "</language_rule>"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Question: {input}\n\n[CRITICAL DIRECTIVE: You MUST evaluate the language of the Question above. Your ENTIRE reply MUST be translated to that exact same language. Do NOT use Portuguese if the question is in English.]",
            ),
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
