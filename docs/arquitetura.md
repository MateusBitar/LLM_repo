# Arquitetura do assistente de portfólio

## Objetivo

Agente conversacional baseado em **RAG (Retrieval-Augmented Generation)** que responde sobre carreira, projetos e contatos a partir de arquivos em `base_conhecimento/`, sem depender de currículo estático.

## Stack

| Camada | Tecnologia |
|--------|------------|
| UI / deploy | Streamlit |
| Orquestração | LangChain |
| LLM | Groq API — Llama 3.3 70B (`temperature=0`) |
| Embeddings | Hugging Face — `intfloat/multilingual-e5-large` |
| Vetores | Chroma (em memória a cada inicialização do processo) |
| Configuração local | `python-dotenv` + `.env` (não versionado) |

## Pipeline

1. **Ingestão:** `DirectoryLoader` lê `base_conhecimento/**/*.txt`.
2. **Chunking:** `RecursiveCharacterTextSplitter` (tamanho/sobreposição definidos no motor).
3. **Indexação:** embeddings → coleção Chroma criada na subida do app.
4. **Recuperação:** retriever **MMR** (`fetch_k=15`, `k=5`) para diversificar trechos.
5. **Geração:** prompt de sistema (escopo, idioma, estilo) + contexto recuperado + pergunta.

## Cache no Streamlit

`@st.cache_resource` envolve a inicialização do motor para não reconstruir o índice a cada rerun. Se o **formato do retorno** de `configurar_motor_nuvem()` mudar, incremente `_IA_RESOURCE_VERSION` em `app_chat.py` para invalidar o cache nos deploys.

## Motor local (opcional)

`motores_ia/motor_local_llama.py` — Ollama (Llama 3 + `nomic-embed-text`) e Chroma persistente em disco, para desenvolvimento sem Groq. Não é usado pelo `app_chat.py` em produção.

## Arquivos principais

- `app_chat.py` — aplicação Streamlit.
- `deploy_info.py` — data de referência (Brasília) injetada no prompt.
- `motores_ia/motor_nuvem_groq.py` — RAG + chain de produção.
