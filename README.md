# Assistente virtual de portfólio (GenAI & RAG)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=flat&logo=python&logoColor=white)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)]()
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)]()
[![Groq](https://img.shields.io/badge/Groq_API-F55036?style=flat&logo=groq&logoColor=white)]()
[![HuggingFace](https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat&logo=huggingface&logoColor=black)]()

**Versão 1.0** · Aplicação web conversacional que substitui o currículo estático: um agente **RAG** responde sobre trajetória, projetos e habilidades a partir de textos em `base_conhecimento/`, com **LLM de alta performance** e prompt enxuto para reduzir alucinações.

**Deploy:** [portfolio-mateus.streamlit.app](https://portfolio-mateus.streamlit.app/)

## Desafio e solução

Pipeline que ingere documentos, gera embeddings, recupera trechos relevantes e condiciona o modelo ao contexto — interface **Streamlit** e histórico de chat por sessão.

## Arquitetura (resumo)

1. **Base de conhecimento:** arquivos `.txt` em `base_conhecimento/`.
2. **Embeddings:** `intfloat/multilingual-e5-large` (Hugging Face).
3. **Vetores:** **Chroma** recriado **em memória** a cada inicialização do processo (deploy sem estado vetorial obsoleto no disco).
4. **LLM:** **Groq** — `llama-3.3-70b-versatile`, `temperature=0`.
5. **Recuperação:** **MMR** com `k=5` e `fetch_k=15`.
6. **UI:** `app_chat.py` + `@st.cache_resource` na inicialização do motor (evita reindexar a cada interação).

Mais detalhes: [docs/arquitetura.md](docs/arquitetura.md).

## Estrutura do repositório

| Caminho | Função |
|---------|--------|
| `app_chat.py` | Entrada Streamlit (chat + aba de projetos). |
| `deploy_info.py` | Data de referência (America/Sao_Paulo) para o prompt. |
| `motores_ia/` | Produção: `motor_nuvem_groq.py`. Opcional local: `motor_local_llama.py` (Ollama). |
| `base_conhecimento/` | Textos que alimentam o RAG. |
| `docs/` | Documentação técnica (`arquitetura.md`). |
| `.streamlit/` | Tema (`config.toml`). |
| `.devcontainer/` | Ambiente [Dev Container](https://containers.dev/) (Python 3.11) para Codespaces / VS Code. |
| `requirements.txt` | Dependências Python. |
| `.env.example` | Modelo de variáveis para desenvolvimento local. |

## Requisitos

- **Python:** 3.10 ou superior. **Streamlit Community Cloud** e o **Dev Container** deste repo usam **3.11** (recomendado para espelhar produção).
- Contas e chaves: [Groq Console](https://console.groq.com) e [Hugging Face](https://huggingface.co/settings/tokens) para embeddings e APIs.

## Stack

Python · LangChain · Streamlit · Chroma · Groq API · Hugging Face Embeddings · `python-dotenv`

## Como rodar localmente

### 1. Clonar

```bash
git clone https://github.com/MateusBitar/LLM_repo.git
cd LLM_repo
```

### 2. Ambiente virtual

**Windows:** `python -m venv venv` → `venv\Scripts\activate`  
**Linux/macOS:** `python -m venv venv` → `source venv/bin/activate`

### 3. Dependências

```bash
pip install -r requirements.txt
```

### 4. Variáveis de ambiente

Copie `.env.example` para `.env` e preencha:

```env
GROQ_API_KEY=sua_chave_groq
HUGGINGFACEHUB_API_TOKEN=sua_chave_hf
HF_TOKEN=sua_chave_hf
```

`HF_TOKEN` costuma ser o **mesmo valor** do token Hugging Face quando bibliotecas o esperam além de `HUGGINGFACEHUB_API_TOKEN`.

### 5. Executar

```bash
streamlit run app_chat.py
```

Abre em `http://localhost:8501`.

### Deploy (Streamlit Cloud)

Configure os mesmos segredos em **App settings → Secrets** (formato TOML), por exemplo:

```toml
GROQ_API_KEY = "..."
HUGGINGFACEHUB_API_TOKEN = "..."
HF_TOKEN = "..."
```

---

Desenvolvido por **Mateus Bitar** — [LinkedIn](https://linkedin.com/in/mateus-bitar)
