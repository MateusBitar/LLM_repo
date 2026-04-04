
# Assistente virtual de portfólio (GenAI & RAG)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)]()
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)]()
[![Groq](https://img.shields.io/badge/Groq_API-F55036?style=flat&logo=groq&logoColor=white)]()
[![HuggingFace](https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat&logo=huggingface&logoColor=black)]()

Aplicação web conversacional que substitui o currículo estático: um agente baseado em **RAG** responde sobre trajetória, projetos e habilidades a partir de textos versionados em `base_conhecimento/`, com **LLM de alta performance** e regras de prompt para reduzir alucinações.

**Deploy público:** [portfolio-mateus.streamlit.app](https://portfolio-mateus.streamlit.app/)

## Desafio e solução

Pipeline que ingere documentos, gera embeddings, recupera trechos relevantes e condiciona o modelo ao contexto recuperado — com interface **Streamlit** e memória de chat por sessão.

## Arquitetura (resumo)

1. **Base de conhecimento:** arquivos `.txt` em `base_conhecimento/`.
2. **Embeddings:** `intfloat/multilingual-e5-large` (Hugging Face).
3. **Vetores:** **Chroma** recriado em memória a cada inicialização do processo (evita estado obsoleto no deploy).
4. **LLM:** **Groq** — Llama 3.3 70B, `temperature=0`.
5. **Recuperação:** **MMR** com `k=5` e `fetch_k=15` para diversificar os trechos enviados ao prompt.
6. **UI:** Streamlit (`app_chat.py`), com `@st.cache_resource` na inicialização do motor.

Detalhes e decisões de engenharia: [docs/arquitetura.md](docs/arquitetura.md).

## Estrutura do repositório

| Caminho | Função |
|---------|--------|
| `app_chat.py` | Entrada Streamlit (chat + aba de projetos). |
| `deploy_info.py` | Data de referência (fuso Brasília) para o prompt. |
| `motores_ia/` | Motor em nuvem (`motor_nuvem_groq.py`) e motor local opcional (`motor_local_llama.py`). |
| `base_conhecimento/` | Textos que alimentam o RAG. |
| `docs/` | Documentação técnica. |
| `.streamlit/` | Tema da aplicação. |

## Tecnologias

- Python, LangChain, Streamlit, Chroma, Groq API, Hugging Face Embeddings.

## Como rodar localmente

### 1. Clonar

```bash
git clone https://github.com/MateusBitar/LLM_repo.git
cd LLM_repo
```

### 2. Ambiente virtual

**Windows:** `python -m venv venv` e `venv\Scripts\activate`  
**Linux/macOS:** `python -m venv venv` e `source venv/bin/activate`

### 3. Dependências

```bash
pip install -r requirements.txt
```

### 4. Variáveis de ambiente

Copie `.env.example` para `.env` e preencha as chaves:

```env
GROQ_API_KEY=sua_chave_groq
HUGGINGFACEHUB_API_TOKEN=sua_chave_hf
HF_TOKEN=sua_chave_hf
```

### 5. Executar

```bash
streamlit run app_chat.py
```

Abre em `http://localhost:8501`.

---

Desenvolvido por **Mateus Bitar** — [LinkedIn](https://linkedin.com/in/mateus-bitar).
