# Motores de inferência

- **`motor_nuvem_groq.py`** — Usado em produção pelo `app_chat.py`: Groq (Llama 3.3 70B), embeddings Hugging Face, Chroma em memória.
- **`motor_local_llama.py`** — Alternativa local com Ollama e Chroma persistido em `./banco_vetorial_mateus` (útil para testes sem consumir API Groq).

Variáveis de ambiente típicas: `GROQ_API_KEY`, `HUGGINGFACEHUB_API_TOKEN` (conforme `.env.example` na raiz do repositório).
