# Contexto do Projeto: Assistente de Portfólio IA (Mateus Bitar)

## 1. Visão Geral do Projeto
- **Objetivo:** Criar um Agente de Inteligência Artificial autônomo baseado em RAG para atuar como assistente de portfólio interativo, respondendo a perguntas de recrutadores sobre as experiências, habilidades e projetos de Mateus Bitar.
- **Stack Tecnológico:** Python 3.11, Streamlit (Front-end/Deploy), LangChain (Orquestração), Groq API / Llama 3.3 70B (LLM), Hugging Face Embeddings (intfloat/multilingual-e5-large), ChromaDB (Banco Vetorial em RAM).
- **Status do Deploy:** Hospedado no Streamlit Cloud, conectado diretamente ao repositório do GitHub.

## 2. O que já foi feito (Histórico de Refatoração)
Nós passamos por um intenso processo de estabilização e troubleshooting de arquitetura:
1. **Estabilização de Ambiente:** Resolvemos erros de compilação de C-extensions e Protobuf travando a versão do Python para 3.11 no Streamlit Cloud.
2. **Quebra de Ancoragem de Idioma:** O modelo (Llama 3.3) sofria "Recency Bias", respondendo em português mesmo quando perguntado em inglês, devido ao peso dos documentos `.txt` ingeridos no RAG. Refatoramos o prompt com "Diretivas Críticas" e redundância de regras na tag `human`.
3. **Refatoração do ChromaDB (In-Memory):** O Streamlit recarregava o código a cada interação, causando duplicação de dados no `persist_directory`. Migramos o banco para a Memória RAM, criando-o do zero a cada inicialização para evitar clones.
4. **Proteção de Execução (`@st.cache_resource`):** Adicionamos cache na função de inicialização do motor para impedir recarregamentos e vazamento de memória.
5. **Algoritmo MMR no Retriever:** Alteramos a busca vetorial para usar *Maximal Marginal Relevance* (`fetch_k=15`, `k=5`) para garantir diversidade nos chunks recuperados e evitar que textos repetidos ocupem toda a janela de contexto.

## 3. O Problema Atual (A Incongruência Nuvem vs. Repositório)
Apesar do repositório no GitHub estar com o código 100% atualizado (com MMR, banco in-memory e prompt blindado) e os arquivos de texto estarem corretos (incluindo o projeto "Sistema de Clipping Jurídico"), **a aplicação em produção continua se comportando como a versão antiga**.
- **Sintomas Atuais na Nuvem:**
  - O sistema ignora as regras de idioma e responde em português para perguntas em inglês.
  - O log de "Debug" revela que a IA **não está** recebendo o texto do "Sistema de Clipping Jurídico".
  - O log mostra que os blocos de texto no retriever continuam vindo **duplicados** (provando que o MMR não está sendo executado).

## 4. Principais Suspeitas
1. **Invalidação de Cache Hostil:** A tag `@st.cache_resource` no Streamlit Cloud "congelou" a instância da função. O servidor puxou o código novo do GitHub, mas continua rodando a versão antiga que está na memória RAM.
2. **Atraso de Sincronização do Streamlit Cloud:** O webhook do GitHub pode não ter engatilhado o rebuild automático da aplicação no servidor da nuvem.
3. **Ghost Files / Arquivos Ocultos:** A possibilidade de haver arquivos `.txt` antigos escondidos no servidor que o `DirectoryLoader` ainda está puxando.

## 5. Arquivos Críticos para Análise
- `motores_ia/motor_nuvem_groq.py` (Lógica do RAG, Embeddings e LLM).
- `app_chat.py` (Front-end Streamlit e injeção do prompt do usuário).
- `base_conhecimento/*.txt` (Arquivos de contexto que alimentam o RAG).