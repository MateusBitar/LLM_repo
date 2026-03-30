
# 🤖 Assistente Virtual de Portfólio (GenAI & RAG)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)]()
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)]()
[![Groq](https://img.shields.io/badge/Groq_API-F55036?style=flat&logo=groq&logoColor=white)]()
[![HuggingFace](https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat&logo=huggingface&logoColor=black)]()

Uma aplicação interativa de Inteligência Artificial Generativa projetada para substituir o formato estático de um currículo tradicional. Este projeto implementa um Agente Autônomo capaz de conversar com recrutadores, responder perguntas sobre minha trajetória profissional e detalhar meus projetos utilizando a arquitetura **RAG (Retrieval-Augmented Generation)**.

## 🎯 O Desafio e a Solução
O objetivo foi criar uma experiência dinâmica onde o usuário pode "entrevistar" meu portfólio. Para isso, desenvolvi uma pipeline de Machine Learning que ingere documentos de texto (minhas experiências, habilidades e projetos acadêmicos/profissionais), converte-os em vetores semânticos e utiliza um LLM de ponta para gerar respostas precisas, blindadas contra alucinações.

## ⚙️ Arquitetura e Engenharia do Projeto
O sistema foi desenhado com foco em modularidade, baixa latência e precisão factual:

1. **Base de Conhecimento (Ingestão):** Arquivos estruturados em `.txt` contendo informações profissionais (ex: experiência como Full Stack na Montezuma e Conde, automações RPA, etc.).
2. **Vetorização (Embeddings):** Utilização do modelo `multilingual-e5-large` da Hugging Face para transformar os textos em embeddings de alta precisão semântica.
3. **Banco de Dados Vetorial:** Implementação do **ChromaDB** para armazenamento persistente e busca rápida do contexto mais relevante (`k=10`).
4. **Motor de Inferência (LLM):** Integração com a API da **Groq** utilizando o modelo **Llama 3.3 70B**, garantindo respostas em milissegundos com capacidade de raciocínio de nível corporativo.
5. **Engenharia de Prompt:** Sistema blindado com `temperature=0.0` e regras estritas de extração de links e formatação, forçando a IA a atuar de forma factual e objetiva.
6. **Interface de Usuário:** Deploy de uma interface web conversacional responsiva utilizando **Streamlit**, com gerenciamento de estado (Session State) para manter a memória do chat.

## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python
* **Orquestração de IA:** LangChain
* **Modelos de Linguagem (LLMs):** Llama 3.3 70B (via Groq API)
* **Embeddings:** Hugging Face API (`intfloat/multilingual-e5-large`)
* **Banco Vetorial:** ChromaDB
* **Front-end / Web App:** Streamlit

## 💻 Como rodar este projeto localmente

### 1. Clone o repositório
```bash
git clone [https://github.com/MateusBitar/LLM_repo.git](https://github.com/MateusBitar/LLM_repo.git)
cd LLM_repo
````

### 2\. Crie e ative o ambiente virtual

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3\. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4\. Configure as Variáveis de Ambiente

Renomeie o arquivo `.env.example` para `.env` e insira suas chaves de API reais (este arquivo será ignorado pelo Git):

```env
GROQ_API_KEY=sua_chave_groq_aqui
HUGGINGFACEHUB_API_TOKEN=sua_chave_hf_aqui
HF_TOKEN=sua_chave_hf_aqui
```

### 5\. Execute a Aplicação

```bash
streamlit run app_chat.py
```

A aplicação estará disponível no seu navegador em `http://localhost:8501`.

-----

*Desenvolvido por **Mateus Bitar** - Conecte-se comigo no [LinkedIn](https://www.google.com/search?q=https://linkedin.com/in/mateus-bitar).*

