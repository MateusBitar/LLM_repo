"""
Aplicação Streamlit do assistente de portfólio (RAG + Groq).

Versão: 1.0
Ponto de entrada recomendado: ``streamlit run app_chat.py``
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st
from groq import RateLimitError

from deploy_info import data_referencia_para_prompt
from motores_ia.motor_nuvem_groq import configurar_motor_nuvem

__version__ = "1.0"

_BASE_CONHECIMENTO = Path(__file__).resolve().parent / "base_conhecimento"

# Sempre anexado ao contexto do LLM: o retriever semântico muitas vezes não puxa estes arquivos.
_ARQUIVOS_LINKS_PROJETO = ("projetos_atuais.txt", "links_projetos.txt")


def _sufixo_links_projetos() -> str:
    partes: list[str] = []
    for nome in _ARQUIVOS_LINKS_PROJETO:
        caminho = _BASE_CONHECIMENTO / nome
        if caminho.is_file():
            partes.append(caminho.read_text(encoding="utf-8"))
    if not partes:
        return ""
    return (
        "\n\n<<< LINKS_E_DEPLOYS_OFICIAIS_PROJETOS — ao mencionar qualquer projeto listado abaixo, "
        "inclua na sua resposta os URLs exatamente como aparecem (GitHub e deploy quando houver) >>>\n\n"
        + "\n\n---\n\n".join(partes)
    )


_MSG_LIMITE_GROQ = (
    "⏳ **O serviço de IA atingiu um limite temporário de uso.**\n\n"
    "Por favor, **aguarde um minuto** e envie sua pergunta de novo. "
    "Se continuar, tente novamente daqui a pouco — não é um problema no seu aparelho."
)


def _invoke_chain_com_retry(chain, payload, placeholder, delays_s=(2, 4, 8)):
    """Reexecuta a cadeia após RateLimitError (429) da Groq com backoff simples."""
    ultimo = len(delays_s)
    for tentativa in range(ultimo + 1):
        try:
            return chain.invoke(payload)
        except RateLimitError:
            if tentativa < ultimo:
                espera = delays_s[tentativa]
                placeholder.markdown(
                    f"⏳ Aguarde um instante; nova tentativa em {espera}s…"
                )
                time.sleep(espera)
            else:
                return _MSG_LIMITE_GROQ
    return _MSG_LIMITE_GROQ


# Streamlit: set_page_config deve ser a primeira chamada da API Streamlit.
st.set_page_config(page_title="Assistente do Mateus", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] img {
            border-radius: 50%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.image("https://github.com/MateusBitar.png", width=150)
    st.header("Mateus Bitar")
    st.markdown("**Full Stack Developer & Data Science**")
    st.divider()
    st.subheader("🔗 Meus Contatos")
    st.markdown("👔 [LinkedIn](https://linkedin.com/in/mateus-bitar)")
    st.markdown("💻 [GitHub](https://github.com/MateusBitar)")
    st.markdown("📧 [E-mail](mailto:mateusrbitar@gmail.com)")
    st.markdown("💬 [WhatsApp](https://wa.me/5561995597474)")
    st.divider()

# Incrementar ao alterar o retorno de configurar_motor_nuvem() para invalidar o cache do Streamlit.
_IA_RESOURCE_VERSION = 7


@st.cache_resource
def inicializar_ia():
    _ = _IA_RESOURCE_VERSION
    return configurar_motor_nuvem()


def obter_ia():
    """
    Inicializa o motor de IA sob demanda e mantém em session_state.

    Evita custo de bootstrap ao abrir somente a aba de projetos.
    """
    if "ia_engine" not in st.session_state:
        with st.spinner("🚀 Inicializando IA (primeiro acesso)..."):
            st.session_state.ia_engine = inicializar_ia()
    return st.session_state.ia_engine

aba_chat, aba_projetos = st.tabs(["💬 Chat com a IA", "🚀 Meus Projetos"])

with aba_chat:
    st.title("🤖 Assistente de Portfólio")
    st.markdown(
        "Olá! Sou a IA treinada para falar sobre a experiência, automações e "
        "projetos de dados do Mateus. O que você gostaria de saber?"
    )
    st.caption(
        "A IA é iniciada no primeiro envio para acelerar a abertura do portfólio."
    )

    container_mensagens = st.container()

    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []

    with container_mensagens:
        for msg in st.session_state.mensagens:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt_usuario := st.chat_input(
        "Pergunte sobre as habilidades e projetos do Mateus..."
    ):
        retriever, chain = obter_ia()
        st.session_state.mensagens.append({"role": "user", "content": prompt_usuario})

        with container_mensagens:
            with st.chat_message("user"):
                st.markdown(prompt_usuario)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("🧠 Consultando a base de dados...")

                documentos = retriever.invoke(prompt_usuario)
                textos_juntos = "\n\n".join(doc.page_content for doc in documentos)
                textos_juntos += _sufixo_links_projetos()

                payload = {
                    "context": textos_juntos,
                    "input": prompt_usuario,
                    "data_referencia": data_referencia_para_prompt(),
                }
                placeholder.markdown("✨ Gerando resposta…")
                resposta = _invoke_chain_com_retry(chain, payload, placeholder)
                placeholder.markdown(resposta)

        st.session_state.mensagens.append({"role": "assistant", "content": resposta})

with aba_projetos:
    st.header("🚀 Portfólio de Projetos")
    st.write(
        "Conheça as soluções que desenvolvi, acesse os códigos-fonte e teste as "
        "aplicações em produção:"
    )
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("🤖 Assistente de Portfólio Inteligente (RAG & GenAI)")
    st.markdown(
        "Aplicação conversacional utilizando LLMs de ponta (Llama 3.3 70B) e "
        "banco de dados vetorial para entrevistas interativas."
    )
    st.markdown("- 💻 [Repositório no GitHub](https://github.com/MateusBitar/LLM_repo)")
    st.markdown("- 🌐 [Testar Aplicação Web](https://portfolio-mateus.streamlit.app/)")
    st.divider()

    st.subheader("📉 Retenção Inteligente (Previsão de Churn com XGBoost)")
    st.markdown(
        "Modelo preditivo de Machine Learning para identificar precocemente clientes "
        "com alto risco de evasão em mercados competitivos."
    )
    st.markdown("- 💻 [Repositório no GitHub](https://github.com/MateusBitar/Previsao-de-Churn)")
    st.markdown("- 🌐 [Testar Aplicação Web](https://previsao-de-churn.streamlit.app/)")
    st.divider()

    st.subheader("🧠 Sistema de Recomendação (SVD)")
    st.markdown(
        "Motor de recomendação personalizado focado em sugerir itens com base no "
        "comportamento histórico do usuário."
    )
    st.markdown("- 💻 [Repositório no GitHub](https://github.com/MateusBitar/recomendation-system)")
    st.divider()

    st.subheader("🎓 Chatbot Nascentia (Projeto Integrador — CEUB)")
    st.markdown(
        "Assistente virtual baseado em RAG (Projeto Integrador em grupo, três semestres no CEUB — "
        "equivalente ao trabalho de conclusão da graduação), integrando LangChain e Hugging Face."
    )
    st.markdown("- 💻 [Repositório no GitHub](https://github.com/FelipeYoshidaCEUB/chatbot-ceub)")
    st.divider()

    st.subheader("⚖️ Sistema de Clipping Jurídico (Legal-tech)")
    st.markdown(
        "Serviço automatizado de varredura, sumarização (NLP) e envio de atualizações "
        "legislativas para escritórios de advocacia."
    )
    st.markdown("- 🚧 *Em desenvolvimento ativo.*")
