import time

import streamlit as st
from groq import RateLimitError

from deploy_info import data_referencia_para_prompt
from motores_ia.motor_nuvem_groq import configurar_motor_nuvem

_MSG_LIMITE_GROQ = (
    "⏳ **O serviço de IA atingiu um limite temporário de uso.**\n\n"
    "Por favor, **aguarde um minuto** e envie sua pergunta de novo. "
    "Se continuar, tente novamente daqui a pouco — não é um problema no seu aparelho."
)


def _invoke_chain_com_retry(chain, payload, placeholder, delays_s=(2, 4, 8)):
    """Alguns 429 da Groq são transitórios; tenta de novo com espera crescente."""
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

# 1. Configuração da Página (DEVE SER O PRIMEIRO COMANDO DO STREAMLIT)
st.set_page_config(page_title="Assistente do Mateus", page_icon="🤖", layout="wide")


# Injeta CSS para deixar a imagem da barra lateral redonda
st.markdown("""
    <style>
        [data-testid="stSidebar"] img {
            border-radius: 50%;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🎨 BARRA LATERAL FIXA (Contatos e Projetos)
# ==========================================
with st.sidebar:
    st.image("https://github.com/MateusBitar.png", width=150) # Ícone de perfil
    st.header("Mateus Bitar")
    st.markdown("**Full Stack Developer & Data Science**")
    
    st.divider() # Linha separadora
    
    st.subheader("🔗 Meus Contatos")
    st.markdown("👔 [LinkedIn](https://linkedin.com/in/mateus-bitar)")
    st.markdown("💻 [GitHub](https://github.com/MateusBitar)")
    st.markdown("📧 [E-mail](mailto:mateusrbitar@gmail.com)")
    st.markdown("💬 [WhatsApp](https://wa.me/5561995597474)")

    st.divider()

# 2. Cache e Inicialização da IA (único @st.cache_resource — invalida junto com o processo no redeploy)
@st.cache_resource
def inicializar_ia():
    return configurar_motor_nuvem()

retriever, chain = inicializar_ia()

# ==========================================
# 🗂️ CRIAÇÃO DAS ABAS PRINCIPAIS
# ==========================================
aba_chat, aba_projetos = st.tabs([
    "💬 Chat com a IA", 
    "🚀 Meus Projetos"
])

# ==========================================
# ABA 1: O CHATBOT PRINCIPAL
# ==========================================
with aba_chat:
    st.title("🤖 Assistente de Portfólio")
    st.markdown("Olá! Sou a IA treinada para falar sobre a experiência, automações e projetos de dados do Mateus. O que você gostaria de saber?")

    # ===============================
    # 1. Cria um container invisível exclusivo para as mensagens
    container_mensagens = st.container()

    # Inicializa o histórico se não existir
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []

    # 2. Renderiza o histórico DENTRO do container
    with container_mensagens:
        for msg in st.session_state.mensagens:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 3. Caixa de texto (Fica FORA do container, ou seja, sempre abaixo)
    if prompt_usuario := st.chat_input("Pergunte sobre as habilidades e projetos do Mateus..."):
        
        # Salva a mensagem do usuário na memória
        st.session_state.mensagens.append({"role": "user", "content": prompt_usuario})
        
        # 4. Renderiza as NOVAS mensagens DENTRO do mesmo container para não descerem
        with container_mensagens:
            with st.chat_message("user"):
                st.markdown(prompt_usuario)
                
            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("🧠 Consultando a base de dados...")
                
                documentos = retriever.invoke(prompt_usuario)
                textos_juntos = "\n\n".join([doc.page_content for doc in documentos])

                payload = {
                    "context": textos_juntos,
                    "input": prompt_usuario,
                    "data_referencia": data_referencia_para_prompt(),
                }
                placeholder.markdown("✨ Gerando resposta…")
                resposta = _invoke_chain_com_retry(chain, payload, placeholder)
                placeholder.markdown(resposta)
                
        # Salva a resposta da IA na memória
        st.session_state.mensagens.append({"role": "assistant", "content": resposta})
# ==========================================
# ABA 2: PORTFÓLIO DE PROJETOS
# ==========================================
with aba_projetos:
    st.header("🚀 Portfólio de Projetos")
    st.write("Conheça as soluções que desenvolvi, acesse os códigos-fonte e teste as aplicações em produção:")
    st.markdown("<br>", unsafe_allow_html=True) # Espaçamento extra

    st.subheader("🤖 Assistente de Portfólio Inteligente (RAG & GenAI)")
    st.markdown("Aplicação conversacional utilizando LLMs de ponta (Llama 3.3 70B) e banco de dados vetorial para entrevistas interativas.")
    st.markdown("- 💻 [Repositório no GitHub](https://github.com/MateusBitar/LLM_repo)")
    # Troque a URL abaixo pelo link real do seu Streamlit Cloud
    st.markdown("- 🌐 [Testar Aplicação Web](COLOQUE_A_URL_DO_SEU_PORTFOLIO_AQUI)")
    st.divider()

    st.subheader("📉 Retenção Inteligente (Previsão de Churn com XGBoost)")
    st.markdown("Modelo preditivo de Machine Learning para identificar precocemente clientes com alto risco de evasão em mercados competitivos.")
    st.markdown("- 💻 [Repositório no GitHub](https://github.com/MateusBitar/Previsao-de-Churn)")
    st.markdown("- 🌐 [Testar Aplicação Web](https://previsao-de-churn.streamlit.app/)")
    st.divider()

    st.subheader("🧠 Sistema de Recomendação (SVD)")
    st.markdown("Motor de recomendação personalizado focado em sugerir itens com base no comportamento histórico do usuário.")
    st.markdown("- 💻 [Repositório no GitHub](https://github.com/MateusBitar/recomendation-system)")
    st.divider()

    st.subheader("🎓 Chatbot Nascentia (TCC em Grupo)")
    st.markdown("Assistente virtual baseado em RAG para auxiliar na consulta de informações acadêmicas e institucionais, integrando LangChain e Hugging Face.")
    st.markdown("- 💻 [Repositório no GitHub](https://github.com/FelipeYoshidaCEUB/chatbot-ceub)")
    st.divider()

    st.subheader("⚖️ Sistema de Clipping Jurídico (Legal-tech)")
    st.markdown("Serviço automatizado de varredura, sumarização (NLP) e envio de atualizações legislativas para escritórios de advocacia.")
    st.markdown("- 🚧 *Em desenvolvimento ativo.*")
