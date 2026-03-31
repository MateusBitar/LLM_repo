import streamlit as st
import os
from motores_ia.motor_nuvem_groq import configurar_motor_nuvem

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
    st.image("https://github.com/MateusBitar.png", width=100) # Ícone de perfil
    st.header("Mateus Bitar")
    st.markdown("**Full Stack Developer & Data Science**")
    
    st.divider() # Linha separadora
    
    st.subheader("🔗 Meus Contatos")
    st.markdown("👔 [LinkedIn](https://linkedin.com/in/mateus-bitar)")
    st.markdown("💻 [GitHub](https://github.com/MateusBitar)")
    st.markdown("📧 [E-mail](mailto:mateusrbitar@gmail.com)")
    
 
# 2. Cache e Inicialização da IA 
@st.cache_resource
def inicializar_ia():
    return configurar_motor_nuvem()

retriever, chain = inicializar_ia()

# ==========================================
# 🗂️ CRIAÇÃO DAS ABAS PRINCIPAIS
# ==========================================
aba_chat, aba_repositorios, aba_aplicacoes = st.tabs([
    "💬 Chat com a IA", 
    "💻 Repositórios (Código)", 
    "🌐 Aplicações (Deploys)"
])

# ==========================================
# ABA 1: O CHATBOT PRINCIPAL
# ==========================================
with aba_chat:
    st.title("🤖 Assistente de Portfólio")
    st.markdown("Olá! Sou a IA treinada para falar sobre a experiência, automações e projetos de dados do Mateus. O que você gostaria de saber?")

    # Gerenciamento do Histórico de Chat
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []

    for msg in st.session_state.mensagens:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Caixa de texto e processamento
    if prompt_usuario := st.chat_input("Pergunte sobre as habilidades e projetos do Mateus..."):
        st.session_state.mensagens.append({"role": "user", "content": prompt_usuario})
        with st.chat_message("user"):
            st.markdown(prompt_usuario)
            
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🧠 Consultando a base de dados...")
            
            documentos = retriever.invoke(prompt_usuario)
            textos_juntos = "\n\n".join([doc.page_content for doc in documentos])
            
            resposta = chain.invoke({
                "context": textos_juntos,
                "input": prompt_usuario
            })
            
            placeholder.markdown(resposta)
            st.session_state.mensagens.append({"role": "assistant", "content": resposta})

# ==========================================
# ABA 2: APENAS LINKS DO GITHUB
# ==========================================
with aba_repositorios:
    st.header("💻 Meus Códigos e Repositórios")
    st.write("Acesse o código-fonte dos meus projetos diretamente no GitHub:")
    
    st.markdown("- 📊 **Previsão de Churn (XGBoost):** [Acessar Repositório](https://github.com/MateusBitar/Previsao-de-Churn)")
    st.markdown("- 🧠 **Sistema de Recomendação (SVD):** [Acessar Repositório](https://github.com/MateusBitar/recomendation-system)")
    st.markdown("- 🤖 **Motor IA deste Portfólio (RAG):** [Acessar Repositório](https://github.com/MateusBitar/LLM_repo)")
    st.markdown("- 🎓 **Chatbot Nascentia (TCC em Grupo):** [Acessar Repositório](https://github.com/FelipeYoshidaCEUB/chatbot-ceub)")

# ==========================================
# ABA 3: APENAS LINKS DE APLICAÇÕES WEB
# ==========================================
with aba_aplicacoes:
    st.header("🌐 Projetos em Produção")
    st.write("Teste as aplicações reais rodando em nuvem:")
    
    st.markdown("- 📉 **App: Retenção Inteligente (Churn):** [Testar Aplicação](https://previsao-de-churn.streamlit.app/)")
    
    # Lembre-se de colar a URL pública real que você gerou no Streamlit Cloud na linha abaixo:
    st.markdown("- 🤖 **App: Assistente de Portfólio IA:** [Testar Aplicação](COLOQUE_A_URL_DO_SEU_PORTFOLIO_AQUI)")
