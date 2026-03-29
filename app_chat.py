import streamlit as st
import os
from motores_ia.motor_local_llama import configurar_motor_local

# 1. Configuração da Página
st.set_page_config(page_title="Assistente do Mateus", page_icon="🤖", layout="wide")

# ==========================================
# 🎨 BARRA LATERAL FIXA (Contatos e Projetos)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100) # Um ícone de perfil genérico
    st.header("Mateus Bitar")
    st.markdown("**Full Stack Developer & Data Science**")
    
    st.divider() # Linha separadora
    
    st.subheader("🔗 Meus Contatos")
    st.markdown("👔 [LinkedIn](www.linkedin.com/in/mateus-bitar)")
    st.markdown("💻 [GitHub](https://github.com/MateusBitar)")
    st.markdown("📧 [E-mail](mailto:mateusrbitar@gmail.com)")
    
    st.divider()
    
    st.subheader("🚀 Acesso Rápido")
    # Lê o arquivo de projetos e plota na tela
    try:
        with open("./base_conhecimento/links_projetos.txt", "r", encoding="utf-8") as file:
            links = file.read()
            st.markdown(links)
    except FileNotFoundError:
        st.caption("Adicione o arquivo links_projetos.txt na base de conhecimento para ver os links aqui.")

# ==========================================
# 🤖 ÁREA PRINCIPAL (O Chat)
# ==========================================
st.title("🤖 Assistente de Portfólio")
st.markdown("Olá! Sou a IA treinada para falar sobre a experiência, automações e projetos de dados do Mateus. O que você gostaria de saber?")

# 2. Cache da IA 
@st.cache_resource
def inicializar_ia():
    return configurar_motor_local()

retriever, chain = inicializar_ia()

# 3. Gerenciamento do Histórico de Chat
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Caixa de texto
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