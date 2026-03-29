import streamlit as st
# A mágica da arquitetura: Importamos o motor que queremos usar hoje
from motores_ia.motor_local_llama import configurar_motor_local

# 1. Configuração da Página
st.set_page_config(page_title="Assistente do Mateus", page_icon="🤖", layout="centered")
st.title("🤖 Assistente de Portfólio - Mateus Bitar")
st.markdown("Olá! Sou a IA treinada para falar sobre a experiência, automações e projetos de dados do Mateus. O que você gostaria de saber?")

# 2. Cache da IA (Para carregar o motor apenas uma vez e não travar o servidor)
@st.cache_resource
def inicializar_ia():
    # Quando formos para a nuvem com API, mudaremos apenas esta linha!
    return configurar_motor_local()

# Inicializa o cérebro (Puxa as funções lá do arquivo do motor)
retriever, chain = inicializar_ia()

# 3. Gerenciamento do Histórico de Chat
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

# Mostra as mensagens antigas na tela
for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Caixa de texto para o usuário digitar
if prompt_usuario := st.chat_input("Pergunte sobre as habilidades e projetos do Mateus..."):
    # Salva e mostra a pergunta do usuário
    st.session_state.mensagens.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user"):
        st.markdown(prompt_usuario)
        
    # Gera e mostra a resposta da IA
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🧠 Consultando a base de dados...")
        
        # Passo A: O buscador vai na pasta base_conhecimento
        documentos = retriever.invoke(prompt_usuario)
        textos_juntos = "\n\n".join([doc.page_content for doc in documentos])
        
        # Passo B: O Motor gera a resposta em português
        resposta = chain.invoke({
            "context": textos_juntos,
            "input": prompt_usuario
        })
        
        # Atualiza a tela com a resposta final
        placeholder.markdown(resposta)
        st.session_state.mensagens.append({"role": "assistant", "content": resposta})