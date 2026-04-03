import os
from collections import Counter
from pathlib import Path

import streamlit as st
from groq import RateLimitError

from deploy_info import data_referencia_para_prompt, rotulo_deploy
from motores_ia.motor_nuvem_groq import configurar_motor_nuvem

_REPO_ROOT = Path(__file__).resolve().parent
_BASE_DIR = _REPO_ROOT / "base_conhecimento"

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

    st.divider()
    st.caption(rotulo_deploy())
    st.caption("Compare este SHA com o último commit no GitHub (Manage app → último deploy).")

# 2. Cache e Inicialização da IA (único @st.cache_resource — invalida junto com o processo no redeploy)
@st.cache_resource
def inicializar_ia():
    return configurar_motor_nuvem()

retriever, chain, ingest_metrics = inicializar_ia()

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

    arquivos_na_nuvem = sorted(os.listdir(_BASE_DIR))
    txt_na_pasta = [f for f in arquivos_na_nuvem if f.endswith(".txt")]
    fontes_ingest = ingest_metrics.get("fontes_ingeridas", [])
    faltando_ingest = sorted(set(txt_na_pasta) - set(fontes_ingest))
    extras_ingest = sorted(set(fontes_ingest) - set(txt_na_pasta))

    with st.expander("Diagnóstico de deploy (ingestão vs disco)", expanded=False):
        st.markdown(f"**{rotulo_deploy()}**")
        st.write("Arquivos `.txt` em `base_conhecimento/` (disco):", txt_na_pasta)
        st.write("Fontes carregadas pelo `DirectoryLoader` (ingestão):", fontes_ingest)
        st.write("Chunks por arquivo (ingestão):", ingest_metrics.get("chunks_por_arquivo"))
        st.write(
            "Retriever:",
            ingest_metrics.get("retriever_search_type"),
            f"k={ingest_metrics.get('retriever_k')}, fetch_k={ingest_metrics.get('retriever_fetch_k')}",
        )
        st.write("Diretório absoluto usado pelo motor:", ingest_metrics.get("diretorio_base"))
        if faltando_ingest:
            st.warning(f"Arquivos no disco mas não listados na ingestão: {faltando_ingest}")
        if extras_ingest:
            st.warning(f"Ingestão lista fontes que não batem com listdir (anomalia): {extras_ingest}")

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
                
                # Busca e processamento
                # Busca e processamento
                documentos = retriever.invoke(prompt_usuario)
                textos_juntos = "\n\n".join([doc.page_content for doc in documentos])
                
                # ==========================================
                # 🛠️ MODO DETETIVE: RAIO-X DO QUE A IA LÊ
                # ==========================================
                with st.expander("🕵️‍♂️ Debug: O que a IA está lendo nos bastidores?"):
                    fontes_recuperadas = [
                        os.path.basename(d.metadata.get("source", "?")) for d in documentos
                    ]
                    st.markdown("**0. Rastreio disco → ingestão → recuperação**")
                    st.write("Arquivos `.txt` no disco:", txt_na_pasta)
                    st.write("Chunks por arquivo (ingestão):", ingest_metrics.get("chunks_por_arquivo"))
                    st.write("Fonte de cada chunk recuperado (ordem MMR):", fontes_recuperadas)
                    st.write("Contagem por fonte (recuperação):", dict(Counter(fontes_recuperadas)))
                    st.markdown("**1. O Contexto Puxado do Banco (textos_juntos):**")
                    st.write(textos_juntos)
                    st.markdown("**2. O Comando Final (input):**")
                    st.write(prompt_usuario)
                # ==========================================
                
                try:
                    resposta = chain.invoke(
                        {
                            "context": textos_juntos,
                            "input": prompt_usuario,
                            "data_referencia": data_referencia_para_prompt(),
                        }
                    )
                except RateLimitError:
                    resposta = (
                        "⏳ **O serviço de IA atingiu um limite temporário de uso** "
                        "(muitas pessoas usando o assistente ao mesmo tempo ou muitas perguntas em sequência).\n\n"
                        "Por favor, **aguarde um minuto** e envie sua pergunta de novo. "
                        "Se continuar, tente novamente daqui a pouco — não é um problema no seu aparelho."
                    )
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
