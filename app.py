import json

import streamlit as st

from assistente import AssistenteEstudos


st.set_page_config(page_title="Assistente Ada", page_icon="🎓", layout="centered")
st.title("🎓 Assistente Ada")
st.caption("Assistente local de estudos sobre Ciência da Computação — RAG + Ollama")


@st.cache_resource
def carregar_assistente() -> AssistenteEstudos:
    return AssistenteEstudos()


try:
    assistente = carregar_assistente()
except Exception as erro:
    st.error(f"Erro ao carregar os modelos: {erro}")
    st.stop()

if "historico" not in st.session_state:
    st.session_state.historico = []

for item in st.session_state.historico:
    with st.chat_message("user"):
        st.write(item["pergunta"])
    with st.chat_message("assistant"):
        st.write(item["resposta"])
        with st.expander("Detalhes da interação"):
            st.code(json.dumps(item["classificacao"], ensure_ascii=False, indent=2), language="json")
            st.write(f"Similaridade: {item['similaridade']:.4f}")
            st.write(f"Tokens da pergunta: {item['tokens_pergunta']}")
            st.write(f"Tokens da resposta: {item['tokens_resposta']}")
            st.write(f"Trecho recuperado: {item['contexto']}")

pergunta = st.chat_input("Faça uma pergunta sobre Ciência da Computação")
if pergunta:
    with st.chat_message("user"):
        st.write(pergunta)
    with st.chat_message("assistant"):
        with st.spinner("Buscando na base e gerando a resposta..."):
            try:
                resultado = assistente.responder(pergunta)
                st.write(resultado["resposta"])
                with st.expander("Detalhes da interação"):
                    st.code(json.dumps(resultado["classificacao"], ensure_ascii=False, indent=2), language="json")
                    st.write(f"Similaridade: {resultado['similaridade']:.4f}")
                    st.write(f"Tokens da pergunta: {resultado['tokens_pergunta']}")
                    st.write(f"Tokens da resposta: {resultado['tokens_resposta']}")
                    st.write(f"Trecho recuperado: {resultado['contexto']}")
                st.session_state.historico.append(resultado)
            except Exception as erro:
                st.error(f"Não foi possível responder. Confirme se o Ollama está aberto e se o modelo phi3 foi baixado. Detalhes: {erro}")
