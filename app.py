import streamlit as st
from supabase import create_client, Client
from groq import Groq
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Asistente IA", page_icon="🤖")

# Cargar secretos de las variables de entorno de Streamlit
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def cargar_modelo_embeddings():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = cargar_modelo_embeddings()

# =========================================================================
# EL CHAT PÚBLICO (Lo que ve el usuario en Kodular)
# =========================================================================
st.title("💬 Consulta con la IA")
st.write("Haz tu pregunta sobre la documentación oficial.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! ¿En qué puedo ayudarte hoy?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_query := st.chat_input("Escribe tu pregunta aquí..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.spinner("Buscando respuestas..."):
        try:
            # Codificación a 384 dimensiones
            vector_embedding = embedding_model.encode(user_query).tolist()
            
            db_response = supabase.rpc(
                "match_documents", 
                {"query_embedding": vector_embedding, "match_threshold": 0.0, "match_count": 3}
            ).execute()

            documentos_encontrados = ""
            if db_response.data:
                for doc in db_response.data:
                    documentos_encontrados += f"\n- {doc['content']}"
            else:
                documentos_encontrados = "Información general sobre normativas municipales."
        except Exception as e:
            documentos_encontrados = "Información base de reglamentos."

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            instrucciones_sistema = (
                f"Eres un asistente servicial que responde en español. Usa este contexto si es útil:\n{documentos_encontrados}"
            )
            
            # Usando el modelo de producción estable de Groq
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": instrucciones_sistema}, {"role": "user", "content": user_query}],
                model="llama-3.3-70b-versatile",
            )
            
            respuesta_final = chat_completion.choices.message.content
            st.write(respuesta_final)
            st.session_state.messages.append({"role": "assistant", "content": respuesta_final})

