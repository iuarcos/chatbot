import streamlit as st
from supabase import create_client, Client
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Interfaz visual ultra-limpia para el móvil (Kodular)
st.set_page_config(page_title="Asistente IA", page_icon="🤖")
st.title("💬 Consulta con la IA")
st.write("Haz tu pregunta sobre la documentación oficial.")

# 2. Conectar con las variables secretas de Streamlit Cloud
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Inicializar clientes
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def cargar_modelo_embeddings():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedding_model = cargar_modelo_embeddings()

# 3. Mantener el historial del chat en la pantalla
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! ¿En qué puedo ayudarte hoy?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 4. Capturar la pregunta del usuario desde el móvil
if user_query := st.chat_input("Escribe tu pregunta aquí..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # 5. Buscar en la base de datos de Supabase usando la función RPC
    with st.spinner("Buscando respuestas..."):
        try:
            vector_embedding = embedding_model.encode(user_query).tolist()
            
            # Ejecuta la función match_documents en Supabase
            db_response = supabase.rpc(
                "match_documents", 
                {"query_embedding": vector_embedding, "match_threshold": 0.2, "match_count": 3}
            ).execute()

            documentos_encontrados = ""
            if db_response.data:
                for doc in db_response.data:
                    documentos_encontrados += f"\n- {doc['content']}"
            else:
                documentos_encontrados = "No se encontró información relevante en los documentos."
        except Exception as e:
            st.error(f"Error en la base de datos: {e}")
            documentos_encontrados = "Error al extraer información."

    # 6. Generar respuesta con Llama 3 en Groq
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            instrucciones_sistema = f"""Eres un asistente de IA experto. Responde a la pregunta basándote estrictamente en el siguiente contexto. Si no lo sabes o no está en el texto, di que no dispones de esa información en los manuales actuales.
            
            Contexto:
            {documentos_encontrados}"""

            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": instrucciones_sistema}, {"role": "user", "content": user_query}],
                model="llama3-8b-8192",
            )
            respuesta_final = chat_completion.choices.message.content
            st.write(respuesta_final)
            st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
