import streamlit as st
from supabase import create_client, Client
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Configuración de la interfaz visual
st.set_page_config(page_title="Mi Consultor IA Gratis", page_icon="🤖")
st.title("💬 Chatea con tus Documentos (100% Gratis)")
st.write("Pregúntame lo que quieras. Los datos se buscan en Supabase y responde Llama 3.")

# 2. Conectar con las variables secretas de Streamlit Cloud
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Inicializar los clientes de Supabase y Groq
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Descargar el modelo de embeddings gratuito de Hugging Face en el servidor
# NOTA: Asegúrate de usar el mismo modelo con el que guardaste los vectores en Supabase
@st.cache_resource
def cargar_modelo_embeddings():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedding_model = cargar_modelo_embeddings()

# 3. Crear el historial de chat para mantener la conversación en pantalla
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! ¿Sobre qué documento tienes dudas?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 4. Capturar la pregunta del usuario
if user_query := st.chat_input("Escribe tu pregunta aquí..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # 5. Generar Embedding GRATIS con Hugging Face y buscar en Supabase
    with st.spinner("Buscando fragmentos relevantes..."):
        try:
            # Generar el vector de la pregunta (MiniLM genera 384 dimensiones)
            vector_embedding = embedding_model.encode(user_query).tolist()

            # Llamar a tu función de Supabase (con el ORDER BY corregido del primer error)
            db_response = supabase.rpc(
                "match_documents", 
                {"query_embedding": vector_embedding, "match_threshold": 0.2, "match_count": 3}
            ).execute()

            # Agrupar el texto de los resultados
            documentos_encontrados = ""
            if db_response.data:
                for doc in db_response.data:
                    documentos_encontrados += f"\n- {doc['content']}" # Cambia 'content' por tu columna de texto
            else:
                documentos_encontrados = "No se encontró información relevante en la base de datos."

        except Exception as e:
            st.error(f"Error en la base de datos: {e}")
            documentos_encontrados = "Error al extraer información."

    # 6. Enviar a Groq (Modelo Llama 3 gratuito) para redactar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Pensando respuesta con Llama 3..."):
            instrucciones_sistema = f"""Eres un asistente de IA muy preciso. Responde a la pregunta basándote estrictamente en el siguiente contexto. Si no sabes la respuesta o no está en el texto, di que no la encuentras en los documentos.
            
            Contexto de tus documentos:
            {documentos_encontrados}"""

            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": instrucciones_sistema},
                    {"role": "user", "content": user_query}
                ],
                model="llama3-8b-8192", # Modelo de Meta ultra rápido y gratis en Groq
            )
            
            respuesta_final = chat_completion.choices[0].message.content
            st.write(respuesta_final)
            
            st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
