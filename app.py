import streamlit as st
from supabase import create_client, Client
from groq import Groq
from sentence_transformers import SentenceTransformer
import pypdf
import io

st.set_page_config(page_title="Asistente IA", page_icon="🤖")
st.title("💬 Consulta con la IA")
st.write("Haz tu pregunta sobre la documentación oficial.")

# Configurar credenciales
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def cargar_modelo_embeddings():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedding_model = cargar_modelo_embeddings()

# =========================================================================
# ESCUCHADOR SECRETO PARA EL STORAGE DE SUPABASE
# =========================================================================
# Si Supabase nos avisa de que subiste un archivo al Storage, este bloque lo procesa en oculto
if "action" in st.query_params and st.query_params["action"] == "procesar_storage":
    try:
        # Obtenemos los detalles del archivo que acabas de subir
        datos_webhook = st.json_request_body() if hasattr(st, 'json_request_body') else {}
        nombre_archivo = datos_webhook.get("record", {}).get("name")
        
        if nombre_archivo and (nombre_archivo.endswith(".pdf") or nombre_archivo.endswith(".md") or nombre_archivo.endswith(".txt")):
            # Descargar el archivo de forma segura desde tu Bucket privado 'conocimiento'
            archivo_bytes = supabase.storage.from_("conocimiento").download(nombre_archivo)
            
            texto_extraido = ""
            if nombre_archivo.endswith(".pdf"):
                lector_pdf = pypdf.PdfReader(io.BytesIO(archivo_bytes))
                for pagina in lector_pdf.pages:
                    texto_extraido += pagina.extract_text() + "\n"
            else:
                texto_extraido = archivo_bytes.decode("utf-8")
                
            if texto_extraido.strip():
                # Cortar en párrafos y vectorizar automáticamente
                fragmentos = [texto_extraido[i:i+1000] for i in range(0, len(texto_extraido), 800)]
                for fragmento in fragmentos:
                    vector = embedding_model.encode(fragmento).tolist()
                    supabase.table("documents").insert({
                        "content": fragmento,
                        "embedding": vector
                    }).execute()
                    
            st.write("OK - Procesado correctamente")
            st.stop()
    except Exception as e:
        st.write(f"Error procesando: {e}")
        st.stop()

# =========================================================================
# EL CHAT PÚBLICO (Lo que ve el usuario en Kodular)
# =========================================================================
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
            vector_embedding = embedding_model.encode(user_query).tolist()
            db_response = supabase.rpc(
                "match_documents", 
                {"query_embedding": vector_embedding, "match_threshold": 0.2, "match_count": 3}
            ).execute()

            documentos_encontrados = ""
            if db_response.data:
                for doc in db_response.data:
                    documentos_encontrados += f"\n- {doc['content']}"
            else:
                documentos_encontrados = "No se encontró información relevante en los manuales."
        except Exception as e:
            st.error(f"Error en la base de datos: {e}")
            documentos_encontrados = "Error al extraer información."

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            instrucciones_sistema = f"Responde basándote estrictamente en este contexto:\n{documentos_encontrados}"
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": instrucciones_sistema}, {"role": "user", "content": user_query}],
                model="openai/gpt-oss-20b",
            )
            respuesta_final = chat_completion.choices.message.content
            st.write(respuesta_final)
            st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
