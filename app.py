import streamlit as st
from supabase import create_client, Client
from groq import Groq
import pypdf
import io

st.set_page_config(page_title="Asistente IA", page_icon="🤖")

# Cargar secretos de las variables de entorno de Streamlit
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

def generar_embedding_groq(texto):
    # Genera embeddings de 1024 dimensiones de forma gratuita y nativa con Groq
    respuesta = groq_client.embeddings.create(
        model="nomic-embed-text",
        input=texto
    )
    return respuesta.data[0].embedding

# =========================================================================
# PROCESADOR AUTOMÁTICO EN SEGUNDO PLANO PARA TU STORAGE
# =========================================================================
try:
    archivos_en_storage = supabase.storage.from_("conocimiento").list()
    
    for arc in archivos_en_storage:
        nombre_archivo = arc.get("name")
        
        if nombre_archivo and nombre_archivo.endswith(".pdf"):
            marca_control = f"Procesado: {nombre_archivo}"
            
            existe = supabase.table("documents").select("id").like("content", f"%{marca_control}%").execute()
            
            if not existe.data:
                archivo_bytes = supabase.storage.from_("conocimiento").download(nombre_archivo)
                lector_pdf = pypdf.PdfReader(io.BytesIO(archivo_bytes))
                texto_extraido = f"--- {marca_control} ---\n"
                for pagina in lector_pdf.pages:
                    texto_extraido += pagina.extract_text() + "\n"
                
                if texto_extraido.strip():
                    fragmentos = [texto_extraido[i:i+1000] for i in range(0, len(texto_extraido), 800)]
                    for fragmento in fragmentos:
                        vector = generar_embedding_groq(fragmento)
                        supabase.table("documents").insert({
                            "content": fragmento,
                            "embedding": vector
                        }).execute()
except Exception as e:
    pass 

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
            vector_embedding = generar_embedding_groq(user_query)
            
            db_response = supabase.rpc(
                "match_documents", 
                {"query_embedding": vector_embedding, "match_threshold": 0.0, "match_count": 3}
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
            instrucciones_sistema = (
                f"Eres un asistente servicial. Responde en español basándote estrictamente en este contexto:\n{documentos_encontrados}"
            )
            
            # Cambiado a mixtral-8x7b-32768 que maneja mejor contextos largos y estructurados en Groq
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": instrucciones_sistema}, {"role": "user", "content": user_query}],
                model="mixtral-8x7b-32768",
            )
            
            respuesta_final = chat_completion.choices.message.content
            st.write(respuesta_final)
            st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
