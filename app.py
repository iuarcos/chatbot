
import streamlit as st
from supabase import create_client
from groq import Groq
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Asistente IUARCOS",
    page_icon="📚",
    layout="centered"
)

TXT_FILE = "PROGRAMA_IUARCOS_260523.txt"
BUCKET = "Bd_conocimiento"

try:
    supabase = create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )
    supabase_admin = create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
    )
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    st.error("No se pudieron conectar los servicios. Revisa la configuración.")
    st.stop()


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


try:
    embedding_model = load_embedding_model()
except Exception:
    st.error("No se pudo cargar el modelo de búsqueda.")
    st.stop()


def split_text(text, chunk_size=1000, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if len(chunk) >= 100:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def index_document():
    from io import BytesIO

    with st.spinner("Preparando el documento..."):
        file_bytes = supabase.storage.from_(BUCKET).download(TXT_FILE)
        text = file_bytes.decode("utf-8-sig", errors="replace").strip()

        if not text:
            raise ValueError("El archivo de texto está vacío.")

        chunks = split_text(text)

        if not chunks:
            raise ValueError("No se pudo dividir el texto en fragmentos útiles.")

        embeddings = embedding_model.encode(
            chunks,
            normalize_embeddings=True
        ).tolist()

        # Elimina únicamente los fragmentos anteriores de este mismo TXT.
        supabase_admin.table("documents").delete().filter(
            "metadata->>source", "eq", TXT_FILE
        ).execute()

        records = [
            {
                "content": chunk,
                "metadata": {
                    "source": TXT_FILE,
                    "chunk_index": index
                },
                "embedding": embedding
            }
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        for offset in range(0, len(records), 50):
            supabase_admin.table("documents").insert(
                records[offset:offset + 50]
            ).execute()

    return len(chunks)


st.title("Asistente IUARCOS")
st.write("Consulta tus dudas sobre la documentación de IUARCOS.")

with st.expander("Preparación interna del documento"):
    if st.button("Indexar documento TXT"):
        try: 
            total = index_document()
            st.success(
                f"Documento indexado correctamente: {total} fragmentos."
            )
        except Exception as e:
            st.error(
                f"Error al indexar: {type(e).__name__}: {e}"
            )

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "¡Hola! Puedes preguntarme sobre la documentación de IUARCOS."
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_query = st.chat_input("Escribe tu pregunta...")

if user_query:
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Consultando la documentación..."):
            try:
                query_embedding = embedding_model.encode(
                    user_query,
                    normalize_embeddings=True
                ).tolist()

                result = supabase.rpc(
                    "match_documents",
                    {
                        "query_embedding": query_embedding,
                        "match_threshold": 0.0,
                        "match_count": 10
                    }
                ).execute()

                documents = result.data or []
                context = "\n\n".join(
                    doc["content"]
                    for doc in documents
                    if doc.get("content")
                )

                if not context.strip():
                    answer = (
                        "No he encontrado información en la documentación "
                        "para responder a esa pregunta."
                    )
                else:
                    completion = groq_client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Eres un asistente virtual que ayuda a los ciudadanos a consultar "
                                    "y comprender información relacionada con IUARCOS y el Ayuntamiento "
                                    "de Arcos de la Frontera.\n\n"

                                    "Responde siempre en español, con un tono natural, cercano, claro "
                                    "y preciso. Tu objetivo es resolver la pregunta del usuario de la "
                                    "forma más útil y directa posible.\n\n"

                                    "Utiliza la documentación proporcionada como fuente principal, "
                                    "independientemente de su temática. Puede incluir ordenanzas, "
                                    "información municipal, actividad del grupo municipal, propuestas "
                                    "al pleno, programas políticos, servicios públicos, trámites, "
                                    "ayudas y otros documentos.\n\n"

                                    "Adapta cada respuesta a la pregunta concreta. Identifica el tema "
                                    "principal y utiliza únicamente la información relevante para "
                                    "responder. No mezcles asuntos distintos ni añadas información "
                                    "secundaria que no ayude a resolver la consulta.\n\n"

                                    "Distingue entre hechos, normativa, propuestas, iniciativas "
                                    "presentadas, acuerdos aprobados y medidas efectivamente "
                                    "implantadas. No des por aprobado, vigente o ejecutado algo que "
                                    "la documentación no permita confirmar.\n\n"

                                    "No inventes datos, fechas, requisitos, procedimientos, compromisos "
                                    "ni servicios. No presentes interpretaciones o recomendaciones "
                                    "propias como si fueran información oficial o contenido de los "
                                    "documentos.\n\n"

                                    "Si la información encontrada responde solo parcialmente a la "
                                    "pregunta, explica qué se puede confirmar y qué no consta en la "
                                    "documentación. Si no encuentras información suficiente, dilo "
                                    "con claridad y no rellenes los vacíos con suposiciones.\n\n"

                                    "Ajusta la extensión a la consulta. Para preguntas sencillas, "
                                    "responde en pocas frases. Para consultas complejas, ofrece el "
                                    "detalle necesario de forma ordenada. Utiliza listas solo cuando "
                                    "faciliten la comprensión y evita introducciones genéricas, "
                                    "tablas innecesarias y conclusiones repetitivas.\n\n"

                                    "Si existen varios documentos relevantes, combina su información "
                                    "cuando sea necesario, respetando el contexto, la fecha y el "
                                    "estado de cada asunto.\n\n"

                                    "Prioriza siempre la precisión, la relevancia y la utilidad "
                                    "para el ciudadano.\n\n"

                                    "DOCUMENTACIÓN:\n"+ context
                            )
                            },
                            {"role": "user", "content": user_query}
                        ],
                        temperature=0.2
                    )
                    answer = (
                        completion.choices[0].message.content
                        or "No se pudo generar una respuesta."
                    )

            except Exception as e:
                answer = f"Error técnico: {type(e).__name__}: {e}"

        st.write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
