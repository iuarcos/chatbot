
import streamlit as st
from supabase import create_client
from groq import Groq
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Asistente IUARCOS",
    page_icon="📚",
    layout="centered"
)

try:
    supabase = create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )
    groq_client = Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )
except Exception:
    st.error("No se pudieron conectar los servicios.")
    st.stop()


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


try:
    embedding_model = load_embedding_model()
except Exception:
    st.error("No se pudo cargar el modelo de búsqueda.")
    st.stop()


SYSTEM_PROMPT = """
Eres el asistente virtual de IUARCOS (Izquierda Unida de Arcos de la Frontera).

Tu función es ayudar a las personas a encontrar y comprender la información
contenida en la documentación proporcionada.

COMPRENSIÓN DE LAS PREGUNTAS
- Interpreta el significado y la intención, no solo las palabras exactas.
- Reconoce sinónimos, paráfrasis, abreviaturas, errores ortográficos y
  distintas maneras de expresar una misma necesidad.
- Por ejemplo, «correo», «email», «correo electrónico» y «dirección de
  contacto» pueden referirse al mismo tipo de dato según el contexto.
- Aplica este criterio a todos los temas de los documentos, no solo a los
  datos de contacto.
- No confundas la ausencia de una palabra exacta con la ausencia del dato.

CONTEXTO CONVERSACIONAL
- Utiliza los mensajes anteriores para entender preguntas de seguimiento,
  pronombres y referencias como «él», «ella», «su», «eso» o «¿y qué más?».
- Si el usuario pregunta por un dato de una persona u organización mencionada
  anteriormente, interpreta la referencia con ese contexto.
- Si el usuario cambia de tema, responde al tema nuevo.
- Si hay varias interpretaciones relevantes, pide una aclaración breve.

USO DE LA DOCUMENTACIÓN
- Basa las respuestas sobre IUARCOS en la documentación facilitada.
- Relaciona fragmentos pertinentes cuando sea necesario.
- Respeta nombres, fechas, cifras, direcciones, propuestas y datos de contacto.
- No inventes información ni atribuyas datos a personas u organizaciones
  sin respaldo documental.
- Distingue los hechos confirmados de las opiniones y propuestas.

CUANDO FALTE INFORMACIÓN
- Revisa todos los fragmentos relevantes del contexto antes de concluir
  que un dato no aparece.
- Si puedes responder parcialmente, proporciona lo confirmado e indica
  brevemente qué parte no has podido verificar.
- Si la documentación no permite contestar, dilo con claridad.
- No afirmes que un dato no existe simplemente porque no aparezca en un
  fragmento concreto.

ESTILO
- Responde en español, de forma natural, directa y clara.
- Contesta primero a lo que se pregunta.
- Para preguntas concretas, responde brevemente.
- Evita introducciones, repeticiones y explicaciones innecesarias.
- Reproduce exactamente los datos concretos que figuren en los documentos.
"""


def build_context(documents, max_documents=18):
    sections = []
    seen = set()

    for doc in documents:
        content = (doc.get("content") or "").strip()
        metadata = doc.get("metadata") or {}
        source = metadata.get("source", "Documento sin nombre")
        key = doc.get("id") or (source, content)

        if not content or key in seen:
            continue

        seen.add(key)
        sections.append(
            f"FUENTE: {source}\n"
            f"CONTENIDO:\n{content}"
        )

        if len(sections) >= max_documents:
            break

    return "\n\n---\n\n".join(sections)


def retrieve_documents(query):
    """Combina búsqueda semántica y búsqueda por palabras clave."""
    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    semantic_result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": 15
        }
    ).execute()

    documents = semantic_result.data or []
    seen_ids = {
        doc.get("id")
        for doc in documents
        if doc.get("id") is not None
    }

    # La búsqueda por palabras clave es complementaria.
    # Si la función SQL todavía no existe, la búsqueda semántica sigue
    # funcionando mientras se configura.
    try:
        keyword_result = supabase.rpc(
            "search_documents_keyword",
            {
                "search_query": query,
                "result_limit": 10
            }
        ).execute()

        for doc in keyword_result.data or []:
            doc_id = doc.get("id")

            if doc_id is None or doc_id not in seen_ids:
                documents.append(doc)
                if doc_id is not None:
                    seen_ids.add(doc_id)

    except Exception:
        pass

    return documents


st.title("Asistente IUARCOS")
st.write(
    "Consulta tus dudas sobre la documentación de IUARCOS."
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "¡Hola! Puedes preguntarme sobre la documentación de IUARCOS."
            )
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_query = st.chat_input("Escribe tu pregunta...")

if user_query:
    # Guardamos el historial anterior antes de añadir el mensaje actual.
    previous_messages = st.session_state.messages.copy()

    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Buscando en la documentación..."):
                # Para preguntas breves de seguimiento, añadimos la última
                # pregunta del usuario a la consulta de recuperación.
                previous_user_query = next(
                    (
                        message["content"]
                        for message in reversed(previous_messages)
                        if message["role"] == "user"
                    ),
                    ""
                )

                retrieval_query = user_query

                if previous_user_query:
                    retrieval_query = (
                        f"Pregunta anterior: {previous_user_query}\n"
                        f"Pregunta actual: {user_query}"
                    )

                documents = retrieve_documents(retrieval_query)
                context = build_context(documents)

            if not context.strip():
                answer = (
                    "No he encontrado información suficiente en la "
                    "documentación para responder a esa pregunta. "
                    "Puedes probar a formularla de otra manera."
                )
            else:
                # Enviamos los mensajes recientes para que Groq comprenda
                # referencias como «su correo» o «¿y qué más?».
                recent_history = [
                    message
                    for message in st.session_state.messages
                    if message["role"] in ("user", "assistant")
                ][-8:]

                completion = groq_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                SYSTEM_PROMPT
                                + "\n\nDOCUMENTACIÓN RECUPERADA:\n"
                                + context
                            )
                        },
                        *recent_history
                    ],
                    temperature=0.1
                )

                answer = (
                    completion.choices[0].message.content
                    or "No se pudo generar una respuesta."
                )

        except Exception:
            answer = (
                "No he podido consultar la documentación correctamente. "
                "Inténtalo de nuevo más tarde."
            )

        st.write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )