import streamlit as st
from supabase import create_client
from groq import Groq
from sentence_transformers import SentenceTransformer
from pathlib import PurePosixPath
import hashlib
import csv
import io

st.set_page_config(
    page_title="Asistente IUARCOS",
    page_icon="📚",
    layout="centered"
)

BUCKET = "Bd_conocimiento"
SUPPORTED_EXTENSIONS = {".txt", ".csv"}
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 50

try:
    supabase = create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )
    supabase_admin = create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
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


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.strip()
    if not text:
        return []

    chunks = []
    step = chunk_size - overlap
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def read_document(filename, file_bytes):
    text = file_bytes.decode(
        "utf-8-sig",
        errors="replace"
    ).strip()

    if filename.lower().endswith(".txt"):
        return text

    if filename.lower().endswith(".csv"):
        if not text:
            return ""

        try:
            dialect = csv.Sniffer().sniff(
                text[:10000],
                delimiters=",;\t|"
            )
        except csv.Error:
            dialect = csv.excel

        reader = csv.reader(
            io.StringIO(text),
            dialect=dialect
        )

        rows = list(reader)

        if not rows:
            return ""

        headers = [
            value.strip() for value in rows[0]
        ]

        lines = [
            "Archivo de datos: " + filename,
            "Columnas: " + " | ".join(headers)
        ]

        for row_number, row in enumerate(rows[1:], start=2):
            fields = []

            for i, value in enumerate(row):
                if i < len(headers):
                    header = headers[i] or f"Columna {i + 1}"
                else:
                    header = f"Columna {i + 1}"

                fields.append(f"{header}: {value.strip()}")

            if fields:
                lines.append(
                    f"Fila {row_number}: " + " | ".join(fields)
                )

        return "\n".join(lines).strip()

    return ""


def list_supported_files():
    """Busca archivos TXT y CSV en la raíz del bucket."""
    files = []
    offset = 0
    limit = 500

    while True:
        batch = supabase_admin.storage.from_(BUCKET).list(
            "",
            {"limit": limit, "offset": offset}
        )

        if not batch:
            break

        for item in batch:
            name = item.get("name", "")

            # Ignora carpetas y otros tipos de archivo.
            if (
                name
                and item.get("id") is not None
                and PurePosixPath(name).suffix.lower()
                in SUPPORTED_EXTENSIONS
            ):
                files.append(name)

        if len(batch) < limit:
            break

        offset += limit

    return files


def index_file(filename, file_bytes):
    """Indexa un archivo solo si es nuevo o ha cambiado."""
    fingerprint = hashlib.sha256(file_bytes).hexdigest()

    existing = (
        supabase_admin.table("documents")
        .select("metadata")
        .filter("metadata->>source", "eq", filename)
        .limit(1)
        .execute()
    )

    old_rows = existing.data or []

    if old_rows:
        metadata = old_rows[0].get("metadata") or {}

        if metadata.get("fingerprint") == fingerprint:
            return 0

    text = read_document(filename, file_bytes)

    if not text:
        raise ValueError(
            f"El archivo {filename} está vacío o no contiene texto."
        )

    chunks = split_text(text)

    if not chunks:
        raise ValueError(
            f"No se pudieron crear fragmentos de {filename}."
        )

    embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True
    ).tolist()

    records = [
        {
            "content": chunk,
            "metadata": {
                "source": filename,
                "fingerprint": fingerprint,
                "chunk_index": i
            },
            "embedding": embedding
        }
        for i, (chunk, embedding)
        in enumerate(zip(chunks, embeddings))
    ]

    # Sustituye los fragmentos anteriores de este archivo.
    # Si falla el borrado, no se insertan nuevos fragmentos.
    supabase_admin.table("documents").delete().filter(
        "metadata->>source", "eq", filename
    ).execute()

    for offset in range(0, len(records), BATCH_SIZE):
        supabase_admin.table("documents").insert(
            records[offset:offset + BATCH_SIZE]
        ).execute()

    return len(chunks)


def sync_documents():
    """
    Comprueba Storage antes de cada consulta.
    Los archivos sin cambios no generan embeddings otra vez.
    """
    files = list_supported_files()
    indexed = 0

    for filename in files:
        file_bytes = supabase_admin.storage.from_(BUCKET).download(
            filename
        )

        indexed += index_file(filename, file_bytes)

    return indexed


def build_context(documents):
    sections = []

    for doc in documents:
        content = doc.get("content", "").strip()
        metadata = doc.get("metadata") or {}
        source = metadata.get("source", "Documento sin nombre")

        if content:
            sections.append(
                f"FUENTE: {source}\n"
                f"CONTENIDO:\n{content}"
            )

    return "\n\n---\n\n".join(sections)


SYSTEM_PROMPT = """
Eres el asistente virtual de IUARCOS, el grupo municipal de Izquierda
Unida en Arcos de la Frontera.

Tu función es responder a las preguntas de la ciudadanía utilizando
la documentación recuperada de la base de conocimiento.

ESTILO DE RESPUESTA
- Responde en español, de forma natural, cercana y directa.
- Contesta primero a lo que te han preguntado.
- Para preguntas sencillas, utiliza una sola frase si es suficiente.
- Evita introducciones como "Según la documentación disponible",
  "En relación con tu consulta" o "Cabe destacar que".
- No repitas la pregunta ni añadas conclusiones innecesarias.
- No incluyas recomendaciones, explicaciones adicionales ni ofertas
  de ayuda que no sean relevantes para la consulta.
- Si el usuario hace una pregunta de seguimiento, interpreta el
  contexto de la conversación. Por ejemplo, "¿y el email?" se refiere
  al correo de la persona o grupo del que se estaba hablando.
- Si preguntan por un correo electrónico, proporciona la dirección
  exacta y, si resulta natural, indica a quién pertenece.

FIDELIDAD A LA DOCUMENTACIÓN
- No inventes nombres, cargos, correos, teléfonos, fechas ni datos.
- Utiliza únicamente los datos que estén respaldados por el contexto.
- Si el documento identifica a una persona como concejal, no cambies
  su cargo ni atribuyas a esa persona datos de otra.
- Distingue entre propuestas presentadas, acuerdos aprobados y
  medidas implantadas.
- Si no encuentras el dato solicitado, di brevemente que no consta
  en la información consultada.
- Si solo encuentras parte de la respuesta, proporciona esa parte y
  aclara de forma concisa qué dato falta.
- No digas que un dato es el único disponible salvo que sea necesario
  para responder y la documentación permita confirmarlo.

FORMATO
- No utilices listas para responder a preguntas que se resuelven
  con una frase.
- No conviertas direcciones de correo en enlaces Markdown.
  Escribe el correo en texto normal.
- Menciona el nombre del archivo solo cuando el usuario pregunte
  por la fuente o cuando sea necesario para aclarar una discrepancia.

DOCUMENTACIÓN RECUPERADA:
"""


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
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Comprobando la documentación..."):
                # 1. Actualiza el índice antes de buscar la respuesta.
                sync_documents()

            with st.spinner("Consultando la documentación..."):
                # 2. Genera el embedding de la pregunta.
                query_embedding = embedding_model.encode(
                    user_query,
                    normalize_embeddings=True
                ).tolist()

                # Esta llamada recupera también los metadatos y la fuente.
                result = supabase.rpc(
                    "match_documents",
                    {
                        "query_embedding": query_embedding,
                        "match_count": 15
                    }
                ).execute()

                documents = result.data or []
                context = build_context(documents)

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
                                "content": SYSTEM_PROMPT + "\n\n" + context
                            },
                            {
                                "role": "user",
                                "content": user_query
                            }
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