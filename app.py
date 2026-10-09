
import streamlit as st
from supabase import create_client
from groq import Groq
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Asistente IA",
    page_icon="📚",
    layout="centered"
)

# 1. Load credentials from Streamlit Secrets
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    SUPABASE_SERVICE_ROLE_KEY = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

    # Public client for document search
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Admin client for indexing and maintenance
    supabase_admin = create_client(
        SUPABASE_URL,
        SUPABASE_SERVICE_ROLE_KEY
    )

    groq_client = Groq(api_key=GROQ_API_KEY)

except Exception as e:
    st.error("Could not initialize the services. Check Streamlit Secrets.")
    st.code(str(e))
    st.stop()


# 2. Load the embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


with st.spinner("Loading the embedding model..."):
    try:
        embedding_model = load_embedding_model()
    except Exception as e:
        st.error("Could not load the embedding model.")
        st.code(str(e))
        st.stop()


# 3. Application title
st.title("Document Assistant")
st.write("Ask questions about the available documentation.")


# 4. Storage diagnostic
with st.expander("Technical test: Supabase Storage"):
    st.write("Bucket: Bd_conocimiento")

    if st.button("List bucket files", key="list_storage"):
        try:
            files = supabase.storage.from_("Bd_conocimiento").list()

            if files:
                st.success("Storage listing succeeded.")
                for file_info in files:
                    st.write(file_info.get("name", "(unnamed object)"))
            else:
                st.warning(
                    "The bucket is accessible, but no files were listed "
                    "at its root."
                )

        except Exception as e:
            st.error("Could not list Storage files.")
            st.code(str(e))

    if st.button("Test PDF download", key="download_pdf"):
        try:
            pdf_bytes = supabase.storage.from_(
                "Bd_conocimiento"
            ).download(
                "PROGRAMA IUARCOS._final_26mayo2023.pdf"
            )

            if pdf_bytes and pdf_bytes.startswith(b"%PDF-"):
                st.success(
                    f"PDF downloaded successfully: {len(pdf_bytes):,} bytes."
                )
            else:
                st.error(
                    "The download returned data, but it does not look "
                    "like a PDF file."
                )

        except Exception as e:
            st.error("PDF download failed.")
            st.code(str(e))

# 4.1 Index PDF into Supabase
with st.expander("Index documentation"):
    st.write(
        "Extract the PDF text, split it into chunks, "
        "generate embeddings and save them to Supabase."
    )

    if st.button("Index PDF now", key="index_pdf"):
        from io import BytesIO
        from pypdf import PdfReader

        try:
            with st.spinner("Downloading and reading the PDF..."):
                pdf_bytes = supabase.storage.from_(
                    "Bd_conocimiento"
                ).download(
                    "PROGRAMA IUARCOS._final_26mayo2023.pdf"
                )

                reader = PdfReader(BytesIO(pdf_bytes))
                pages = [
                    page.extract_text() or ""
                    for page in reader.pages
                ]
                full_text = "\n".join(pages).strip()

            if not full_text:
                st.error(
                    "No text could be extracted from the PDF. "
                    "It may be scanned or image-based."
                )
            else:
                chunk_size = 1000
                overlap = 150
                chunks = []
                start = 0

                while start < len(full_text):
                    chunk = full_text[start:start + chunk_size].strip()
                    if chunk:
                        chunks.append(chunk)
                    start += chunk_size - overlap

                with st.spinner("Generating embeddings..."):
                    embeddings = embedding_model.encode(
                        chunks,
                        normalize_embeddings=True
                    ).tolist()

                st.write(f"Text extracted: {len(full_text):,} characters")
                st.write(f"Chunks prepared: {len(chunks)}")

                with st.spinner("Saving chunks to Supabase..."):
                    # Remove previous chunks from this source.
                    # Existing legacy rows with null metadata are preserved.
                    supabase_admin.table("documents").delete().filter(
                        "metadata->>source", "eq",
                        "PROGRAMA IUARCOS._final_26mayo2023.pdf"
                    ).execute()

                    records = [
                        {
                            "content": chunk,
                            "metadata": {
                                "source": "PROGRAMA IUARCOS._final_26mayo2023.pdf",
                                "chunk_index": index,
                                "page_source": "PDF"
                            },
                            "embedding": embedding
                        }
                        for index, (chunk, embedding)
                        in enumerate(zip(chunks, embeddings))
                    ]

                    batch_size = 50
                    for offset in range(0, len(records), batch_size):
                        supabase_admin.table("documents").insert(
                            records[offset:offset + batch_size]
                        ).execute()

                st.success(
                    f"Indexing completed: {len(records)} chunks saved."
                )

        except Exception as e:
            st.error("Indexing failed.")
            st.exception(e)

# 5. Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! Ask me a question about the available documents."
            )
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# 6. Search documents and answer
user_query = st.chat_input("Type your question here...")

if user_query:
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documentation..."):
            try:
                query_embedding = embedding_model.encode(
                    user_query
                ).tolist()

                result = supabase.rpc(
                    "match_documents",
                    {
                        "query_embedding": query_embedding,
                        "match_threshold": 0.0,
                        "match_count": 3
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
                        "I could not find relevant text in the available "
                        "documentation. Please check that the documents "
                        "have been indexed correctly."
                    )
                else:
                    completion = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a helpful assistant. "
                                    "Answer in Spanish. Use the supplied "
                                    "document excerpts as the source of truth. "
                                    "If they do not contain enough information "
                                    "to answer, say so. Do not invent facts.\n\n"
                                    "DOCUMENT EXCERPTS:\n" + context
                                )
                            },
                            {
                                "role": "user",
                                "content": user_query
                            }
                        ],
                        temperature=0.2
                    )

                    answer = completion.choices[0].message.content

                    if not answer:
                        answer = "The model returned an empty response."

            except Exception as e:
                answer = (
                    "An error occurred while searching the documents "
                    "or generating the answer. Check the technical details "
                    "below."
                )
                st.error(str(e))

        st.write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )