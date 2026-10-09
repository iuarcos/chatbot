
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
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
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