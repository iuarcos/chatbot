
import streamlit as st
from supabase import create_client

st.title("Prueba de acceso a Supabase Storage")

url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]

supabase = create_client(url, key)

bucket = "Bd_conocimiento"
archivo = "PROGRAMA IUARCOS._final_26mayo2023.pdf"

if st.button("Comprobar acceso al PDF"):
    try:
        respuesta = supabase.storage.from_(bucket).download(archivo)

        st.success("PDF descargado correctamente.")
        st.write(f"Tamaño descargado: {len(respuesta):,} bytes")

    except Exception as e:
        st.error("No se ha podido descargar el PDF.")
        st.code(str(e))