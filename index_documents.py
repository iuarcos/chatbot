
import os
import csv
import io
import hashlib
from pathlib import PurePosixPath

from supabase import create_client
from sentence_transformers import SentenceTransformer


BUCKET = "Bd_conocimiento"
SUPPORTED_EXTENSIONS = {".txt", ".csv"}
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 50

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"]
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def split_text(text):
    text = text.strip()
    if not text:
        return []

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start in range(0, len(text), step):
        chunk = text[start:start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def read_document(filename, file_bytes):
    text = file_bytes.decode(
        "utf-8-sig", errors="replace"
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

        rows = list(csv.reader(io.StringIO(text), dialect=dialect))

        if not rows:
            return ""

        headers = [value.strip() for value in rows[0]]
        lines = [
            "Archivo de datos: " + filename,
            "Columnas: " + " | ".join(headers)
        ]

        for row_number, row in enumerate(rows[1:], start=2):
            fields = []

            for i, value in enumerate(row):
                header = (
                    headers[i] if i < len(headers) and headers[i]
                    else f"Columna {i + 1}"
                )
                fields.append(f"{header}: {value.strip()}")

            if fields:
                lines.append(
                    f"Fila {row_number}: " + " | ".join(fields)
                )

        return "\n".join(lines).strip()

    return ""


def list_supported_files():
    files = []
    offset = 0
    limit = 500

    while True:
        batch = supabase.storage.from_(BUCKET).list(
            "",
            {"limit": limit, "offset": offset}
        )

        if not batch:
            break

        for item in batch:
            name = item.get("name", "")

            if (
                name
                and item.get("id") is not None
                and PurePosixPath(name).suffix.lower()
                in SUPPORTED_EXTENSIONS
            ):
                files.append(item)

        if len(batch) < limit:
            break

        offset += limit

    return files


def get_existing(filename):
    result = (
        supabase.table("documents")
        .select("id, metadata")
        .filter("metadata->>source", "eq", filename)
        .execute()
    )
    return result.data or []


def storage_markers(item):
    storage_metadata = item.get("metadata") or {}

    return {
        "storage_updated_at": (
            item.get("updated_at") or item.get("created_at")
        ),
        "storage_etag": (
            storage_metadata.get("eTag")
            or storage_metadata.get("etag")
        ),
        "storage_size": storage_metadata.get("size"),
    }


def storage_metadata_matches(existing, markers):
    if not existing:
        return False

    old = existing[0].get("metadata") or {}

    # ETag o fecha de modificación permiten detectar cambios sin descargar.
    reliable_keys = [
        key for key in ("storage_etag", "storage_updated_at")
        if markers.get(key) is not None
    ]

    if not reliable_keys:
        return False

    return all(
        old.get(key) == markers[key]
        for key in reliable_keys
    )


def update_storage_markers(existing, markers):
    # Si el contenido no ha cambiado, actualiza solo los marcadores
    # para no descargar repetidamente el mismo archivo.
    for row in existing:
        metadata = row.get("metadata") or {}
        metadata.update(markers)

        supabase.table("documents").update(
            {"metadata": metadata}
        ).eq("id", row["id"]).execute()


def index_file(filename, file_bytes, markers):
    fingerprint = hashlib.sha256(file_bytes).hexdigest()

    existing = get_existing(filename)

    if existing:
        old_fingerprint = (
            existing[0].get("metadata") or {}
        ).get("fingerprint")

        if old_fingerprint == fingerprint:
            update_storage_markers(existing, markers)
            print(f"Sin cambios de contenido: {filename}")
            return 0

    text = read_document(filename, file_bytes)

    if not text:
        print(f"AVISO: {filename} está vacío; se conserva el índice anterior.")
        return 0

    chunks = split_text(text)

    if not chunks:
        print(f"AVISO: no se generaron fragmentos para {filename}.")
        return 0

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
                "chunk_index": i,
                **markers,
            },
            "embedding": embedding,
        }
        for i, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        )
    ]

    # Insertamos primero los nuevos fragmentos. Si falla la inserción,
    # no borramos deliberadamente el índice anterior.
    for offset in range(0, len(records), BATCH_SIZE):
        supabase.table("documents").insert(
            records[offset:offset + BATCH_SIZE]
        ).execute()

    # Una vez insertados, retiramos los fragmentos de versiones anteriores.
    supabase.table("documents").delete().filter(
        "metadata->>source", "eq", filename
    ).filter(
        "metadata->>fingerprint", "neq", fingerprint
    ).execute()

    print(f"Indexado: {filename} ({len(chunks)} fragmentos)")
    return len(chunks)


def delete_removed_files(current_filenames):
    # Solo borra registros de archivos que ya no están en Storage.
    result = (
        supabase.table("documents")
        .select("id, metadata")
        .execute()
    )

    sources = set()

    for row in result.data or []:
        source = (row.get("metadata") or {}).get("source")
        if source:
            sources.add(source)

    for source in sources - current_filenames:
        print(f"Eliminado de Storage; retirando índice: {source}")

        supabase.table("documents").delete().filter(
            "metadata->>source", "eq", source
        ).execute()


def main():
    files = list_supported_files()
    current_filenames = {item["name"] for item in files}
    total_chunks = 0

    for item in files:
        filename = item["name"]
        markers = storage_markers(item)
        existing = get_existing(filename)

        # Si la fecha o el ETag coinciden, no se descarga el archivo.
        if storage_metadata_matches(existing, markers):
            print(f"Sin cambios en Storage: {filename}")
            continue

        try:
            file_bytes = supabase.storage.from_(BUCKET).download(
                filename
            )
            total_chunks += index_file(
                filename, file_bytes, markers
            )
        except Exception as exc:
            # No borramos el índice anterior si falla la descarga.
            print(f"ERROR procesando {filename}: {exc}")
            raise

    delete_removed_files(current_filenames)
    print(f"Proceso terminado. Fragmentos nuevos: {total_chunks}")


if __name__ == "__main__":
    main()