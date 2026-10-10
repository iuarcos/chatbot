
import os
import csv
import io
import re
import hashlib
from pathlib import PurePosixPath

from supabase import create_client
from sentence_transformers import SentenceTransformer


BUCKET = "Bd_conocimiento"
SUPPORTED_EXTENSIONS = {".txt", ".csv"}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 50

# Cambia esta versión cuando modifiques la estrategia de fragmentación.
INDEX_VERSION = "2"

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"]
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def split_text(text):
    """
    Divide el documento procurando conservar párrafos y frases.
    Evita cortar palabras, direcciones de correo y URL.
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    units = []

    for paragraph in paragraphs:
        if len(paragraph) <= CHUNK_SIZE:
            units.append(paragraph)
            continue

        # Separar por frases cuando sea posible.
        sentences = re.split(
            r'(?<=[.!?])\s+',
            paragraph
        )

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) <= CHUNK_SIZE:
                units.append(sentence)
                continue

            # Si una frase es demasiado larga, dividir por espacios.
            words = sentence.split()
            piece = ""

            for word in words:
                candidate = f"{piece} {word}".strip()

                if len(candidate) > CHUNK_SIZE and piece:
                    units.append(piece)
                    piece = word
                else:
                    piece = candidate

            if piece:
                units.append(piece)

    # Agrupar las unidades procurando no superar CHUNK_SIZE.
    chunks = []
    current = ""

    for unit in units:
        candidate = f"{current}\n{unit}".strip()

        if len(candidate) <= CHUNK_SIZE or not current:
            current = candidate
        else:
            chunks.append(current)
            current = unit

    if current:
        chunks.append(current)

    # Solapamiento breve, preferiblemente de una frase completa.
    if CHUNK_OVERLAP > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]

        for i in range(1, len(chunks)):
            previous = chunks[i - 1]
            current = chunks[i]

            last_sentence = re.split(
                r'(?<=[.!?])\s+',
                previous
            )[-1].strip()

            if (
                last_sentence
                and len(last_sentence) <= CHUNK_OVERLAP
                and last_sentence not in current
                and len(last_sentence) + len(current) + 1
                <= CHUNK_SIZE
            ):
                current = last_sentence + "\n" + current

            overlapped.append(current)

        chunks = overlapped

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

        rows = list(
            csv.reader(
                io.StringIO(text),
                dialect=dialect
            )
        )

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
                    headers[i]
                    if i < len(headers) and headers[i]
                    else f"Columna {i + 1}"
                )
                fields.append(f"{header}: {value.strip()}")

            if fields:
                lines.append(
                    f"Fila {row_number}: "
                    + " | ".join(fields)
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

    # Fuerza la reindexación si cambia la versión del fragmentador.
    if old.get("index_version") != INDEX_VERSION:
        return False

    reliable_keys = [
        key
        for key in ("storage_etag", "storage_updated_at")
        if markers.get(key) is not None
    ]

    if not reliable_keys:
        return False

    return all(
        old.get(key) == markers[key]
        for key in reliable_keys
    )


def update_storage_markers(existing, markers):
    for row in existing:
        metadata = row.get("metadata") or {}
        metadata.update(markers)

        (
            supabase.table("documents")
            .update({"metadata": metadata})
            .eq("id", row["id"])
            .execute()
        )


def index_file(filename, file_bytes, markers):
    fingerprint = hashlib.sha256(
        file_bytes
        + f"|index_version={INDEX_VERSION}".encode("utf-8")
    ).hexdigest()

    existing = get_existing(filename)

    if existing:
        old_fingerprint = (
            existing[0].get("metadata") or {}
        ).get("fingerprint")

        old_version = (
            existing[0].get("metadata") or {}
        ).get("index_version")

        if (
            old_fingerprint == fingerprint
            and old_version == INDEX_VERSION
        ):
            update_storage_markers(existing, markers)
            print(f"Sin cambios de contenido: {filename}")
            return 0

    text = read_document(filename, file_bytes)

    if not text:
        print(
            f"AVISO: {filename} está vacío; "
            "se conserva el índice anterior."
        )
        return 0

    chunks = split_text(text)

    if not chunks:
        print(
            f"AVISO: no se generaron fragmentos para {filename}."
        )
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
                "index_version": INDEX_VERSION,
                "chunk_index": i,
                **markers,
            },
            "embedding": embedding,
        }
        for i, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        )
    ]

    # Insertar primero los fragmentos nuevos.
    for offset in range(0, len(records), BATCH_SIZE):
        (
            supabase.table("documents")
            .insert(records[offset:offset + BATCH_SIZE])
            .execute()
        )

    # Borrar los fragmentos anteriores solo después de insertar.
    (
        supabase.table("documents")
        .delete()
        .filter("metadata->>source", "eq", filename)
        .filter("metadata->>fingerprint", "neq", fingerprint)
        .execute()
    )

    print(
        f"Indexado: {filename} "
        f"({len(chunks)} fragmentos, versión {INDEX_VERSION})"
    )

    return len(chunks)


def delete_removed_files(current_filenames):
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
        print(
            f"Eliminado de Storage; retirando índice: {source}"
        )

        (
            supabase.table("documents")
            .delete()
            .filter("metadata->>source", "eq", source)
            .execute()
        )


def main():
    files = list_supported_files()
    current_filenames = {
        item["name"] for item in files
    }

    total_chunks = 0

    for item in files:
        filename = item["name"]
        markers = storage_markers(item)
        existing = get_existing(filename)

        if storage_metadata_matches(existing, markers):
            print(f"Sin cambios en Storage: {filename}")
            continue

        try:
            file_bytes = (
                supabase.storage
                .from_(BUCKET)
                .download(filename)
            )

            total_chunks += index_file(
                filename,
                file_bytes,
                markers
            )

        except Exception as exc:
            print(
                f"ERROR procesando {filename}: "
                f"{type(exc).__name__}: {exc}"
            )
            raise

    delete_removed_files(current_filenames)

    print(
        f"Proceso terminado. "
        f"Fragmentos nuevos: {total_chunks}"
    )


if __name__ == "__main__":
    main()
