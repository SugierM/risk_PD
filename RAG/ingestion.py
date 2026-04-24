import os
import hashlib
import json
import fitz
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DOCS_DIR,
)

def file_hash(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_pages(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append({"text": text, "page_number": page_num})
    doc.close()
    return pages

def chunk_pages(pages, source_file, regulation_name, language):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    texts, metadatas = [], []
    for page in pages:
        chunks = splitter.split_text(page["text"])
        for i, chunk_text in enumerate(chunks):
            texts.append(chunk_text)
            metadatas.append({
                "source_file": source_file,
                "page_number": page["page_number"],
                "regulation_name": regulation_name,
                "language": language,
                "chunk_index": i,
                "content": chunk_text 
            })
    return texts, metadatas

def run_ingestion():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Qdrant Ingestion Pipeline ===")
    print(f"Używane urządzenie: {device.upper()}")

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)

    print(f"Ładuję model: {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(
        EMBEDDING_MODEL, 
        cache_folder='/app/embedding_cache',
        device=device
    )
    vector_size = embed_model.get_sentence_embedding_dimension()

    collections = client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)
    
    if not exists:
        print(f"Tworzę nową kolekcję: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size, 
                distance=models.Distance.COSINE
            ),
        )
    else:
        print(f"Kolekcja {COLLECTION_NAME} już istnieje.")

    hash_file = "/app/embedding_cache/ingested_hashes.json"
    ingested_hashes = {}
    if os.path.exists(hash_file):
        try:
            with open(hash_file) as f: 
                ingested_hashes = json.load(f)
        except:
            ingested_hashes = {}

    if not os.path.exists(DOCS_DIR):
        print(f"BŁĄD: Folder {DOCS_DIR} nie istnieje!")
        return

    found_files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")]
    print(f"Znaleziono {len(found_files)} plików PDF.")

    for filename in found_files:
        filepath = os.path.join(DOCS_DIR, filename)
        current_hash = file_hash(filepath)

        if ingested_hashes.get(filename) == current_hash:
            print(f"  [SKIP] {filename} (brak zmian)")
            continue

        print(f"  [PROCESUJĘ] {filename}...")
        
        reg_name = filename.rsplit('.', 1)[0].replace('_', ' ')
        pol_keywords = ["pl", "knf", "rekomendacja", "ustawa", "rodo", "polska", "uchwała", "uchwala"]
        lang = "pl" if any(k in filename.lower() for k in pol_keywords) else "en"

        pages = extract_pages(filepath)
        texts, metadatas = chunk_pages(pages, filename, reg_name, lang)

        if texts:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.Filter(
                    must=[models.FieldCondition(key="source_file", match=models.MatchValue(value=filename))]
                ),
            )

            print(f"    Generowanie embeddingów dla {len(texts)} fragmentów...")
            embeddings = embed_model.encode(texts, normalize_embeddings=True).tolist()

            points = []
            for i in range(len(texts)):
                p_id = hashlib.md5(f"{filename}_{i}".encode()).hexdigest()
                points.append(models.PointStruct(
                    id=p_id,
                    vector=embeddings[i],
                    payload=metadatas[i]
                ))
            
            batch_size = 50 
            print(f"    Wysyłanie do Qdrant (łącznie {len(points)} fragmentów) w paczkach po {batch_size}...")
            
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=batch,
                    wait=True
                )
                if i % 500 == 0 and i > 0:
                    print(f"      Zapisano już {i} fragmentów...")

            ingested_hashes[filename] = current_hash
            print(f"    SUKCES: {filename} zindeksowany.")

    with open(hash_file, "w") as f: 
        json.dump(ingested_hashes, f, indent=4)
        
    print("=== Pipeline zakończony pomyślnie ===")

if __name__ == "__main__":
    run_ingestion()