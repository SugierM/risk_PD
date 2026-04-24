import os

# --- Infrastructure ---
# Zmieniamy Chroma na Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pd_regulations")

# Ollama - adres host.docker.internal pozwala na dostęp do Ollamy zainstalowanej na Windows/Mac
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

# Model embeddingowy - zgodnie z Twoim folderem embedding_cache
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "intfloat/multilingual-e5-large"
)

# Model LLM - Bielik to świetny wybór dla polskiego kontekstu bankowego
LLM_MODEL = os.getenv("LLM_MODEL", "SpeakLeash/bielik-11b-v2.1-instruct:Q6_K")
DOCS_DIR = os.getenv("DOCS_DIR", "/app/docs")

# --- RAG Parameters ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K = 8

# --- Prompts ---
SYSTEM_PROMPT = """\
Jesteś ekspertem regulacyjnym ds. modeli PD (Probability of Default) \
stosowanych w bankach.
Odpowiadasz na pytania na podstawie WYŁĄCZNIE dostarczonych fragmentów \
dokumentów regulacyjnych.

ZASADY:
1. Odpowiadaj ZAWSZE po polsku, niezależnie od języka dokumentu źródłowego.
2. Przy KAŻDYM stwierdzeniu podaj źródło w formacie:
   [Źródło: {{regulation_name}}, strona {{page_number}}, \
plik: {{source_file}}]
3. Jeśli fragmenty nie zawierają odpowiedzi — powiedz to wprost. \
NIE wymyślaj informacji.
4. Jeśli pytanie dotyczy konkretnej zmiennej/kolumny danych — oceń:
   - Czy na podstawie dostarczonych dokumentów można zdecydować, czy dana zmienna może zostać wykorzystana podczas budowania modelu
   - Jakie są ryzyka związane z jej użyciem w modelu PD
5. Bądź precyzyjny i konkretny. Podawaj numery artykułów i paragrafów.

FRAGMENTY DOKUMENTÓW:
{context}
"""

USER_PROMPT = "Pytanie: {question}"