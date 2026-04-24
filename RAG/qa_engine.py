from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from llm_provider import LLMProvider
from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
    SYSTEM_PROMPT,
    USER_PROMPT,
)

class RAGEngine:
    def __init__(self):
        self._embed_model: SentenceTransformer | None = None
        self._qdrant_client: QdrantClient | None = None
        self.llm = LLMProvider()
        
    @property
    def embed_model(self) -> SentenceTransformer:
        if self._embed_model is None:
            self._embed_model = SentenceTransformer(
                EMBEDDING_MODEL, 
                cache_folder='/app/embedding_cache'
            )
        return self._embed_model

    @property
    def client(self) -> QdrantClient:
        if self._qdrant_client is None:
            self._qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        return self._qdrant_client

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[dict]:
        query_embedding = self.embed_model.encode(
            question, normalize_embeddings=True
        ).tolist()

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            with_payload=True
        )

        chunks = []
        for res in response.points:
            chunks.append({
                "id": res.id,
                "text": res.payload.get("content", ""),
                "metadata": res.payload,
                "score": res.score,
            })
        return chunks


    def build_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            header = (
                f"[Fragment {i} | {meta.get('regulation_name', 'Nieznany')} | "
                f"str. {meta.get('page_number', '?')} | {meta.get('source_file', '?')}]"
            )
            parts.append(f"{header}\n{chunk['text']}")
        return "\n\n\n\n".join(parts)

    def ask(self, question: str, top_k: int = TOP_K, stream: bool = False) -> dict:
        chunks = self.retrieve(question, top_k=top_k)
        context = self.build_context(chunks)

        system_msg = SYSTEM_PROMPT.format(context=context)
        user_msg = USER_PROMPT.format(question=question)

        sources = [
            {
                "fragment_id": i + 1,
                "regulation": c["metadata"].get("regulation_name"),
                "page": c["metadata"].get("page_number"),
                "file": c["metadata"].get("source_file"),
                "score": round(c["score"], 4),
                "text_preview": c["text"][:300],
            }
            for i, c in enumerate(chunks)
        ]

        result = self.llm.generate(system_msg, user_msg, stream=stream)

        if stream:
            return {"stream": result, "sources": sources}
        else:
            return {"answer": result, "sources": sources}