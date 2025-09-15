from qdrant_client import QdrantClient, models
from src.models import Chunk

class QdrantRepository:
    def __init__(self, url: str):
        self.client = QdrantClient(url=url)

    def upsert(self, collection_name: str, chunk: Chunk):
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(collection_name, vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE))
        
        self.client.upsert(
            collection_name, 
            points=[
                models.PointStruct(
                    id=chunk.id,
                    payload={"document": chunk.document_name, 
                             "chunk_index": chunk.chunk_index, 
                             "start_page": chunk.start_page, 
                             "end_page": chunk.end_page,
                             "text": chunk.text,
                             "pages_content": chunk.pages_content},
                    vector=chunk.embedding
                )
            ]
        )

    def search(self, collection_name: str, search_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        search_results = self.client.query_points(
            collection_name, 
            search_embedding, 
            limit=top_k)
        chunks = []
        for result in search_results.points:
            chunks.append(Chunk(
                id=result.id,
                text=result.payload["text"],
                document_name=result.payload["document"],
                chunk_index=result.payload["chunk_index"],
                start_page=result.payload["start_page"],
                end_page=result.payload["end_page"],
                pages_content=result.payload["pages_content"],
            ))
        return chunks