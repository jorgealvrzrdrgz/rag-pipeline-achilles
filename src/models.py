from pydantic import BaseModel

class Chunk(BaseModel):
    id: str
    document_name: str
    text: str | None = None
    chunk_index: int
    start_page: int
    end_page: int
    pages_content: dict[int, str]
    embedding: list[float] | None = None

class ChunkReference(BaseModel):
    document_name: str
    chunk_index: int

class RAGResponse(BaseModel):
    query: str
    search_query: str | None = None
    answer: str
    retrieved_chunks: list[ChunkReference] | None = None
    references: list[ChunkReference] | None = None
