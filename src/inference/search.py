from src.shared.qdrant_repository import QdrantRepository
from src.shared.embeddings import EmbeddingsService
from src.models import Chunk
from src.inference.reranker import Reranker


class SearchTool:

    def __init__(self, qdrant_repository: QdrantRepository, embeddings_service: EmbeddingsService, reranker: Reranker | None = None):
        self.qdrant_repository = qdrant_repository
        self.embeddings_service = embeddings_service
        self.reranker = reranker

    def search(self, query: str):
        search_embedding = self.embeddings_service.get_embeddings(query)
        if self.reranker:
            search_results = self.qdrant_repository.search("rag-pipeline", search_embedding, top_k=25)
            search_results = self.reranker.rerank(query, search_results)
        else:
            search_results = self.qdrant_repository.search("rag-pipeline", search_embedding, top_k=5)
        return search_results
        
    def format_search_results(self, search_results: list[Chunk], query: str) -> str:
        search_tool_output = "<ToolResponse>\n"
        search_tool_output += f"  <results query={query}>\n"
        for point in search_results:
            search_tool_output += f"    <chunk id={point.id} document_name={point.document_name} chunk_index={point.chunk_index}>\n"
            for page_index in point.pages_content:
                search_tool_output += f"      <page index={page_index}>\n"
                indented_content = "\n".join(
                f"        {line}" for line in point.pages_content[page_index].splitlines()
                )
                search_tool_output += f"      {indented_content}\n"
                search_tool_output += "      </page>\n"
            search_tool_output += "    </chunk>\n"
        search_tool_output += "  </results>\n"
        search_tool_output += "</ToolResponse>"
        return search_tool_output

    def __call__(self, query: str) -> str:
        search_results = self.search(query)
        return self.format_search_results(search_results, query), search_results
