import json
import re

from openai import OpenAI

from src.inference.search import SearchTool
from src.models import RAGResponse, ChunkReference


class AskService:
    def __init__(self, openai_client: OpenAI,system_prompt: str, tools: list[dict], search_tool: SearchTool):
        
        self.openai_client = openai_client
        self.system_prompt = system_prompt
        self.tools = tools
        self.search_tool = search_tool
        
    def ask(self, query: str) -> RAGResponse:
        input_list = [
            {"role": "developer", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]
        response = self.openai_client.responses.create(
            input=input_list,
            model="gpt-4.1-2025-04-14",
            tools=self.tools
        )
        input_list += response.output
        tool_calls = [out for out in response.output if out.type == "function_call"]
        
        if not tool_calls:
            return response
        
        for tool_call in tool_calls:
            name = tool_call.name
            args = json.loads(tool_call.arguments)
            if name == "search":
                result, retrieved_chunks = self.search_tool(**args)
                input_list.append({"type": "function_call_output", "call_id": tool_call.call_id, "output": result})
        
        response = self.openai_client.responses.create(
            input=input_list,
            model="gpt-4.1-2025-04-14",
            tools=self.tools
        )
        references = self.postprocess_references(response.output_text, retrieved_chunks)
        return RAGResponse(
            query=query,
            search_query=args["query"],
            answer=response.output_text,
            retrieved_chunks=[ChunkReference(chunk_index=chunk.chunk_index, document_name=chunk.document_name) for chunk in retrieved_chunks],
            references=[ChunkReference(chunk_index=chunk.chunk_index, document_name=chunk.document_name) for chunk in references]
        )

    def postprocess_references(self, response: str, retrieved_chunks: list[ChunkReference]) -> list[ChunkReference]:
        
        pattern = r'\[([^:]+)::(\d+)\]'
        matches = re.findall(pattern, response)
        chunk_lookup = {}
        for chunk in retrieved_chunks:
            key = (chunk.document_name, chunk.chunk_index)
            chunk_lookup[key] = chunk
        
        referenced_chunks = []
        for document_name, chunk_index_str in matches:
            chunk_index = int(chunk_index_str)
            key = (document_name, chunk_index)
            
            if key in chunk_lookup:
                chunk = chunk_lookup[key]
                if chunk not in referenced_chunks:
                    referenced_chunks.append(chunk)
        
        return referenced_chunks