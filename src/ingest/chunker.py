import uuid
from typing import Any, Dict, List, Optional, Tuple

from src.models import Chunk


class Chunker:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def chunk(
        self, 
        document_data: Dict[str, Any],
        chunk_size: int = 512,
        overlap: int = 128,
    ) -> List[Chunk]:
        """
        Método principal para crear chunks de un documento.
        
        Args:
            document_data: Diccionario con "document_name" y "pages" {página: texto}
            chunk_size: Tamaño máximo de cada chunk en tokens
            overlap: Número de tokens de solapamiento entre chunks
            
        Returns:
            Lista de objetos Chunk con todos los metadatos
        """
        document_name = document_data.get("document_name", "unknown")
        pages_dict = document_data.get("pages", {})
        
        self._validate_parameters(pages_dict, chunk_size, overlap)
        all_ids, page_map = self._tokenize_pages(pages_dict)
        if not all_ids:
            return []
        full_text = self.tokenizer.decode(all_ids, skip_special_tokens=True)
        return self._create_chunks(
            all_ids, page_map, full_text, chunk_size, overlap, document_name, pages_dict
        )

    def _validate_parameters(
        self, 
        pages_dict: Dict[int, str], 
        chunk_size: int, 
        overlap: int
    ) -> None:
        """Valida los parámetros de entrada."""
        if not isinstance(pages_dict, dict):
            raise ValueError("pages_dict debe ser un dict[int, str].")
        if chunk_size <= 0:
            raise ValueError("chunk_size debe ser > 0.")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap debe ser >= 0 y < chunk_size.")

    def _tokenize_pages(
        self, 
        pages_dict: Dict[int, str], 
    ) -> Tuple[List[int], List[Optional[int]]]:
        """
        Tokeniza todas las páginas y crea un mapeo de tokens a páginas.
        
        Returns:
            Tupla de (lista_tokens, mapeo_token_a_pagina)
        """
        # Ordenar por número de página (ascendente, 1-based)
        items: List[Tuple[int, str]] = sorted(pages_dict.items(), key=lambda x: x[0])

        all_ids: List[int] = []
        page_map: List[Optional[int]] = []  # token -> página (1-based)

        for page_num, text in items:
            text = text or ""
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if ids:
                all_ids.extend(ids)
                page_map.extend([page_num] * len(ids))

        return all_ids, page_map

    def _create_chunks(
        self,
        all_ids: List[int],
        page_map: List[Optional[int]],
        full_text: str,
        chunk_size: int,
        overlap: int,
        document_name: str,
        pages_dict: Dict[int, str]
    ) -> List[Chunk]:
        """
        Crea los chunks con sus metadatos correspondientes.
        """
        chunks: List[Chunk] = []
        stride = chunk_size - overlap
        n = len(all_ids)

        start = 0
        chunk_index = 0
        while start < n:
            end = min(start + chunk_size, n)
            
            chunk_data = self._create_single_chunk(
                all_ids, page_map, full_text, start, end, n, chunk_index, document_name, pages_dict
            )
            chunks.append(chunk_data)

            if end == n:
                break
            start += stride
            chunk_index += 1

        return chunks

    def _create_single_chunk(
        self,
        all_ids: List[int],
        page_map: List[Optional[int]],
        full_text: str,
        start: int,
        end: int,
        total_tokens: int,
        chunk_index: int,
        document_name: str,
        pages_dict: Dict[int, str]
    ) -> Chunk:
        """
        Crea un chunk individual con todos sus metadatos.
        """
        ids_chunk = all_ids[start:end]
        map_chunk = page_map[start:end]

        # Páginas (1-based) cubiertas por este chunk
        start_page = self._find_start_page(map_chunk)
        end_page = self._find_end_page(map_chunk)

        # Texto del chunk
        text_chunk = self.tokenizer.decode(ids_chunk, skip_special_tokens=True)

        # Posiciones de carácter en el texto completo
        start_char, end_char = self._calculate_char_positions(
            all_ids, start, end, total_tokens, full_text
        )

        # Extraer el contenido de las páginas específicas del chunk
        pages_content = self._extract_pages_content(map_chunk, pages_dict, ids_chunk)

        return Chunk(
            id=str(uuid.uuid4()),
            document_name=document_name,
            text=text_chunk,
            chunk_index=chunk_index,
            start_page=start_page or 1,    # Default a 1 si es None
            end_page=end_page or 1,        # Default a 1 si es None
            pages_content=pages_content
        )

    def _find_start_page(self, map_chunk: List[Optional[int]]) -> Optional[int]:
        """Encuentra la primera página en el chunk."""
        return next((p for p in map_chunk if p is not None), None)

    def _find_end_page(self, map_chunk: List[Optional[int]]) -> Optional[int]:
        """Encuentra la última página en el chunk."""
        return next((p for p in reversed(map_chunk) if p is not None), None)

    def _extract_pages_content(
        self, 
        map_chunk: List[Optional[int]], 
        pages_dict: Dict[int, str],
        ids_chunk: List[int]
    ) -> Dict[int, str]:
        """
        Extrae solo la porción de cada página que aparece en este chunk específico.
        
        Args:
            map_chunk: Mapeo de tokens a páginas para este chunk
            pages_dict: Diccionario completo de páginas del documento
            ids_chunk: IDs de tokens de este chunk específico
            
        Returns:
            Diccionario con las páginas y solo la porción de contenido que aparece en este chunk
        """
        pages_content = {}
        
        # Agrupar tokens por página
        for i, page_num in enumerate(map_chunk):
            if page_num is not None:
                if page_num not in pages_content:
                    pages_content[page_num] = []
                pages_content[page_num].append(ids_chunk[i])
        
        # Decodificar solo los tokens correspondientes a cada página en este chunk
        result = {}
        for page_num, token_ids in pages_content.items():
            if token_ids:  # Solo si hay tokens para esta página
                page_text_portion = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                result[page_num] = page_text_portion
        
        return result

    def _calculate_char_positions(
        self,
        all_ids: List[int],
        start: int,
        end: int,
        total_tokens: int,
        full_text: str
    ) -> Tuple[int, int]:
        """
        Calcula las posiciones de caracteres del chunk en el texto completo.
        """
        start_char = (
            len(self.tokenizer.decode(all_ids[:start], skip_special_tokens=True)) 
            if start > 0 else 0
        )
        end_char = (
            len(self.tokenizer.decode(all_ids[:end], skip_special_tokens=True)) 
            if end < total_tokens else len(full_text)
        )
        return start_char, end_char

    # Mantener el método original para compatibilidad hacia atrás
    def chunk_document_pages_from_dict_no_sep(
        self, 
        pages_dict: Dict[int, str],
        chunk_size: int = 512,
        overlap: int = 128,
    ) -> List[Chunk]:
        """
        Método legacy - usar 'chunk' en su lugar.
        """
        document_data = {
            "document_name": "unknown",
            "pages": pages_dict
        }
        return self.chunk(document_data, chunk_size, overlap)
    