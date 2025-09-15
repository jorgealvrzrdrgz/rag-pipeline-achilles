import fitz
from pathlib import Path


class Parser:
    def __init__(self):
        pass

    def parse(self, doc_file: str) -> dict[str, any]:
        """
        Extrae el texto de un documento PDF.
        
        Args:
            doc_file: Ruta al archivo PDF a procesar
            
        Returns:
            dict con:
            - "document_name": nombre del archivo sin extensión
            - "pages": dict[int, str] con número de página como clave y texto como valor
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            Exception: Si hay error al procesar el PDF
        """
        file_path = Path(doc_file)
        if not file_path.exists():
            raise FileNotFoundError(f"El archivo no existe: {doc_file}")
        
        if not file_path.suffix.lower() == '.pdf':
            raise ValueError(f"El archivo debe ser un PDF: {doc_file}")
        
        try:
            # Extraer el nombre del documento sin extensión
            document_name = file_path.stem
            
            doc = fitz.open(doc_file)
            
            pages_text = {}
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                pages_text[page_num + 1] = text  # Páginas numeradas desde 1
            
            doc.close()
            
            print(f"Texto extraído de {len(pages_text)} páginas del archivo: {file_path.name}")
            
            return {
                "document_name": document_name,
                "pages": pages_text
            }
            
        except Exception as e:
            raise Exception(f"Error al procesar el PDF {doc_file}: {str(e)}")