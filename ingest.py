import os

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from src.ingest.pdf_downloader import PdfDownloader
from src.ingest.parser import Parser
from src.ingest.chunker import Chunker
from src.shared.embeddings import EmbeddingsService
from src.models import Chunk
from src.shared.qdrant_repository import QdrantRepository
from src.shared.environment import Environment


def main():
    """
    Función principal para el proceso de ingesta de documentos.
    
    Este proceso:
    1. Descarga los PDFs desde las URLs configuradas
    2. Parsea los PDFs para extraer el texto
    3. Divide el texto en chunks
    4. Genera embeddings para cada chunk
    5. Almacena los chunks con sus embeddings en Qdrant
    """
    print("Iniciando proceso de ingesta de documentos...")
    
    # Inicializar configuración y servicios
    environment = Environment()
    
    print("Cargando modelo de embeddings...")
    model = AutoModel.from_pretrained(environment.EMBEDDINGS_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(environment.EMBEDDINGS_TOKENIZER)
    embeddings_service = EmbeddingsService(model, tokenizer)
    qdrant_repository = QdrantRepository(environment.QDRANT_URL)
    
    # Descargar PDFs
    print("Descargando PDFs...")
    pdfs_urls = environment.PDFS_URLS
    pdf_downloader = PdfDownloader()
    
    for pdf_url in pdfs_urls:
        pdf_downloader.download(pdf_url)
    
    print(f"Descargados {len(pdfs_urls)} PDFs")
    
    # Procesar PDFs y generar chunks
    print("Procesando PDFs y generando chunks...")
    all_chunks: list[Chunk] = []
    
    for pdf in os.listdir("data"):
        if not pdf.endswith('.pdf'):
            continue
            
        print(f"Procesando {pdf}...")
        parser = Parser()
        result = parser.parse(os.path.join("data", pdf))
        
        chunker = Chunker(tokenizer)
        chunks = chunker.chunk(result)
        all_chunks.extend(chunks)
    
    print(f"Generados {len(all_chunks)} chunks en total")
    
    # Generar embeddings y almacenar en Qdrant
    print("Generando embeddings y almacenando en Qdrant...")
    for chunk in tqdm(all_chunks, desc="Procesando chunks"):
        embedding = embeddings_service.get_embeddings(chunk.text)
        chunk.embedding = embedding
        qdrant_repository.upsert(environment.QDRANT_COLLECTION, chunk)
    
    print("¡Proceso de ingesta completado exitosamente!")


if __name__ == "__main__":
    main()
