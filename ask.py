import json

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from openai import OpenAI

from src.inference.ask_service import AskService
from src.shared.embeddings import EmbeddingsService
from src.shared.qdrant_repository import QdrantRepository
from src.inference.search import SearchTool
from src.inference.reranker import Reranker
from src.shared.environment import Environment


def main():
    """
    Función principal para el proceso de evaluación de consultas.
    
    Este proceso:
    1. Carga la configuración y herramientas necesarias
    2. Inicializa los servicios de embeddings, búsqueda y reranking
    3. Procesa las consultas del dataset de evaluación
    4. Genera respuestas usando el servicio AskService
    5. Guarda los resultados en un archivo JSONL
    """
    print("Iniciando proceso de evaluación de consultas...")
    
    # Inicializar configuración
    environment = Environment()
    
    # Cargar herramientas y prompts
    print("Cargando herramientas y prompts...")
    with open("prompts/tools.json", "r") as f:
        tools = json.load(f)
    
    with open("prompts/system_prompt.txt", "r") as f:
        system_prompt = f.read()
    
    # Cargar dataset de evaluación
    print("Cargando dataset de evaluación...")
    data = []
    with open("eval.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"Cargadas {len(data)} consultas para evaluar")
    
    # Inicializar modelos y servicios
    print("Inicializando modelos y servicios...")
    model = AutoModel.from_pretrained(environment.EMBEDDINGS_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(environment.EMBEDDINGS_TOKENIZER)
    
    reranker = Reranker(model_name=environment.RERANKER_MODEL)
    embeddings_service = EmbeddingsService(model, tokenizer)
    qdrant_repository = QdrantRepository(environment.QDRANT_URL)
    openai_client = OpenAI(api_key=environment.OPENAI_API_KEY)
    
    # Configurar herramienta de búsqueda (con o sin reranking)
    rerank = environment.RERANK
    if rerank:
        print("Configurando búsqueda con reranking...")
        search_tool = SearchTool(qdrant_repository, embeddings_service, reranker)
    else:
        print("Configurando búsqueda sin reranking...")
        search_tool = SearchTool(qdrant_repository, embeddings_service)
    
    # Inicializar servicio de consultas
    ask_service = AskService(openai_client, system_prompt, tools, search_tool)
    
    # Procesar consultas
    print("Procesando consultas...")
    results = []
    for item in tqdm(data, desc="Evaluando consultas"):
        query = item["query"]
        response = ask_service.ask(query)
        results.append(response)
    
    # Guardar resultados
    print("Guardando resultados...")
    results = [result.model_dump() for result in results]
    
    # Determinar el nombre del archivo basado en si se usó reranking
    output_filename = "results.jsonl" if rerank else "results_no_reranker.jsonl"
    
    with open(output_filename, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"¡Proceso completado! Se procesaron {len(results)} consultas.")
    print(f"Resultados guardados en '{output_filename}'")


if __name__ == "__main__":
    main()