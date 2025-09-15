# RAG Pipeline 

## Descripción

Este proyecto implementa un asistente conversacional con RAG construido sobre un corpus documental compuesto por 5 papers relevantes sobre modelos basados en la arquitectura Transformer.

El flujo que se sigue para la implementación de este asistente es:

### Ingesta
1. Se descargan los documentos de Internet
2. Se extrae el texto de los documentos
3. Se "chunkeriza" el texto de los documentos
4. Se generan embeddings de cada chunk
5. Se ingesta en un Qdrant

### Generación
1. Se lanza una pregunta
2. Se convierte la query a embedding
3. Se recuperan los chunks con embeddings más similares al de la query
4. Se rerankean los 25 chunks con mayor similitud
5. Se inyecta los 5 chunks del rerankeo con mayor scoring al LLM para que responda la pregunta

## Configuración del entorno

### Opción 1: Con Poetry

1. **Instalar Poetry** (si no lo tienes):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Clonar el repositorio**:
   ```bash
   git clone <url-del-repo>
   cd achilles
   ```

3. **Instalar dependencias**:
   ```bash
   # Solo dependencias de producción
   poetry install --only main
   
   # O con dependencias de desarrollo (incluye notebooks)
   poetry install
   ```

4. **Activar el entorno virtual**:
   ```bash
   poetry shell
   ```

### Opción 2: Con pip y entorno virtual

1. **Clonar el repositorio**:
   ```bash
   git clone <url-del-repo>
   cd achilles
   ```

2. **Crear y activar entorno virtual**:
   ```bash
   # Crear entorno virtual
   python -m venv venv
   
   # Activar (Linux/Mac)
   source venv/bin/activate
   
   # Activar (Windows)
   venv\Scripts\activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verificar instalación**:
   ```bash
   python -c "import torch; print('✅ PyTorch instalado correctamente')"
   ```

### Requisitos del sistema
- Python 3.13
- Docker (para Qdrant)
- GPU opcional (para acelerar modelos de embeddings y reranker)

## Cómo lanzar

Para levantar la base de datos vectorial Qdrant:

```
docker compose -f etc/docker-compose.yml up -d 
```

Una vez levantado Qdrant, se procede a ingestar los datos:
```
python -m ingest
```

Para ejecutar la inferencia, es necesario crear un archivo `.env` e introducir el API KEY de OpenAI:

```bash
# Crear archivo .env
touch .env

# Agregar las siguientes líneas al archivo .env:
echo "OPENAI_API_KEY=tu_api_key_aqui" >> .env
echo "RERANK=True" >> .env  # Opcional: False para deshabilitarlo
```

O crear manualmente el archivo `.env` con el siguiente contenido:
```
OPENAI_API_KEY=tu_api_key_aqui
RERANK=True
```

Con los datos ingestados y el API KEY de OpenAI introducido, se puede proceder a lanzar la inferencia sobre el archivo de datos `eval.jsonl`:
```
python -m ask
```

Este comando dará como resultado un archivo `results.jsonl` con las respuestas a las preguntas de `eval.jsonl`, los chunks recuperados durante la fase de recuperación de la información, y los chunks referenciados en la respuesta del LLM. 

### Reinicio

En caso de querer "limpiar" la base de datos para volver a lanzar la ingesta:
```
docker compose -f etc/docker-compose.yml down
rm -rf etc/.data
rm -rf data
```

## Decisiones técnicas

Se detallan aquí las decisiones técnicas para la construcción del pipeline del asistente RAG.

### Extracción de texto
Para la extracción del texto de los PDFs se utiliza la librería de Python `PyMuPDF`. El principal motivo para escoger este extractor es su ligereza. A diferencia de otros parsers que pueden ofrecer textos de mayor calidad, `PyMuPDF` destaca por su rapidez y sencillez de uso, permitiendo una fase de extracción de textos ligera.

Además, los documentos seleccionados para el asistente tienen una estructura clara, lo cual es beneficioso para `PyMuPDF`. Como punto negativo, los documentos tienen bastantes tablas, por lo que podría haber sido de ayuda un OCR basado en LLMs que fuese capaz de capturar esa información de forma más estructurada. No obstante, esto hubiese supuesto un procesamiento más lento si decidiesemos usar un modelo local o añadir dependencias de proveedores externos si decidiesemos usar el API de un modelo privativo.

Considerando estos motivos, se llega a la conclusión de que `PyMuPDF` es una buena elección para la tarea que aquí se atañe.

### Chunkizado
Para el chunkizado, se escoge un tamaño de chunk de 512 tokens y 128 de overlap. La decisión de escoger ese tamaño de chunk es que es lo suficientemente grande como para contener información relevante y lo suficientemente pequeño como para que el modelo de embeddings capte el significado semántico especifico del chunk, y que no genere una representación vectorial demasiado generalista. El overlap de 128 tokens es suficiente para mantener contextos y no dejar chunks "a medias" en la búsqueda.

Además, durante la fase de chunkizado se extraen como metadatos las páginas en las que se encuentra el texto de cada chunk, con la intención de aportar esta información al LLM durante la fase de inferencia, aportando más contexto.

### Modelo de embedding

Como modelo de embeddings se selecciona un modelo que puede ejecutarse localmente para evitar sumar fricción a la ejecución con API_KEY's de proveedores externos. El modelo seleccionado es
`Qwen/Qwen3-Embedding-0.6B`. Además de por ser un modelo open source, los principales motivos son:
1. Modelo multilingüe: proporciona representaciones semánticas coherentes entre idiomas. Dado que los documentos utilizados están en inglés y las queries se hacen en español esto es necesario.
2. Buena calidad/tamaño: se trata de un modelo de 600M de parámetros, lo cual está lejos de otros modelos de embeddings más potentes del orden de billones de parámetros, pero hace posible ejecutarlo en local y obtiene scorings altos en varios de los principales benchmarks. 

### Base de datos vectorial
Como base de datos vectorial se escoge Qdrant. Una de las principales razones es una vez más que sea open source y se pueda levantar con tan solo un comando, reduciendo la fricción de la ejecución.

### Reranker
Como modelo de reranker se usa `Alibaba-NLP/gte-multilingual-reranker-base`, el cual se trata de un modelo de 306M de parámetros. Como se ha comentado previamente, se tiene preferencia por un modelo open source por los motivos antes expuestos. Dentro de los modelos open source disponibles, si bien existen alternativas con menos parámetros (~100M), son modelos que o bien tienen una ventana de contexto muy reducida (por lo general, 128 tokens para la query y 384 para el chunk) o bien no son multimodales.

La tarea de reranking es una tarea muy costosa computacionalmente hablando, ya que es necesario calcular el embedding de tanto la query como de los `top_k` chunks inicialmente recuperados. En el caso del modelo seleccionado, ejecutándolo en mi ordenador, los tiempos se disparan hasta 1 minuto por rerankeo, por lo que se ha incluido la posibilidad de deshabilitar el uso del reranker en el parámetro RERANK de `.env`. Para ello, simplemente es necesario configurar `RERANK=False`en dicho archivo. En caso de usarse el reranker, el modelo de embeddings recupera los 25 chunks con mayor similitud y estos 25 son introducidos en el reranker, inyectando los 5 con mayor scoring en el LLM. En caso de no usarse, se proporcionan al LLM los 5 chunks con mayor similitud de la búsqueda con embeddings.

Otra alternativa para reducir los tiempos del reranker hubiese sido usar un modelo alojado en la nube con mayores recursos computacionales. Sin embargo, esto aumentaría la fricción. 

Por lo observado en los resultados, si bien el reranker puede devolver resultados ligeramente mejores que simplemente el modelo de embeddings, con únicamente este último sería suficiente para obtener un buen rankeo en la recuperación de información.

### LLM
Como LLM se escoge `gpt-4.1-2025-04-14`. Aunque esto no siga la intención de evitar usar modelos que requieran de API_KEYs externas, sí que considero importante usar un buen modelo que sea capaz de desenvolverse bien en contextos relativamente largos y con una buena capacidad de seguir instrucciones. Además, el cliente de OpenAI es suficientemente popular como para que no aumente demasiado la fricción a la hora de ejecutar.

Cada inferencia está compuesta por un prompt de sistema (`prompts/system_prompt.txt`) junto con la consulta en lenguaje natural como mensaje de usuario. Además, se dota al LLM de la herramienta `search`. Esta herramienta toma como entrada la consulta y se encarga de devolver los 5 chunks más relevantes a dicha consulta con un formato XML estructurado. Un ejemplo de este formato es:
```xml
<ToolResponse>
  <results query={search_query}>
    <chunk id={chunk_id} document_name={document_name} chunk_index={chunk_index}>
      <page index={page_index}>
        Lorem ipsum dolor sit amet
      </page>
      <page index={page_index}>
        consectetur adipiscing elit
      </page>
    </chunk>
    <chunk id={chunk_id} document_name={document_name} chunk_index={chunk_index}>
      ...
    </chunk>
    ...
  </results>
</ToolResponse>
```
De esta forma, el LLM recibe los chunks de forma estructurada, aportando así un contexto más claro.

La decisión de incluir una herramienta `search` en vez de inyectar directamente los chunks al contexto del LLM es que:
1. La query de búsqueda generada por la herramienta puede ser una reformulación de la query original, por lo que podemos eliminar ruido con esta reformulación.
2. Mejor manejo de interacciones habituales con asistentes. Para interacciones habituales como "Hola" o "Quien eres?" no es necesario recuperar información de los documentos (en este caso), por lo que nos ahorramos la búsqueda vectorial. 

## Métricas

