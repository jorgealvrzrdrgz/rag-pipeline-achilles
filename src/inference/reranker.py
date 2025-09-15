import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.models import Chunk

class Reranker:

    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True,
            torch_dtype=torch.float16
        )
        self.model.eval()

    def rerank(self, query, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        pairs = [[query, chunk.text] for chunk in chunks]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=1024)
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
        return [chunks[i] for i in torch.sort(scores, descending=True).indices.tolist()][:top_k]
