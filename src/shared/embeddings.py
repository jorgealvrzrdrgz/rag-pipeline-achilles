import torch
import torch.nn.functional as F
from torch import Tensor


class EmbeddingsService:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_embeddings(self, text: str) -> list[float]:
        tokens = self.tokenizer(text, return_tensors="pt")
        output = self.model(**tokens)
        embeddings = self.last_token_pool(output.last_hidden_state, tokens["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings[0].tolist()
    