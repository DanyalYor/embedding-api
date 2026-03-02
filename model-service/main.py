from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction

class EmbedService():
    """"""

    def __init__(self, model_name: str, device: str = "cpu")
        self.device = device
        self.model_name = model_name

        provider = (
            "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        )

        self.model = ORTModelForFeatureExtraction.from_pretrained(
            model_name,
            provider=provider,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
         
    def embed(self, request):
        texts = list(request.texts)

        if not texts:
            return []

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(axis=1)

        return embeddings
