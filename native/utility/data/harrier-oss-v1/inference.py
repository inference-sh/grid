import json
import logging
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, TextMeta
from pydantic import Field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Pre-configured prompts matching config_sentence_transformers.json
PROMPT_TEMPLATES = {
    "web_search_query": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
    "sts_query": "Instruct: Retrieve semantically similar text\nQuery: ",
    "bitext_query": "Instruct: Retrieve parallel sentences in a different language\nQuery: ",
}


class AppSetup(BaseAppInput):
    model_id: str = Field(
        default="microsoft/harrier-oss-v1-0.6b",
        description="Model variant to load: microsoft/harrier-oss-v1-270m (270M params, 640d), microsoft/harrier-oss-v1-0.6b (0.6B params, 1024d), or microsoft/harrier-oss-v1-27b (27B params, 5376d)",
    )


class AppInput(BaseAppInput):
    texts: List[str] = Field(description="Texts to embed")
    instruction: Optional[str] = Field(
        default=None,
        description="Task instruction for queries (e.g. 'Given a web search query, retrieve relevant passages that answer the query'). Only needed for query-side encoding, not for documents.",
    )
    prompt_name: Optional[str] = Field(
        default=None,
        description="Pre-configured prompt name (e.g. 'web_search_query', 'sts_query', 'bitext_query'). Alternative to providing a custom instruction.",
    )


class AppOutput(BaseAppOutput):
    embeddings: File = Field(description="JSON file containing embedding vectors")
    dimension: int = Field(description="Embedding dimension")
    count: int = Field(description="Number of embeddings")


class App(BaseApp):
    async def setup(self, config: AppSetup):
        import torch
        from accelerate import Accelerator

        accelerator = Accelerator()
        self._device = accelerator.device
        self._use_raw = False

        logger.info(f"Loading model {config.model_id} on {self._device}")

        # Try SentenceTransformer first (works for 270m, 0.6b).
        # Falls back to raw transformers for 27b where AutoProcessor
        # fails because gemma3_text inherits gemma3's Gemma3Processor
        # which requires an image_processor that doesn't exist.
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                config.model_id,
                device=str(self._device),
                model_kwargs={"dtype": "auto"},
            )
        except Exception as e:
            if "image processor" not in str(e).lower() and "image_processor" not in str(e).lower():
                raise
            logger.info(f"SentenceTransformer failed ({e}), falling back to raw transformers")
            self._use_raw = True
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(config.model_id)
            self._raw_model = AutoModel.from_pretrained(config.model_id, torch_dtype=torch.bfloat16)
            self._raw_model.eval()
            self._raw_model.to(self._device)

        logger.info("Model loaded")

    async def run(self, input_data: AppInput) -> AppOutput:
        logger.info(f"Encoding {len(input_data.texts)} texts")

        if self._use_raw:
            embedding_list = self._run_raw(input_data)
        else:
            embedding_list = self._run_st(input_data)

        dimension = len(embedding_list[0])
        count = len(embedding_list)
        total_tokens = sum(len(t.split()) for t in input_data.texts)

        # Write embeddings to JSON file instead of inline response
        output_path = "/tmp/embeddings.json"
        with open(output_path, "w") as f:
            json.dump(embedding_list, f)

        logger.info(f"Encoded {count} texts, dimension={dimension}")

        return AppOutput(
            embeddings=File(path=output_path),
            dimension=dimension,
            count=count,
            output_meta=OutputMeta(
                inputs=[TextMeta(tokens=total_tokens)],
            ),
        )

    def _run_st(self, input_data: AppInput) -> list:
        encode_kwargs = {}
        if input_data.prompt_name:
            encode_kwargs["prompt_name"] = input_data.prompt_name
        elif input_data.instruction:
            encode_kwargs["prompt"] = f"Instruct: {input_data.instruction}\nQuery: "

        embeddings = self.model.encode(
            input_data.texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            **encode_kwargs,
        )
        return embeddings.tolist()

    def _run_raw(self, input_data: AppInput) -> list:
        import torch
        import torch.nn.functional as F

        # Apply instruction prefix if provided
        texts = list(input_data.texts)
        if input_data.prompt_name and input_data.prompt_name in PROMPT_TEMPLATES:
            prefix = PROMPT_TEMPLATES[input_data.prompt_name]
            texts = [prefix + t for t in texts]
        elif input_data.instruction:
            prefix = f"Instruct: {input_data.instruction}\nQuery: "
            texts = [prefix + t for t in texts]

        batch_dict = self._tokenizer(
            texts, max_length=32768, padding=True, truncation=True, return_tensors="pt",
        )
        batch_dict = {k: v.to(self._device) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = self._raw_model(**batch_dict)

        # Last-token pooling
        attention_mask = batch_dict["attention_mask"]
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            embeddings = outputs.last_hidden_state[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = outputs.last_hidden_state.shape[0]
            embeddings = outputs.last_hidden_state[
                torch.arange(batch_size, device=self._device), sequence_lengths
            ]

        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.float().cpu().tolist()

    async def unload(self):
        import torch

        for attr in ("model", "_raw_model"):
            if hasattr(self, attr):
                delattr(self, attr)
        if hasattr(self, "_tokenizer"):
            del self._tokenizer
        torch.cuda.empty_cache()
