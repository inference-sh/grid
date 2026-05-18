import json
import logging
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, TextMeta
from pydantic import Field
from typing import List, Literal, Optional

from .embedding_utils import resolve_texts, chunk_text

logger = logging.getLogger(__name__)

MODEL_MAX_TOKENS = 32768

PROMPT_TEMPLATES = {
    "web_search_query": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
    "sts_query": "Instruct: Retrieve semantically similar text\nQuery: ",
    "bitext_query": "Instruct: Retrieve parallel sentences in a different language\nQuery: ",
}


class AppSetup(BaseAppInput):
    model_id: Literal[
        "microsoft/harrier-oss-v1-270m",
        "microsoft/harrier-oss-v1-0.6b",
        "microsoft/harrier-oss-v1-27b",
    ] = Field(
        default="microsoft/harrier-oss-v1-0.6b",
        description="Model variant: 270m (640d), 0.6b (1024d), or 27b (5376d)",
    )


class AppInput(BaseAppInput):
    texts: Optional[List[str]] = Field(
        default=None,
        json_schema_extra={"x-promoted": True},
        description="Texts to embed (one embedding per text).",
    )
    files: Optional[List[File]] = Field(
        default=None,
        json_schema_extra={"x-promoted": True},
        description="Files containing texts to embed. Supports .txt (one text per line), .jsonl (one JSON string per line), or .json (array of strings).",
    )
    model_config = {
        "json_schema_extra": {
            "anyOf": [
                {"properties": {"texts": {"not": {"type": "null"}}}},
                {"properties": {"files": {"not": {"type": "null"}}}}
            ]
        }
    }
    instruction: Optional[str] = Field(
        default=None,
        description="Task instruction for queries (e.g. 'Given a web search query, retrieve relevant passages that answer the query'). Only needed for query-side encoding, not for documents.",
    )
    prompt_name: Optional[str] = Field(
        default=None,
        description="Pre-configured prompt name (e.g. 'web_search_query', 'sts_query', 'bitext_query'). Alternative to providing a custom instruction.",
    )
    chunk_strategy: Optional[Literal["fixed", "recursive"]] = Field(
        default=None,
        description="Chunking strategy. 'fixed': split by token count. 'recursive': split at paragraph/sentence/word boundaries within token budget. None: no chunking.",
    )
    chunk_size: int = Field(
        default=512,
        description="Target chunk size in tokens. Clamped to model max (32768). Only used when chunk_strategy is set.",
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between chunks in tokens. Only used when chunk_strategy is set.",
    )


class AppOutput(BaseAppOutput):
    embeddings: List[File] = Field(default_factory=list, description="JSON files containing embeddings, one per input text")
    inline_embeddings: Optional[List[dict]] = Field(default=None, description="Inline embeddings when output is small enough to skip file upload")
    dimension: int = Field(description="Embedding dimension")
    count: int = Field(description="Total number of embeddings across all files")


class App(BaseApp):
    async def setup(self, config: AppSetup):
        import torch
        from accelerate import Accelerator

        accelerator = Accelerator()
        self._device = accelerator.device
        self._use_raw = False

        logger.info(f"Loading model {config.model_id} on {self._device}")

        # 27b is gemma3_text architecture — SentenceTransformer fails on it because
        # Gemma3Processor requires an image_processor that doesn't exist for text-only.
        # Loading 27B params only to fail wastes ~90s, so skip straight to raw transformers.
        # 270m and 0.6b are qwen3 and work fine with SentenceTransformer.
        use_raw = "27b" in config.model_id

        if not use_raw:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                config.model_id,
                device=str(self._device),
                model_kwargs={"dtype": "auto"},
            )
            self._tok = self.model.tokenizer
        else:
            self._use_raw = True
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(config.model_id)
            self._tok = self._tokenizer
            self._raw_model = AutoModel.from_pretrained(config.model_id, torch_dtype=torch.bfloat16)
            self._raw_model.eval()
            self._raw_model.to(self._device)

        logger.info("Model loaded")

    async def run(self, input_data: AppInput) -> AppOutput:
        texts = resolve_texts(input_data.texts, input_data.files)
        chunk_size = min(input_data.chunk_size, MODEL_MAX_TOKENS)
        overlap = min(input_data.chunk_overlap, chunk_size // 2)

        logger.info(f"Encoding {len(texts)} texts (chunk_strategy={input_data.chunk_strategy}, chunk_size={chunk_size})")

        output_files = []
        total_count = 0
        total_tokens = 0
        dimension = None

        INLINE_THRESHOLD = 64 * 1024  # 64KB — skip file upload for small outputs

        all_results = []
        for i, text in enumerate(texts):
            if input_data.chunk_strategy:
                chunks = chunk_text(text, input_data.chunk_strategy, chunk_size, overlap, self._tok)
            else:
                chunks = chunk_text(text, "none", chunk_size, overlap, self._tok)

            chunk_texts = [c["text"] for c in chunks]
            total_tokens += sum(c["end_token"] - c["start_token"] for c in chunks)

            embeddings = self._embed(chunk_texts, input_data.instruction, input_data.prompt_name)
            dimension = len(embeddings[0])
            total_count += len(embeddings)

            file_data = {"embeddings": embeddings}
            if input_data.chunk_strategy:
                file_data["chunks"] = [{
                    "index": j,
                    "text": c["text"],
                    "start_token": c["start_token"],
                    "end_token": c["end_token"],
                    "start_char": c["start_char"],
                    "end_char": c["end_char"],
                } for j, c in enumerate(chunks)]

            all_results.append(file_data)

        # Return inline when small enough, otherwise upload files
        estimated_size = len(json.dumps(all_results))
        if estimated_size <= INLINE_THRESHOLD:
            logger.info(f"Encoded {total_count} embeddings inline ({estimated_size} bytes), dimension={dimension}")
            return AppOutput(
                inline_embeddings=all_results,
                dimension=dimension,
                count=total_count,
                output_meta=OutputMeta(
                    inputs=[TextMeta(tokens=total_tokens)],
                ),
            )

        for i, file_data in enumerate(all_results):
            path = f"/tmp/embeddings_{i}.json"
            with open(path, "w") as f:
                json.dump(file_data, f)
            output_files.append(File(path=path))

        logger.info(f"Encoded {total_count} embeddings across {len(output_files)} files, dimension={dimension}")
        return AppOutput(
            embeddings=output_files,
            dimension=dimension,
            count=total_count,
            output_meta=OutputMeta(
                inputs=[TextMeta(tokens=total_tokens)],
            ),
        )

    def _embed(self, texts: List[str], instruction: Optional[str], prompt_name: Optional[str]) -> list:
        if self._use_raw:
            return self._embed_raw(texts, instruction, prompt_name)
        return self._embed_st(texts, instruction, prompt_name)

    def _embed_st(self, texts: List[str], instruction: Optional[str], prompt_name: Optional[str]) -> list:
        encode_kwargs = {}
        if prompt_name:
            encode_kwargs["prompt_name"] = prompt_name
        elif instruction:
            encode_kwargs["prompt"] = f"Instruct: {instruction}\nQuery: "

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            **encode_kwargs,
        )
        return embeddings.tolist()

    def _embed_raw(self, texts: List[str], instruction: Optional[str], prompt_name: Optional[str]) -> list:
        import torch
        import torch.nn.functional as F

        processed = list(texts)
        if prompt_name and prompt_name in PROMPT_TEMPLATES:
            prefix = PROMPT_TEMPLATES[prompt_name]
            processed = [prefix + t for t in processed]
        elif instruction:
            prefix = f"Instruct: {instruction}\nQuery: "
            processed = [prefix + t for t in processed]

        batch_dict = self._tokenizer(
            processed, max_length=MODEL_MAX_TOKENS, padding=True, truncation=True, return_tensors="pt",
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
