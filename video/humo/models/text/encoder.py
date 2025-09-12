import os
from dataclasses import dataclass
from typing import List, Optional, Union
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPTextModel,
    CLIPTokenizerFast,
    T5EncoderModel,
    T5TokenizerFast,
)
from transformers.tokenization_utils_base import BatchEncoding

from common.fs import download_and_extract
from common.logger import get_logger

logger = get_logger(__name__)

MODEL_TYPES = {
    "clip": (CLIPTokenizerFast, CLIPTextModel),
    "t5": (T5TokenizerFast, T5EncoderModel),
    "llm14b": (AutoTokenizer, AutoModelForCausalLM),
}


@dataclass
class TextEncoderOutput:
    embeddings: Union[torch.FloatTensor, List[torch.FloatTensor]]
    masks: Union[torch.BoolTensor, List[torch.BoolTensor]]
    pooled: Optional[Union[torch.FloatTensor, List[torch.FloatTensor]]]


class TextEncoder(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.tokenizers = []
        self.models = nn.ModuleList([])

        # Disable tokenizer parallelism since we already use distributed training.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        for model in config.models:
            tokenizer_cls, model_cls = MODEL_TYPES[model.type]
            path = download_and_extract(model.path)
            max_length = model.max_length

            if model.type == "llm14b":
                tokenizer = tokenizer_cls.from_pretrained(
                    path,
                    model_max_length=max_length,
                    use_fast=False,
                    trust_remote_code=True,
                    padding_side="right",
                    truncation_side="right",
                    add_eod_token=True,
                )
                tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
                model = model_cls.from_pretrained(path, trust_remote_code=True, bf16=True)
            else:
                tokenizer = tokenizer_cls.from_pretrained(path, model_max_length=max_length)
                model = model_cls.from_pretrained(path, torch_dtype=torch.bfloat16)
            self.tokenizers.append(tokenizer)
            self.models.append(model)

    def forward(self, text: Union[str, List[str]]) -> TextEncoderOutput:
        embeddings, masks, pooled = [], [], []

        for encoder_config, tokenizer, model in zip(
            self.config.models, self.tokenizers, self.models
        ):
            if encoder_config.type == "llm14b":
                use_mask = encoder_config.get("mask", True)
                tokens = tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                ).to(model.device)
                token_ids = tokens["input_ids"]
                attention_mask = tokens["attention_mask"]
                num_tokens = attention_mask.sum(dim=1)
                range_ids = torch.arange(len(token_ids), device=token_ids.device, dtype=torch.long)
                token_ids[range_ids, num_tokens.clamp(max=token_ids.size(1) - 1)] = (
                    tokenizer.pad_token_id
                )
                attention_mask[range_ids, num_tokens.clamp(max=token_ids.size(1) - 1)] = 1
                tokens = BatchEncoding({"input_ids": token_ids, "attention_mask": attention_mask})
                output = model.transformer(
                    input_ids=tokens.input_ids,
                    attention_mask=attention_mask if use_mask else None,
                    output_hidden_states=False,
                    use_cache=False,
                )
                emb = output.last_hidden_state  # batch_size, num_tokens, feat_dim
                # emb *= tokens.attention_mask.unsqueeze(-1)

                embeddings.append(emb)
                masks.append(
                    tokens.attention_mask.bool() if use_mask else tokens.attention_mask > -1
                )

            else:
                # Tokenizer
                tokens = tokenizer(
                    text=text,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                # Encoder
                use_mask = encoder_config.get("mask", True)
                input_ids = tokens.input_ids.to(model.device)
                attention_mask = tokens.attention_mask.to(model.device)
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask if use_mask else None,
                    output_hidden_states=True,
                )

                # Save embeddings from the defined layer.
                layer = encoder_config.get("layer", "last")
                if layer == "last":
                    embeddings.append(output.last_hidden_state)
                elif layer == "penultimate":
                    embeddings.append(model.text_model.final_layer_norm(output.hidden_states[-2]))
                elif layer == "penultimate_nonorm":
                    embeddings.append(output.hidden_states[-2])
                else:
                    raise NotImplementedError(f"Unknown layer type: {layer}.")

                # Save masks
                masks.append(attention_mask.bool() if use_mask else attention_mask > -1)

                # Save pooled output if available.
                if hasattr(output, "pooler_output"):
                    pooled.append(output.pooler_output)

            output_config = self.config.get("output") or OmegaConf.create()
            embedding_output_type = output_config.get("embedding_and_mask", "undefined")
            pooled_output_type = output_config.get("pooled", "undefined")

            # Select or merge embeddings and mask if needed.
            if embedding_output_type == "undefined" and len(self.models) == 1:
                embeddings = embeddings[0]
                masks = masks[0]
            elif embedding_output_type == "channel_concat":
                embeddings = torch.cat(embeddings, dim=-1)
                masks = sum(masks).bool()
            elif embedding_output_type == "last":
                embeddings = embeddings[-1]
                masks = masks[-1]
            else:
                raise NotImplementedError(f"output.embedding_and_mask: {embedding_output_type}")

            # Select or merge pooled output if needed.
            if pooled_output_type == "undefined":
                pooled = None
            elif pooled_output_type == "channel_concat":
                pooled = torch.cat(pooled, dim=-1)
            elif pooled_output_type == "first":
                pooled = pooled[0]
            elif pooled_output_type == "last":
                pooled = pooled[-1]
            else:
                raise NotImplementedError(f"output.pooled: {pooled_output_type}")

        # Return final results.
        return TextEncoderOutput(embeddings, masks, pooled)
