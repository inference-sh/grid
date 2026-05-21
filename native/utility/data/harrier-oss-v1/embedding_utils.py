"""Shared utilities for embedding apps — chunking, text resolution, file parsing.

Usage:
    Symlink into each embedding app directory and import with relative import:
        ln -s ../embedding_utils.py my-app/embedding_utils.py
        # in inference.py: from .embedding_utils import resolve_texts, chunk_text
    Requires __init__.py in the app dir (from .inference import App).

Tokenizer contract:
    chunk_text() accepts any HuggingFace PreTrainedTokenizer. It uses:
        - tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        - tokenizer.encode(text)
    Any model with a standard HF tokenizer works.

Output contract:
    All embedding apps should return List[File] with JSON files containing:
        {"embeddings": [[float, ...], ...]}                    # no chunking
        {"embeddings": [...], "chunks": [{text, start_token,   # with chunking
          end_token, start_char, end_char, index}, ...]}
"""

import json
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def resolve_texts(texts: Optional[List[str]], files: Optional[list]) -> List[str]:
    """Merge texts from both texts array and uploaded files. Requires at least one source."""
    result = []
    if texts:
        result.extend(texts)
    if files:
        for f in files:
            result.extend(_read_text_file(f.path))
    if not result:
        raise ValueError("Provide at least one of texts or files")
    return result


def _read_text_file(path: str) -> List[str]:
    """Read texts from a file.

    - .json: array of strings, each is a separate text
    - .jsonl: one JSON string per line, each is a separate text
    - Everything else (.txt, .md, etc.): entire file is ONE text
      Use chunk_strategy to split documents into chunks.
    """
    with open(path, "r") as f:
        content = f.read()

    if path.endswith(".json"):
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain an array of strings")
        return [str(t) for t in data]

    if path.endswith(".jsonl"):
        return [str(json.loads(line)) for line in content.strip().splitlines() if line.strip()]

    # Whole file is one text — use chunking to split if needed
    return [content.strip()]


def chunk_text(text: str, strategy: str, chunk_size: int, overlap: int, tokenizer) -> list:
    """Chunk text using the given strategy. Returns list of chunk dicts with text, token, and char offsets."""
    if strategy == "fixed":
        return _chunk_fixed(text, chunk_size, overlap, tokenizer)
    elif strategy == "recursive":
        return _chunk_recursive(text, chunk_size, overlap, tokenizer)
    return _no_chunk(text, tokenizer)


def _no_chunk(text: str, tokenizer) -> list:
    """Single chunk — the whole text."""
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    return [{
        "text": text,
        "start_token": 0,
        "end_token": len(encoded["input_ids"]),
        "start_char": 0,
        "end_char": len(text),
    }]


def _chunk_fixed(text: str, chunk_size: int, overlap: int, tokenizer) -> list:
    """Split text into fixed-size token chunks with overlap."""
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    if len(tokens) <= chunk_size:
        return [{"text": text, "start_token": 0, "end_token": len(tokens), "start_char": 0, "end_char": len(text)}]

    chunks = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        start_char = offsets[start][0]
        end_char = offsets[end - 1][1]

        chunks.append({
            "text": text[start_char:end_char],
            "start_token": start,
            "end_token": end,
            "start_char": start_char,
            "end_char": end_char,
        })
        if end >= len(tokens):
            break
        start += step

    return chunks


def _chunk_recursive(text: str, chunk_size: int, overlap: int, tokenizer) -> list:
    """Split at natural boundaries (paragraph > sentence > word) within token budget."""
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    if len(tokens) <= chunk_size:
        return [{"text": text, "start_token": 0, "end_token": len(tokens), "start_char": 0, "end_char": len(text)}]

    break_points = _find_break_points(text, offsets, chunk_size, tokenizer)

    chunks = []
    for bp_idx, (tok_start, tok_end) in enumerate(break_points):
        actual_start = max(0, tok_start - overlap) if (overlap > 0 and bp_idx > 0) else tok_start
        start_char = offsets[actual_start][0]
        end_char = offsets[tok_end - 1][1]

        chunks.append({
            "text": text[start_char:end_char],
            "start_token": actual_start,
            "end_token": tok_end,
            "start_char": start_char,
            "end_char": end_char,
        })

    return chunks if chunks else _no_chunk(text, tokenizer)


def _find_break_points(text: str, offsets: list, chunk_size: int, tokenizer) -> list:
    """Find token-level break points at natural text boundaries."""
    num_tokens = len(offsets)
    separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]
    breaks = []
    start = 0

    while start < num_tokens:
        if start + chunk_size >= num_tokens:
            breaks.append((start, num_tokens))
            break

        best_break = None
        search_end = min(start + chunk_size, num_tokens)
        search_start_char = offsets[start][0]
        search_end_char = offsets[search_end - 1][1]
        chunk_text = text[search_start_char:search_end_char]

        for sep in separators:
            sep_pos = chunk_text.rfind(sep)
            if sep_pos > 0:
                target_char = search_start_char + sep_pos + len(sep)
                for t in range(start, search_end):
                    if offsets[t][0] >= target_char:
                        if t > start:
                            best_break = t
                        break
                if best_break is not None:
                    break

        if best_break is None or best_break <= start:
            best_break = min(start + chunk_size, num_tokens)

        breaks.append((start, best_break))
        start = best_break

    return breaks
