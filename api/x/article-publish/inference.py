import os
import re
import uuid
import logging
import requests
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional, List, Tuple, Dict
from .x_helper import upload_file, get_content_type, get_media_category, raise_api_error, format_rate_limit_from_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Markdown → DraftJS converter
# ---------------------------------------------------------------------------

def _key():
    return uuid.uuid4().hex[:5]


def _parse_inline(text: str) -> Tuple[str, list, list]:
    """Parse inline markdown into clean text, style ranges, and link spans.

    Returns (clean_text, inline_style_ranges, links) where each link is
    (offset, length, url) — caller assigns entity keys.
    """
    spans = []
    used = set()

    def overlaps(start, end):
        return any(s <= start < e or s < end <= e for s, e in used)

    # Inline code: X has no code style, so strip backticks but keep text unstyled
    for m in re.finditer(r'`([^`]+)`', text):
        if not overlaps(m.start(), m.end()):
            spans.append((m.start(), m.end(), m.group(1), None))
            used.add((m.start(), m.end()))

    for m in re.finditer(r'\*\*(.+?)\*\*', text):
        if not overlaps(m.start(), m.end()):
            spans.append((m.start(), m.end(), m.group(1), 'bold'))
            used.add((m.start(), m.end()))

    for m in re.finditer(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', text):
        if not overlaps(m.start(), m.end()):
            spans.append((m.start(), m.end(), m.group(1), 'italic'))
            used.add((m.start(), m.end()))

    for m in re.finditer(r'~~(.+?)~~', text):
        if not overlaps(m.start(), m.end()):
            spans.append((m.start(), m.end(), m.group(1), 'strikethrough'))
            used.add((m.start(), m.end()))

    for m in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', text):
        if not overlaps(m.start(), m.end()):
            spans.append((m.start(), m.end(), m.group(1), ('LINK', m.group(2))))
            used.add((m.start(), m.end()))

    if not spans:
        return text, [], []

    spans.sort(key=lambda x: x[0])

    parts = []
    styles = []
    links = []
    prev = 0
    pos = 0

    for start, end, content, style in spans:
        before = text[prev:start]
        parts.append(before)
        pos += len(before)

        cstart = pos
        parts.append(content)
        pos += len(content)

        if isinstance(style, tuple):
            links.append((cstart, len(content), style[1]))
        elif style is not None:
            styles.append({'style': style, 'offset': cstart, 'length': len(content)})

        prev = end

    parts.append(text[prev:])
    return ''.join(parts), styles, links


def _make_text_block(text: str, block_type: str, entities: list) -> dict:
    """Create a text block with inline formatting, appending any new entities."""
    clean, styles, links = _parse_inline(text)

    block = {'text': clean, 'type': block_type, 'key': _key()}
    if styles:
        block['inline_style_ranges'] = styles

    if links:
        eranges = []
        for offset, length, url in links:
            ek = len(entities)
            eranges.append({'key': ek, 'offset': offset, 'length': length})
            entities.append({
                'key': str(ek),
                'value': {'type': 'link', 'mutability': 'mutable', 'data': {'url': url}},
            })
        block['entity_ranges'] = eranges

    return block


def _make_atomic(entity_value: dict, entities: list) -> dict:
    """Create an atomic block referencing a new entity."""
    ek = len(entities)
    entities.append({'key': str(ek), 'value': entity_value})
    return {
        'text': ' ',
        'type': 'atomic',
        'key': _key(),
        'entity_ranges': [{'key': ek, 'offset': 0, 'length': 1}],
    }


def markdown_to_draftjs(
    md: str,
    media_map: Optional[Dict[int, Tuple[str, str]]] = None,
) -> dict:
    """Convert markdown to X Articles DraftJS content_state.

    media_map: {file_index: (media_id, media_category)} for uploaded files.
    """
    media_map = media_map or {}
    blocks = []
    entities = []
    lines = md.split('\n')
    i = 0
    para_buf = []

    def flush_para():
        if not para_buf:
            return
        text = ' '.join(para_buf)
        blocks.append(_make_text_block(text, 'unstyled', entities))
        para_buf.clear()

    while i < len(lines):
        line = lines[i]

        # --- fenced code block ---
        code_match = re.match(r'^```(\w*)', line)
        if code_match:
            flush_para()
            lang = code_match.group(1)
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1  # skip closing ```
            code_body = '\n'.join(code_lines)
            md_str = f'```{lang}\n{code_body}\n```' if lang else f'```\n{code_body}\n```'
            blocks.append(_make_atomic(
                {'type': 'markdown', 'mutability': 'immutable', 'data': {'markdown': md_str}},
                entities,
            ))
            continue

        # --- GFM table (lines starting with |) ---
        if re.match(r'^\|', line):
            flush_para()
            table_lines = []
            while i < len(lines) and re.match(r'^\|', lines[i]):
                table_lines.append(lines[i])
                i += 1
            blocks.append(_make_atomic(
                {'type': 'markdown', 'mutability': 'immutable', 'data': {'markdown': '\n'.join(table_lines)}},
                entities,
            ))
            continue

        # --- horizontal rule ---
        if re.match(r'^(---+|___+|\*\*\*+)\s*$', line):
            flush_para()
            blocks.append(_make_atomic(
                {'type': 'divider', 'mutability': 'immutable', 'data': {}},
                entities,
            ))
            i += 1
            continue

        # --- image: ![alt](file:N) ---
        img_match = re.match(r'^!\[([^\]]*)\]\(file:(\d+)\)\s*$', line)
        if img_match:
            flush_para()
            idx = int(img_match.group(2))
            if idx in media_map:
                mid, mcat = media_map[idx]
                blocks.append(_make_atomic(
                    {
                        'type': 'image', 'mutability': 'immutable',
                        'data': {'media_items': [{'media_id': mid, 'media_category': mcat}]},
                    },
                    entities,
                ))
            else:
                logger.warning(f"Media file:{idx} not found in uploads, skipping image")
            i += 1
            continue

        # --- embedded tweet: {{tweet:ID}} ---
        tweet_match = re.match(r'^\{\{tweet:(\d+)\}\}\s*$', line)
        if tweet_match:
            flush_para()
            blocks.append(_make_atomic(
                {'type': 'post', 'mutability': 'immutable', 'data': {'post_id': tweet_match.group(1)}},
                entities,
            ))
            i += 1
            continue

        # --- latex block: $$...$$ ---
        if line.strip().startswith('$$') and line.strip().endswith('$$') and len(line.strip()) > 4:
            flush_para()
            latex = line.strip()[2:-2].strip()
            blocks.append(_make_atomic(
                {'type': 'latex', 'mutability': 'immutable', 'data': {}},
                entities,
            ))
            # latex text goes in the block text, not entity data
            blocks[-1]['text'] = latex
            i += 1
            continue

        # --- headers ---
        h_match = re.match(r'^(#{1,3})\s+(.+)', line)
        if h_match:
            flush_para()
            level = len(h_match.group(1))
            htype = {1: 'header-one', 2: 'header-two', 3: 'header-three'}[level]
            blocks.append(_make_text_block(h_match.group(2), htype, entities))
            i += 1
            continue

        # --- blockquote ---
        bq_match = re.match(r'^>\s?(.*)', line)
        if bq_match:
            flush_para()
            blocks.append(_make_text_block(bq_match.group(1), 'blockquote', entities))
            i += 1
            continue

        # --- unordered list ---
        ul_match = re.match(r'^[\-\*]\s+(.+)', line)
        if ul_match:
            flush_para()
            blocks.append(_make_text_block(ul_match.group(1), 'unordered-list-item', entities))
            i += 1
            continue

        # --- ordered list ---
        ol_match = re.match(r'^\d+\.\s+(.+)', line)
        if ol_match:
            flush_para()
            blocks.append(_make_text_block(ol_match.group(1), 'ordered-list-item', entities))
            i += 1
            continue

        # --- blank line = paragraph separator ---
        if not line.strip():
            flush_para()
            i += 1
            continue

        # --- regular text: accumulate into paragraph ---
        para_buf.append(line)
        i += 1

    flush_para()

    if not blocks:
        blocks.append({'text': '', 'type': 'unstyled', 'key': _key()})

    return {'blocks': blocks, 'entities': entities}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class AppInput(BaseAppInput):
    title: str = Field(
        description="Article title (headline)",
        min_length=1,
    )
    content: str = Field(
        description=(
            "Article body in markdown. Supports # headings, **bold**, *italic*, "
            "~~strikethrough~~, `code`, [links](url), lists, blockquotes, "
            "fenced code blocks, GFM tables, $$LaTeX$$, horizontal rules. "
            "Reference uploaded media as ![alt](file:0), ![alt](file:1). "
            "Embed tweets as {{tweet:ID}}."
        ),
        min_length=1,
    )
    cover_image: Optional[File] = Field(
        None,
        description="Cover image displayed at the top of the article",
    )
    media: Optional[List[File]] = Field(
        None,
        description="Media files referenced in content as file:0, file:1, etc. Supports images (JPG, PNG, WEBP, GIF) and video (MP4).",
    )
    publish: bool = Field(
        True,
        description="Publish immediately. Set false to save as draft only.",
    )


class AppOutput(BaseAppOutput):
    article_id: str = Field(description="ID of the created article")
    title: str = Field(description="Article title")
    published: bool = Field(description="Whether the article was published")
    post_id: Optional[str] = Field(None, description="ID of the post created on publish")
    post_url: Optional[str] = Field(None, description="URL of the post created on publish")


class App(BaseApp):
    client: Client = None
    api_session: requests.Session = None

    async def setup(self):
        access_token = os.environ.get("X_ACCESS_TOKEN")
        if not access_token:
            raise ValueError(
                "X_ACCESS_TOKEN not found. "
                "Please ensure the X.com integration is connected in Settings."
            )
        self.client = Client(access_token=access_token)
        self.api_session = requests.Session()
        self.api_session.headers.update({
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        })
        logger.info("X.com client initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        # Upload inline media
        media_map: Dict[int, Tuple[str, str]] = {}
        if input_data.media:
            for idx, mf in enumerate(input_data.media):
                ct = mf.content_type or get_content_type(mf.path)
                cat = get_media_category(ct)
                mid = await upload_file(self.client, mf.path, ct)
                media_map[idx] = (mid, cat)
                logger.info(f"Uploaded media file:{idx} ({cat}) → {mid}")

        # Upload cover image
        cover_media = None
        if input_data.cover_image:
            cf = input_data.cover_image
            ct = cf.content_type or get_content_type(cf.path)
            cat = get_media_category(ct)
            mid = await upload_file(self.client, cf.path, ct)
            cover_media = {"media_id": mid, "media_category": cat}
            logger.info(f"Uploaded cover image ({cat}) → {mid}")

        # Convert markdown → DraftJS
        content_state = markdown_to_draftjs(input_data.content, media_map)
        block_count = len(content_state['blocks'])
        entity_count = len(content_state['entities'])
        logger.info(f"Converted markdown: {block_count} blocks, {entity_count} entities")

        # Create draft via X API (not in xdk yet)
        payload = {"title": input_data.title, "content_state": content_state}
        if cover_media:
            payload["cover_media"] = cover_media

        resp = self.api_session.post("https://api.x.com/2/articles/draft", json=payload)
        if not resp.ok:
            if resp.status_code == 429:
                raise ValueError(format_rate_limit_from_response(resp))
            raise ValueError(f"Failed to create article draft ({resp.status_code}): {resp.text[:500]}")
        draft_data = resp.json()
        article_id = draft_data["data"]["id"]
        logger.info(f"Draft created: article_id={article_id}")

        # Publish if requested
        post_id = None
        post_url = None
        if input_data.publish:
            resp = self.api_session.post(f"https://api.x.com/2/articles/{article_id}/publish")
            if not resp.ok:
                if resp.status_code == 429:
                    raise ValueError(f"Draft created (id={article_id}) but publish rate limited. {format_rate_limit_from_response(resp)}")
                raise ValueError(
                    f"Draft created (id={article_id}) but publish failed ({resp.status_code}): {resp.text[:500]}"
                )
            pub_data = resp.json()
            post_id = pub_data["data"]["post_id"]
            post_url = f"https://x.com/i/web/status/{post_id}"
            logger.info(f"Article published: {post_url}")

        return AppOutput(
            article_id=article_id,
            title=input_data.title,
            published=input_data.publish,
            post_id=post_id,
            post_url=post_url,
        )

    async def unload(self):
        self.client = None
        if self.api_session:
            self.api_session.close()
        self.api_session = None
