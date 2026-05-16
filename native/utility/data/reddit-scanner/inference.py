import csv
import io
import os
import re
from typing import Optional

import asyncpraw
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from inferencesh.models.usage import OutputMeta, TextMeta
from pydantic import Field


class AppInput(BaseAppInput):
    subreddit: str = Field(description="Subreddit to scan (without r/ prefix)")
    max_posts: int = Field(default=100, description="Maximum number of posts to fetch (max 1000)")
    include_comments: bool = Field(default=True, description="Fetch comments for each post")
    min_words: int = Field(default=5, description="Minimum words to include in parsed output")
    max_words: int = Field(default=10000, description="Maximum words to include in parsed output")
    sort: str = Field(default="new", description="Sort order: new, hot, top, rising")
    time_filter: str = Field(default="all", description="Time filter for top sort: hour, day, week, month, year, all")


class AppOutput(BaseAppOutput):
    csv_file: File = Field(description="Raw CSV with all posts and comments")
    parsed_file: File = Field(description="Cleaned text output (one entry per line)")
    total_posts: int = Field(description="Number of posts scanned")
    total_comments: int = Field(description="Number of comments fetched")
    parsed_entries: int = Field(description="Number of entries after cleaning")


# Sanitization patterns
URL_RE = re.compile(r"https?://\S+")
EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001F9FF]|[\u2600-\u26FF]|[\u2700-\u27BF]"
    r"|[\uFE00-\uFE0F]|[\U0001F000-\U0001F02F]|[\U0001F0A0-\U0001F0FF]"
)
HTML_RE = re.compile(r"<[^>]*>")
MULTI_SPACE_RE = re.compile(r"\s{2,}")
OPENER_RE = re.compile(
    r"(?im)^(hey|hi|hello|yo|what's up|anyone else|quick question|genuine question"
    r"|so i've been|i've been|just wanted to)[^\n]*\n*"
)
CLOSER_RE = re.compile(
    r"(?im)(\n|\r|^)(thanks.*$|would love to hear.*$|curious.*$|let me know.*$"
    r"|happy to.*$|feedback welcome.*$|what do you think.*$|thoughts\?.*$"
    r"|any thoughts.*$|appreciate any.*$)"
)
PROMO_RE = re.compile(
    r"(?i)(link in (the )?comments?|dm me|check it out|sign up|join us|github\.com"
    r"|repo link|waitlist|free credits|pricing|subscription|i built|we built"
    r"|i made|we made|i created|we created|just launched|just released|just shipped"
    r"|excited to (launch|announce|share)|launching|beta access|early access"
    r"|try it out|give it a try|looking for contributors|looking for feedback"
    r"|would love feedback|AMA|ask me anything|introducing|announcing)"
)
EMPTY_BULLET_RE = re.compile(r"(?m)^\s*[-•*]\s*$")
BULLET_SPAM_RE = re.compile(r"(\n[-•*][^\n]*){5,}")
OPINION_RE = re.compile(
    r"(?im)^(i think|i feel|personally|in my opinion|honestly|to be real"
    r"|hot take|unpopular opinion|imo|imho)[^\n]*\n*"
)
QUESTION_ONLY_RE = re.compile(r"(?m)^.{0,200}\?\s*$")
SPECIAL_CHARS_RE = re.compile(r"[^\w\s.,!?;:'\"()\-]")


def clean_text(text: str) -> str:
    """Remove markdown formatting and normalize for CSV."""
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"^>\s+", "", text)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = text.replace(",", ";")
    text = " ".join(text.split())
    return text.strip()


def sanitize(text: str) -> Optional[str]:
    """Multi-pass sanitization pipeline. Returns None if promotional."""
    cleaned = text
    cleaned = URL_RE.sub("", cleaned)
    cleaned = HTML_RE.sub("", cleaned)
    cleaned = EMOJI_RE.sub("", cleaned)
    cleaned = MULTI_SPACE_RE.sub(" ", cleaned)
    cleaned = OPENER_RE.sub("", cleaned)
    cleaned = CLOSER_RE.sub("", cleaned)

    if PROMO_RE.search(text):
        return None

    cleaned = EMPTY_BULLET_RE.sub("", cleaned)
    cleaned = BULLET_SPAM_RE.sub("\n", cleaned)
    cleaned = OPINION_RE.sub("", cleaned)
    cleaned = QUESTION_ONLY_RE.sub("", cleaned)
    cleaned = SPECIAL_CHARS_RE.sub(" ", cleaned)
    cleaned = MULTI_SPACE_RE.sub(" ", cleaned)
    return cleaned.strip() or None


class App(BaseApp):
    async def setup(self, config):
        self.client_id = os.environ["REDDIT_CLIENT_ID"]
        self.client_secret = os.environ["REDDIT_CLIENT_SECRET"]

    async def run(self, input_data: AppInput) -> AppOutput:
        max_posts = min(input_data.max_posts, 1000)
        self.logger.info(f"Scanning r/{input_data.subreddit} for up to {max_posts} posts (sort={input_data.sort})")

        reddit = asyncpraw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent="inference.sh:reddit-scanner:v1.0",
        )

        try:
            return await self._scan(reddit, input_data, max_posts)
        finally:
            await reddit.close()

    async def _scan(self, reddit, input_data: AppInput, max_posts: int) -> AppOutput:
        subreddit = await reddit.subreddit(input_data.subreddit)

        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["type", "subreddit", "id", "author", "content", "url", "parent_id"])

        if input_data.sort == "hot":
            listing = subreddit.hot(limit=max_posts)
        elif input_data.sort == "top":
            listing = subreddit.top(time_filter=input_data.time_filter, limit=max_posts)
        elif input_data.sort == "rising":
            listing = subreddit.rising(limit=max_posts)
        else:
            listing = subreddit.new(limit=max_posts)

        total_posts = 0
        total_comments = 0
        seen_ids = set()
        parsed_lines = []
        seen_content = set()

        async for post in listing:
            if post.id in seen_ids:
                continue
            seen_ids.add(post.id)
            total_posts += 1

            title = clean_text(post.title or "")
            body = clean_text(post.selftext or "")
            text = f"{title} | {body}" if body else title
            author = str(post.author) if post.author else "[deleted]"

            writer.writerow(["post", input_data.subreddit, post.id, author, text, post.permalink, ""])

            # Parse for cleaned output
            sanitized = sanitize(text)
            if sanitized and sanitized not in seen_content:
                words = sanitized.split()
                if input_data.min_words <= len(words) <= input_data.max_words:
                    seen_content.add(sanitized)
                    parsed_lines.append(sanitized)

            # Fetch comments
            if input_data.include_comments and post.num_comments > 0:
                self.logger.info(f"Fetching comments for post {post.id} ({post.num_comments} comments)")
                try:
                    post.comment_sort = "best"
                    await post.load()
                    await post.comments.replace_more(limit=0)
                    for comment in post.comments.list():
                        if not hasattr(comment, "body") or not comment.body:
                            continue
                        comment_text = clean_text(comment.body)
                        if not comment_text:
                            continue
                        c_author = str(comment.author) if comment.author else "[deleted]"
                        writer.writerow(["comment", input_data.subreddit, comment.id, c_author, comment_text, comment.permalink, post.id])
                        total_comments += 1

                        sanitized_c = sanitize(comment_text)
                        if sanitized_c and sanitized_c not in seen_content:
                            words = sanitized_c.split()
                            if input_data.min_words <= len(words) <= input_data.max_words:
                                seen_content.add(sanitized_c)
                                parsed_lines.append(sanitized_c)
                except Exception as e:
                    self.logger.warning(f"Error fetching comments for {post.id}: {e}")

            if total_posts % 25 == 0:
                self.logger.info(f"Progress: {total_posts}/{max_posts} posts, {total_comments} comments")

        self.logger.info(f"Done: {total_posts} posts, {total_comments} comments, {len(parsed_lines)} parsed entries")

        csv_path = "/tmp/reddit_scan.csv"
        with open(csv_path, "w") as f:
            f.write(csv_buf.getvalue())

        parsed_path = "/tmp/reddit_parsed.txt"
        with open(parsed_path, "w") as f:
            for line in parsed_lines:
                f.write(line + "\n")

        total_chars = sum(len(line) for line in parsed_lines)

        return AppOutput(
            csv_file=File(path=csv_path),
            parsed_file=File(path=parsed_path),
            total_posts=total_posts,
            total_comments=total_comments,
            parsed_entries=len(parsed_lines),
            output_meta=OutputMeta(
                outputs=[TextMeta(char_count=total_chars, token_count=total_chars // 4)]
            ),
        )
