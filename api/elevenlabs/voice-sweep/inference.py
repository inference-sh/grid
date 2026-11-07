"""
ElevenLabs Voice Sweep

Admin maintenance: delete cloned voices on the platform ElevenLabs account
that are stale or past an explicit expiry.

ElevenLabs has no last-used field, so last use is the newest /v1/history
entry for the voice. A voice is deleted when it is older than `days` and has
no history entry in the last `days`, or when its description carries
`expires:YYYY-MM-DD` and that date has passed. Dry run by default.
"""

import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import BaseModel, Field

BASE = "https://api.elevenlabs.io"
EXPIRES_RE = re.compile(r"expires:(\d{4}-\d{2}-\d{2})")


class AppInput(BaseAppInput):
    days: int = Field(default=30, ge=1, le=365, description="Delete voices older than this with no use in this many days.")
    delete: bool = Field(default=False, description="Actually delete. False is a dry run that only reports.")
    keep: List[str] = Field(default_factory=list, max_length=500, description="Voice IDs never deleted.")


class VoiceRow(BaseModel):
    voice_id: str
    name: str
    created: Optional[str] = None
    last_used: Optional[str] = None
    expires: Optional[str] = None
    reason: str


class AppOutput(BaseAppOutput):
    total_cloned: int = Field(description="Cloned voices on the account before the sweep")
    stale: int = Field(description="Voices selected for deletion")
    deleted: int = Field(description="Voices deleted (0 on a dry run)")
    failed: int = Field(description="Deletes that failed")
    voices: List[VoiceRow] = Field(description="Voices selected for deletion, with the reason")
    report: File = Field(description="Full report (JSON) of every cloned voice and its decision")
    subscription: Optional[dict] = Field(default=None, description="Voice limits from /v1/user/subscription, if the key may read them")
    warning: Optional[str] = Field(default=None, description="Why the sweep could not decide or delete, if so")


def _day(ts):
    return time.strftime("%Y-%m-%d", time.gmtime(ts)) if ts else None


class App(BaseApp):
    async def setup(self, config=None):
        self.logger = logging.getLogger(__name__)
        self.key = os.environ.get("ELEVENLABS_KEY")
        if not self.key:
            raise RuntimeError("ELEVENLABS_KEY is required")

    def _call(self, method, path, params=None, retries=4):
        url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
        for attempt in range(retries + 1):
            req = urllib.request.Request(url, method=method, headers={"xi-api-key": self.key})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    body = r.read()
                    return json.loads(body) if body else {}
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError("%s %s -> %s %s" % (method, path, e.code, e.read()[:300]))

    def _cloned_voices(self):
        out, tok = [], None
        while True:
            p = {"category": "cloned", "voice_type": "personal", "page_size": 100, "include_total_count": "false"}
            if tok:
                p["next_page_token"] = tok
            d = self._call("GET", "/v2/voices", p)
            out += d.get("voices", [])
            if not d.get("has_more"):
                return out
            tok = d.get("next_page_token")

    def _last_used(self, voice_id):
        h = self._call("GET", "/v1/history", {"voice_id": voice_id, "page_size": 1}).get("history", [])
        return h[0]["date_unix"] if h else None

    def _delete(self, voice_id):
        try:
            self._call("DELETE", "/v1/voices/%s" % voice_id)
            return True
        except Exception as e:
            self.logger.error("delete %s failed: %s", voice_id, e)
            return False

    async def run(self, input_data: AppInput) -> AppOutput:
        now = int(time.time())
        cutoff = now - input_data.days * 86400
        today = _day(now)
        keep = set(input_data.keep)

        sub, warning = None, None
        try:
            d = self._call("GET", "/v1/user/subscription")
            sub = {k: d.get(k) for k in ("tier", "voice_limit", "voice_slots_used", "voice_add_edit_counter",
                                         "max_voice_add_edits", "next_character_count_reset_unix", "status")}
            self.logger.info("subscription: %s", sub)
        except Exception as e:
            self.logger.warning("subscription unreadable: %s", e)

        voices = self._cloned_voices()
        self.logger.info("cloned voices: %d", len(voices))
        try:
            with ThreadPoolExecutor(8) as pool:
                used = list(pool.map(lambda v: self._last_used(v["voice_id"]), voices))
        except Exception as e:
            warning = "history unreadable, nothing selected or deleted: %s" % e
            self.logger.error(warning)
            path = "/tmp/voice_sweep_report.json"
            with open(path, "w") as f:
                json.dump({"voices": [{"voice_id": v["voice_id"], "name": v.get("name"),
                                       "created": _day(v.get("created_at_unix"))} for v in voices]}, f, ensure_ascii=False, indent=1)
            return AppOutput(total_cloned=len(voices), stale=0, deleted=0, failed=0, voices=[],
                             report=File(path=path), subscription=sub, warning=warning)

        rows, stale = [], []
        for v, lu in zip(voices, used):
            created = v.get("created_at_unix")
            m = EXPIRES_RE.search(v.get("description") or "")
            expires = m.group(1) if m else None
            if v["voice_id"] in keep:
                reason = "keep: listed"
            elif expires and expires < today:
                reason = "delete: expired %s" % expires
            elif lu and lu >= cutoff:
                reason = "keep: used"
            elif created and created >= cutoff:
                reason = "keep: new"
            else:
                reason = "delete: unused %dd" % input_data.days
            row = VoiceRow(voice_id=v["voice_id"], name=v.get("name", ""), created=_day(created),
                           last_used=_day(lu), expires=expires, reason=reason)
            rows.append(row)
            if reason.startswith("delete"):
                stale.append(row)
        self.logger.info("stale: %d", len(stale))

        deleted = failed = 0
        if input_data.delete and stale:
            with ThreadPoolExecutor(8) as pool:
                ok = list(pool.map(lambda r: self._delete(r.voice_id), stale))
            deleted = sum(ok)
            failed = len(ok) - deleted
            self.logger.info("deleted: %d, failed: %d", deleted, failed)

        path = "/tmp/voice_sweep_report.json"
        with open(path, "w") as f:
            json.dump({"days": input_data.days, "delete": input_data.delete,
                       "voices": [r.model_dump() for r in rows]}, f, ensure_ascii=False, indent=1)

        return AppOutput(total_cloned=len(voices), stale=len(stale), deleted=deleted, failed=failed,
                         voices=stale, report=File(path=path), subscription=sub)
