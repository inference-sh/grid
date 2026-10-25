#!/usr/bin/env python3
"""A fake Anthropic Messages API for running the real claude CLI in tests.

  fake-anthropic.py [port]   (prints "listening http://127.0.0.1:<port>")

POST /v1/messages streams a tool_use of the first offered tool whose name
whose name contains "weather" when the latest user turn is plain text and
mentions "tool"; a turn mentioning "slow" streams 40 parts 250ms apart; otherwise streams a short text reply. GET
/api/oauth/usage answers a plan at 12.5% of its 5-hour window and 30% of its
week. Any other path answers {}. Requests are logged to stderr (path, model, tool names).
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def sse(w, event, data):
    w.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
    w.flush()


def last_user_text(body):
    for msg in reversed(body.get("messages", [])):
        if msg.get("role") != "user":
            continue
        c = msg.get("content")
        if isinstance(c, str):
            return c, False
        texts = [b.get("text", "") for b in c if b.get("type") == "text"]
        has_result = any(b.get("type") == "tool_result" for b in c)
        return " ".join(texts), has_result
    return "", False


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.split("?")[0] == "/api/oauth/usage":
            return self._json({"five_hour": {"utilization": 12.5, "resets_at": "2026-10-25T15:00:00Z"},
                               "limits": [{"kind": "weekly_all", "percent": 30, "resets_at": "2026-10-27T00:00:00Z"}]})
        self._json({})

    def do_HEAD(self):
        self._json({})

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(n) if n else b""
        path = self.path.split("?")[0]
        if path == "/v1/messages/count_tokens":
            return self._json({"input_tokens": 10})
        if path != "/v1/messages":
            return self._json({})
        body = json.loads(raw or b"{}")
        tools = [t.get("name") for t in body.get("tools", []) or []]
        text, has_result = last_user_text(body)
        auth = "bearer" if self.headers.get("authorization") else ("x-api-key" if self.headers.get("x-api-key") else "none")
        print(f"fake-anthropic: {path} model={body.get('model')} stream={body.get('stream')} auth={auth} tools={tools} user={text[:60]!r}", file=sys.stderr, flush=True)
        model = body.get("model", "claude")
        tool = next((t for t in tools if "weather" in t), None)
        want_tool = tool and "tool" in text.lower() and not has_result
        if not body.get("stream"):
            return self._json({"id": "msg_fake", "type": "message", "role": "assistant", "model": model,
                               "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                               "usage": {"input_tokens": 10, "output_tokens": 1}})
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("connection", "close")
        self.end_headers()
        w = self.wfile
        usage = {"input_tokens": 21, "output_tokens": 1, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        sse(w, "message_start", {"type": "message_start", "message": {"id": "msg_fake", "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": None, "usage": usage}})
        if want_tool:
            sse(w, "content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_fake_1", "name": tool, "input": {}}})
            for part in ('{"city": ', '"Paris"}'):
                sse(w, "content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": part}})
            stop = "tool_use"
        else:
            sse(w, "content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
            slow = "slow" in text.lower()
            parts = [f"part{i} " for i in range(40)] if slow else ["Hi from ", "the fake ", "Anthropic."]
            try:
                for part in parts:
                    sse(w, "content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": part}})
                    if slow:
                        time.sleep(0.25)
            except (BrokenPipeError, ConnectionResetError):
                print("fake-anthropic: client went away mid-stream", file=sys.stderr, flush=True)
                return
            stop = "end_turn"
        sse(w, "content_block_stop", {"type": "content_block_stop", "index": 0})
        sse(w, "message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": {"output_tokens": 7}})
        sse(w, "message_stop", {"type": "message_stop"})
        self.close_connection = True


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 0), H)
    print(f"listening http://127.0.0.1:{srv.server_address[1]}", flush=True)
    srv.serve_forever()
