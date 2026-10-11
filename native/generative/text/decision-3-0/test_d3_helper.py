"""No-network tests for d3_helper: request passing, batch sizing, error grouping, answer typing.

    pytest test_d3_helper.py
"""

import logging
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from d3_helper import MAX_BATCH, AppInput, batch_size, decide, widest_prompt

LOGGER = logging.getLogger("test")

ROUTE = {"id": "route", "instructions": "Which team?", "options": [{"name": "returns"}, {"name": "billing"}]}
URGENCY = {"id": "urgency", "instructions": "How urgent?", "levels": ["Routine", "Soon", "Today"]}
RECEIPT = {"id": "receipt", "instructions": "Is there a receipt?"}


class FakeRuntime:
    """Stands in for the loaded d3 runtime: records the request, replays answers."""

    model_name = "d3-fake"

    def __init__(self, answers, lengths=None, sequences=None, tokens=120):
        self.answers = answers
        self.prepared = SimpleNamespace(lengths=lengths or {}, sequences=sequences or {"q": [0] * 100})
        self.tokens = tokens
        self.batch_size = MAX_BATCH
        self.request = None

    def prepare(self, state, questions, images=None, videos=None):
        self.request = {"state": state, "questions": questions, "images": images, "videos": videos}
        return self.prepared

    def run(self, prepared):
        return {}, self.tokens

    def respond(self, prepared, probabilities, tokens):
        return {"model": self.model_name, "answers": self.answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}


ANSWERS = {
    "route": {"type": "choice", "choice": "returns", "confidence": 0.9, "probabilities": {"returns": 0.97, "billing": 0.03}},
    "urgency": {"type": "score", "score": 1.5, "confidence": 0.4, "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
                "legend": {"0": "Routine", "1": "Soon", "2": "Today"}},
    "receipt": {"type": "noul", "noul": 0.8},
}


def test_decide_passes_media_paths_and_types_answers(tmp_path):
    image, video = tmp_path / "a.png", tmp_path / "b.mp4"
    image.write_bytes(b"\x89PNG")
    video.write_bytes(b"\x00")
    runtime = FakeRuntime(ANSWERS)
    out = decide(runtime, AppInput(state="x", images=[str(image)], videos=[str(video)],
                                   choices=[ROUTE], scores=[URGENCY], nouls=[RECEIPT]), 4096, LOGGER)
    assert runtime.request["images"] == [str(image)] and runtime.request["videos"] == [str(video)]
    assert out.choices["route"].choice == "returns"
    assert out.scores["urgency"].normalized == 0.75
    assert out.nouls["receipt"].noul == 0.8
    assert out.model == "d3-fake" and out.input_tokens == 120


def test_decide_sends_no_media_as_none():
    runtime = FakeRuntime({"receipt": ANSWERS["receipt"]})
    decide(runtime, AppInput(state="x", nouls=[RECEIPT]), 4096, LOGGER)
    assert runtime.request["images"] is None and runtime.request["videos"] is None


def test_batch_size_bounds_padded_tokens():
    assert batch_size(100, 4096) == MAX_BATCH
    assert batch_size(1000, 4096) == 4
    assert batch_size(40000, 4096) == 1
    assert batch_size(0, 4096) == MAX_BATCH


def test_widest_prompt_reads_text_and_media_plans():
    assert widest_prompt(SimpleNamespace(lengths={"a": 900}, sequences={})) == 900
    assert widest_prompt(SimpleNamespace(lengths={}, sequences={"a": [0] * 30, "b": [0] * 70})) == 70
    assert widest_prompt(SimpleNamespace(lengths={}, sequences={})) == 0


def test_decide_sets_batch_from_widest_prompt():
    runtime = FakeRuntime({"receipt": ANSWERS["receipt"]}, lengths={"receipt": 1000}, sequences={})
    decide(runtime, AppInput(state="x", nouls=[RECEIPT]), 4096, LOGGER)
    assert runtime.batch_size == 4


def test_failed_questions_are_grouped_by_reason():
    answers = {f"q{i}": {"type": "noul", "error": "max_length_exceeded"} for i in range(10)}
    runtime = FakeRuntime(answers)
    with pytest.raises(RuntimeError, match=r"10 of 10 questions could not be answered\. max_length_exceeded for 'q0'.* and 2 more"):
        decide(runtime, AppInput(state="x", nouls=[{"id": f"q{i}", "instructions": "y"} for i in range(10)]), 4096, LOGGER)


def test_unreadable_media_is_a_readable_error():
    runtime = FakeRuntime({})

    def refuse(*args, **kwargs):
        raise ValueError("videos must be base64 MP4, WebM, QuickTime or Matroska data URLs")

    runtime.prepare = refuse
    with pytest.raises(ValueError, match="the request could not be read: videos must be"):
        decide(runtime, AppInput(state="x", nouls=[RECEIPT]), 4096, LOGGER)


def test_input_limits_media_counts(tmp_path):
    video = tmp_path / "b.mp4"
    video.write_bytes(b"\x00")
    AppInput(videos=[str(video)], nouls=[RECEIPT])
    with pytest.raises(ValidationError):
        AppInput(videos=[str(video)] * 5, nouls=[RECEIPT])
    with pytest.raises(ValidationError, match="nothing to evaluate"):
        AppInput(nouls=[RECEIPT])
