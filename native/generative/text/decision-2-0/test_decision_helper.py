"""No-network tests for decision_helper: request building, validation, answer parsing.

    pytest test_decision_helper.py
"""

import logging

import pytest
from pydantic import ValidationError

from decision_helper import AppInput, decide

LOGGER = logging.getLogger("test")


class FakeModel:
    """Stands in for the loaded Decision 2.0 model: records the request, replays answers."""

    model_name = "Decision-2.0-Fake"
    max_input_tokens = 8192

    def __init__(self, answers, tokens=120):
        self.answers = answers
        self.tokens = tokens
        self.request = None

    def system_one(self, *, state, questions):
        self.request = {"state": state, "questions": questions}
        return {
            "model": self.model_name,
            "answers": self.answers,
            "usage": {"input_tokens": self.tokens, "output_tokens": 0},
        }


def full_input() -> AppInput:
    return AppInput(
        state={"ticket": "charged twice"},
        choices=[{
            "id": "team",
            "instructions": "Which team?",
            "options": [{"name": "billing", "description": {"what": "charges"}}, {"name": "other"}],
        }],
        scores=[{"id": "urgency", "instructions": "How urgent?", "levels": ["Routine", "Soon", "Today"]}],
        nouls=[
            {"id": "refund", "instructions": "Is a refund requested?"},
            {"id": "policy", "instructions": "Does policy allow it?", "criteria": {"true": "Covered"}},
        ],
    )


def full_answers() -> dict:
    return {
        "team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "other": 0.1}, "confidence": 0.53},
        "urgency": {
            "type": "score", "score": 1.5, "confidence": 0.2,
            "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
            "legend": {"0": "Routine", "1": "Soon", "2": "Today"},
        },
        "refund": {"type": "noul", "noul": 0.87},
        "policy": {"type": "noul", "noul": 0.4},
    }


def test_build_questions_rebuilds_the_keyed_map():
    questions = full_input().questions()
    assert list(questions) == ["team", "urgency", "refund", "policy"]
    assert questions["team"] == {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": {"what": "charges"}, "other": None},
    }
    assert questions["urgency"] == {"type": "score", "instructions": "How urgent?", "criteria": ["Routine", "Soon", "Today"]}
    assert questions["refund"] == {"type": "noul", "instructions": "Is a refund requested?"}
    assert questions["policy"]["criteria"] == {"true": "Covered"}


@pytest.mark.parametrize("payload, message", [
    ({"state": "x"}, "at least one question"),
    ({"state": "x", "nouls": [{"id": "a", "instructions": "q?"}], "scores": [{"id": "a", "instructions": "q", "levels": ["lo", "hi"]}]}, "unique"),
    ({"state": "x", "scores": [{"id": "s", "instructions": "q", "levels": ["only"]}]}, "at least 2"),
    ({"state": "x", "choices": [{"id": "c", "instructions": "q", "options": [{"name": "a"}, {"name": "a"}]}]}, "repeated option names"),
    ({"state": "x", "nouls": [{"id": "n", "instructions": ""}]}, "empty instructions"),
])
def test_bad_input_is_rejected_at_validation(payload, message):
    with pytest.raises(ValidationError, match=message):
        AppInput(**payload)


def test_decide_types_the_answers_and_meters_input_tokens():
    model = FakeModel(full_answers(), tokens=864)
    out = decide(model, full_input(), LOGGER)

    assert model.request["state"] == {"ticket": "charged twice"}
    assert out.choices["team"].choice == "billing"
    assert out.choices["team"].probabilities == {"billing": 0.9, "other": 0.1}
    assert out.scores["urgency"].score == 1.5
    assert out.scores["urgency"].normalized == 0.75
    assert out.scores["urgency"].legend["2"] == "Today"
    assert out.nouls["refund"].noul == 0.87
    assert out.model == "Decision-2.0-Fake"
    assert out.input_tokens == 864
    assert [m.tokens for m in out.output_meta.inputs] == [864]
    assert [m.tokens for m in out.output_meta.outputs] == [0]


def test_a_question_the_model_rejects_fails_the_run_with_the_limit():
    answers = full_answers()
    answers["refund"] = {"type": "noul", "error": "max_length_exceeded"}
    with pytest.raises(RuntimeError, match=r"1 of 4 questions.*max_length_exceeded for 'refund':.*8192 tokens"):
        decide(FakeModel(answers), full_input(), LOGGER)


def test_many_failures_are_grouped_by_reason():
    ids = [f"q{i}" for i in range(20)]
    data = AppInput(state="x", nouls=[{"id": i, "instructions": "q?"} for i in ids])
    answers = {i: {"type": "noul", "error": "max_length_exceeded"} for i in ids}
    with pytest.raises(RuntimeError) as failure:
        decide(FakeModel(answers), data, LOGGER)
    message = str(failure.value)
    assert message.startswith("20 of 20 questions could not be answered. max_length_exceeded for 'q0', ")
    assert "'q7' and 12 more" in message and "'q8'" not in message
    assert message.count("never truncated") == 1


def test_a_missing_answer_fails_the_run():
    answers = full_answers()
    del answers["policy"]
    with pytest.raises(RuntimeError, match=r"no answer for: \['policy'\]"):
        decide(FakeModel(answers), full_input(), LOGGER)
