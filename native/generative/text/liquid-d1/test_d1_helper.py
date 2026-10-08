"""No-network tests for d1_helper.py: question mapping, validation and answer typing."""

import pytest

from d1_helper import (
    ChoiceOption,
    ChoiceQuestion,
    NoulCriteria,
    NoulQuestion,
    ScoreQuestion,
    build_answers,
    build_questions,
    check_questions,
)

CHOICE = ChoiceQuestion(id="team", instructions="Which team?", options=[ChoiceOption(name="billing", description="Charges"), ChoiceOption(name="other")])
SCORE = ScoreQuestion(id="urgency", instructions="How urgent?", levels=["Can wait", "Today", "Now"])
NOUL = NoulQuestion(id="refund", instructions="Refund request?", criteria=NoulCriteria(true="Asks for money back"))


def test_build_questions_uses_the_models_format():
    questions = build_questions([CHOICE], [SCORE], [NOUL, NoulQuestion(id="plain", instructions="Is it rude?")])
    assert questions == {
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Charges", "other": None}},
        "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Can wait", "Today", "Now"]},
        "refund": {"type": "noul", "instructions": "Refund request?", "criteria": {"true": "Asks for money back"}},
        "plain": {"type": "noul", "instructions": "Is it rude?"},
    }


def test_check_questions_rejects_empty_and_repeats():
    with pytest.raises(ValueError, match="at least one question"):
        check_questions([], [], [])
    with pytest.raises(ValueError, match="repeated: \\['team'\\]"):
        check_questions([CHOICE], [], [NoulQuestion(id="team", instructions="x")])
    twice = ChoiceQuestion(id="c", instructions="x", options=[ChoiceOption(name="a"), ChoiceOption(name="a")])
    with pytest.raises(ValueError, match="repeated option names"):
        check_questions([twice], [], [])


def test_build_answers_types_and_normalizes():
    questions = build_questions([CHOICE], [SCORE], [NOUL])
    answers = {
        "team": {"type": "choice", "choice": "billing", "confidence": 0.9, "probabilities": {"billing": 0.9, "other": 0.1}},
        "urgency": {"type": "score", "score": 1.5, "confidence": 0.5, "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5}, "legend": {"0": "Can wait", "1": "Today", "2": "Now"}},
        "refund": {"type": "noul", "noul": 0.97},
    }
    out = build_answers(answers, questions)
    assert out["choices"]["team"].choice == "billing"
    assert out["scores"]["urgency"].normalized == 0.75
    assert out["nouls"]["refund"].noul == 0.97


def test_build_answers_names_a_missing_answer():
    with pytest.raises(RuntimeError, match="no answer for: \\['refund'\\]"):
        build_answers({}, build_questions([], [], [NOUL]))
