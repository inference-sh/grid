"""No-network test for d1_helper.py: questions as the text the d1 models read."""

from inferencesh.models.decision import DecisionInput

from d1_helper import text_questions


def test_text_questions_writes_structure_as_json_text():
    data = DecisionInput(
        state="x",
        choices=[{"id": "team", "instructions": {"ask": "Which team?"}, "options": [{"name": "billing", "description": {"covers": "charges"}}, {"name": "other"}]}],
        scores=[{"id": "urgency", "instructions": "How urgent?", "levels": ["Can wait", {"label": "Now"}]}],
        nouls=[{"id": "refund", "instructions": "Refund request?", "criteria": {"true": "Asks for money back"}}],
    )
    assert text_questions(data) == {
        "team": {"type": "choice", "instructions": '{"ask": "Which team?"}', "criteria": {"billing": '{"covers": "charges"}', "other": None}},
        "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Can wait", '{"label": "Now"}']},
        "refund": {"type": "noul", "instructions": "Refund request?", "criteria": {"true": "Asks for money back"}},
    }
