"""No-network tests for the request mapping, the read-out and the server flags. Run: pytest test_jev_27b_v1_1.py"""

import asyncio
import importlib
import math
import os
import sys
import types

import pytest

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jev-27b-v1-1")
pkg = types.ModuleType("jev27v11")
pkg.__path__ = [APP_DIR]
sys.modules["jev27v11"] = pkg
m = importlib.import_module("jev27v11.inference")

LEVELS_3 = ["calm", "annoyed", "furious"]


def make_input(**overrides):
    data = {
        "state": {"ticket": "charged twice"},
        "choices": [{"id": "team", "instructions": "Which team?", "options": [
            {"name": "billing", "description": "Charges and refunds"}, {"name": "orders"}, {"name": "other"}]}],
        "scores": [{"id": "tone", "instructions": "How angry?", "levels": LEVELS_3}],
        "nouls": [
            {"id": "refund", "instructions": "Is a refund requested?"},
            {"id": "legal", "instructions": "Is a lawyer mentioned?", "criteria": {"true": "explicit mention", "false": "no mention"}},
        ],
    }
    data.update(overrides)
    return m.AppInput(**data)


def test_questions_map_to_jev_format_shapes():
    qs = {qid: (q, n) for qid, q, n in m.jev_questions(make_input())}
    assert qs["team"] == ({"type": "choice", "instructions": "Which team?",
                           "criteria": {"billing": "Charges and refunds", "orders": None, "other": None}}, 3)
    assert qs["tone"] == ({"type": "choice", "instructions": "How angry?", "criteria": {"0": "calm", "1": "annoyed", "2": "furious"}}, 3)
    assert qs["refund"] == ({"type": "noul", "instructions": "Is a refund requested?"}, 2)
    assert qs["legal"][0]["criteria"] == {"true": "explicit mention", "false": "no mention"}


def test_readout_request_and_probabilities_follow_the_engine():
    body = m.readout_request(m.MODEL_ID, "prompt", [32, 33, 34])
    assert body["max_tokens"] == 1 and body["logprobs"] == 3 and body["allowed_token_ids"] == [32, 33, 34]
    assert body["return_tokens_as_token_ids"] is True and body["temperature"] == 1.0
    response = {"choices": [{"logprobs": {"top_logprobs": [{"token_id:32": math.log(0.2), "token_id:33": math.log(0.6)}]}}]}
    probs = m.readout_probabilities(response, [32, 33, 34], 1.0)
    assert probs[0] == pytest.approx(0.25) and probs[1] == pytest.approx(0.75) and probs[2] < 1e-300


def test_answers_in_contract_form():
    answers = m.build_answers(make_input(), {"team": [0.7, 0.2, 0.1], "tone": [0.1, 0.1, 0.8], "refund": [0.3, 0.7], "legal": [0.9, 0.1]})
    assert answers["team"]["choice"] == "billing"
    assert answers["tone"]["score"] == pytest.approx(1.7)
    assert answers["refund"]["noul"] == 0.7 and answers["legal"]["noul"] == 0.1


def test_kernel_warmup_covers_every_width():
    counts = m.warmup_option_counts(255)
    keys = {m.logprob_kernel_key(n) for n in counts}
    assert keys == {m.logprob_kernel_key(n) for n in range(2, 256)}
    assert len(counts) == len(keys)


def test_server_follows_serve_vllm_sh():
    cmd = m.server_command("/models/jev", 8090)
    for flag in ("--no-enable-prefix-caching", "--no-enable-chunked-prefill"):
        assert flag in cmd
    assert cmd[cmd.index("--max-num-batched-tokens") + 1] == cmd[cmd.index("--max-model-len") + 1] == "32768"
    assert cmd[cmd.index("--attention-backend") + 1] == "FLASH_ATTN"
    assert cmd[cmd.index("--max-logprobs") + 1] == "255"
    env = m.server_env({"HF_HOME": "/cache/app/hf"})
    assert env["STARTLUX_PREFIX_BIDIR"] == "1" and env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert env["VLLM_CACHE_ROOT"] == "/cache/app/vllm"


def test_images_are_refused(tmp_path):
    image = tmp_path / "x.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    app = m.App()
    with pytest.raises(ValueError, match="text-only"):
        asyncio.run(app.run(make_input(images=[{"path": str(image)}])))
