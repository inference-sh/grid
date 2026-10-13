"""No-network tests for meta_llm: request mapping, stream folding, metering."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from inferencesh import File
from inferencesh.models.llm import ContextMessage, ContextMessageRole, LLMDelta, LLMInput

_spec = importlib.util.spec_from_file_location("meta_llm", Path(__file__).with_name("meta_llm.py"))
meta_llm = importlib.util.module_from_spec(_spec)
sys.modules["meta_llm"] = meta_llm
_spec.loader.exec_module(meta_llm)

WONDER = ("minimal", "low", "medium", "high", "xhigh", "max")


def body(efforts=WONDER, **kw):
    return meta_llm.build_request_body(LLMInput(text="hi", **kw), "m", max_output_tokens=1000, efforts=efforts)


def fold(events):
    state = meta_llm._create_initial_state()
    deltas = [d for d in (meta_llm._handle_event(e, state) for e in events) if d is not None]
    for d in deltas:
        LLMDelta(**d)
    return state, deltas


def completed(**usage):
    return {"type": "response.completed", "response": {"id": "r", "status": "completed", "usage": usage}}


class TestReasoning:
    def test_none_maps_to_minimal(self):
        assert body(reasoning_effort="none")["reasoning"]["effort"] == "minimal"

    def test_supported_value_passes_through(self):
        assert body(reasoning_effort="high")["reasoning"]["effort"] == "high"

    def test_out_of_set_value_maps_to_nearest(self):
        assert body(("low", "high"), reasoning_effort="medium")["reasoning"]["effort"] in ("low", "high")

    def test_no_effort_control_sends_no_effort(self):
        assert "effort" not in body(None, reasoning_effort="high")["reasoning"]

    def test_summary_requested(self):
        assert body(reasoning_effort="high")["reasoning"]["summary"] == "auto"

    def test_summary_can_be_dropped(self):
        req = meta_llm.build_request_body(LLMInput(text="hi"), "m", max_output_tokens=10, efforts=WONDER, with_summary=False)
        assert req["reasoning"] == {"effort": "minimal"}

    def test_encrypted_reasoning_always_included(self):
        assert body()["include"] == ["reasoning.encrypted_content"]


class TestRequest:
    def test_max_output_tokens_capped(self):
        assert body()["max_output_tokens"] == 1000                  # platform default 64000, capped at the model limit
        assert body(max_tokens=500)["max_output_tokens"] == 500
        assert body(max_tokens=1)["max_output_tokens"] == 16        # API minimum
        inp = LLMInput(text="hi"); inp.max_tokens = None
        assert "max_output_tokens" not in meta_llm.build_request_body(inp, "m", max_output_tokens=1000, efforts=WONDER)

    def test_store_off(self):
        assert body()["store"] is False and body()["stream"] is True

    def test_sampling_sent_when_set(self):
        req = body(temperature=0.3, top_p=0.9, frequency_penalty=0.5, presence_penalty=-0.5, top_k=5, seed=1, stop=["x"])
        assert req["temperature"] == 0.3 and req["top_p"] == 0.9
        assert req["frequency_penalty"] == 0.5 and req["presence_penalty"] == -0.5
        for unsupported in ("top_k", "seed", "stop"):
            assert unsupported not in req

    def test_platform_defaults_are_not_sent(self):
        assert not {"temperature", "top_p", "frequency_penalty", "presence_penalty"} & set(body())
        assert body(temperature=0.7)["temperature"] == 0.7  # the default value, but set by the caller

    def test_json_schema_format_is_flat(self):
        req = body(response_format={"type": "json_schema", "name": "r", "json_schema": {"type": "object"}, "strict": True})
        assert req["text"]["format"] == {"type": "json_schema", "name": "r", "schema": {"type": "object"}, "strict": True}


class TestTools:
    TOOL = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}}]

    def test_tools_are_flat_with_auto_choice(self):
        req = body(tools=self.TOOL)
        assert req["tools"] == [{"type": "function", "name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}]
        assert req["tool_choice"] == "auto" and req["parallel_tool_calls"] is True

    def test_none_choice_sends_no_tools(self):
        req = body(tools=self.TOOL, tool_choice={"mode": "none"})
        assert "tools" not in req and "tool_choice" not in req

    @pytest.mark.parametrize("choice", [{"mode": "required"}, {"mode": "function", "name": "f"}])
    def test_unsupported_choice_refused_before_request(self, choice):
        with pytest.raises(ValueError, match="only 'auto' and 'none'"):
            body(tools=self.TOOL, tool_choice=choice)


class TestInput:
    def test_image_url_passed_without_detail_when_no_local_copy(self):
        content = meta_llm._user_content("hi", [SimpleNamespace(uri="https://example.com/a.png", path=None)], None)
        assert content[1] == {"type": "input_image", "image_url": "https://example.com/a.png"}

    def test_local_copy_wins_over_url(self, tmp_path):
        """Meta fetches URLs itself and honours robots.txt; the downloaded copy goes inline."""
        from PIL import Image

        png = tmp_path / "a.png"
        Image.new("RGB", (8, 8), (255, 0, 0)).save(png)
        content = meta_llm._user_content("hi", [SimpleNamespace(uri="https://example.com/a.png", path=str(png))], None)
        assert content[1]["image_url"].startswith("data:image/png;base64,")
        pdf = tmp_path / "a.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        content = meta_llm._user_content("hi", None, [SimpleNamespace(uri="https://example.com/a.pdf", path=str(pdf), filename="a.pdf")])
        assert content[1]["file_data"].startswith("data:application/pdf;base64,") and "file_url" not in content[1]

    def test_file_url_passed(self):
        content = meta_llm._user_content("hi", None, [SimpleNamespace(uri="https://example.com/a.pdf", path=None, filename="a.pdf")])
        assert content[1] == {"type": "input_file", "file_url": "https://example.com/a.pdf"}

    def test_local_file_is_a_data_url_with_its_type(self, tmp_path):
        mp4 = tmp_path / "clip.mp4"
        mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        content = meta_llm._user_content("hi", None, [File(path=str(mp4))])
        part = content[1]
        assert part["type"] == "input_file" and part["filename"] == "clip.mp4"
        assert part["file_data"].startswith("data:video/mp4;base64,")

    def test_webp_is_kept(self, tmp_path):
        from PIL import Image

        webp = tmp_path / "shot.webp"
        Image.new("RGB", (64, 64), (0, 160, 0)).save(webp, format="WEBP")
        _, items = meta_llm.build_input(LLMInput(text="look", images=[File(path=str(webp))]))
        assert items[0]["content"][1]["image_url"].startswith("data:image/webp;base64,")

    def test_too_many_images_refused(self, monkeypatch):
        parts = [{"type": "input_text", "text": "hi"}] + [{"type": "input_image", "image_url": f"https://example.com/{i}.png"} for i in range(51)]
        monkeypatch.setattr(meta_llm, "_user_content", lambda text, images, files: parts)
        with pytest.raises(ValueError, match="at most 50 images"):
            meta_llm.build_input(LLMInput(text="hi"))

    def test_system_and_context(self):
        inp = LLMInput(
            system_prompt="be brief",
            context=[ContextMessage(role=ContextMessageRole.USER, text="a"), ContextMessage(role=ContextMessageRole.ASSISTANT, text="b")],
            text="c",
        )
        instructions, items = meta_llm.build_input(inp)
        assert instructions == "be brief"
        assert items == [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "b"}]},
            {"role": "user", "content": "c"},
        ]


class TestToolResults:
    def test_text_only_tool_result_is_a_string(self):
        _, items = meta_llm.build_input(LLMInput(text="42", role=ContextMessageRole.TOOL, tool_call_id="t1"))
        assert items == [{"type": "function_call_output", "call_id": "t1", "output": "42"}]

    def test_tool_images_follow_in_a_user_message(self, tmp_path):
        png = tmp_path / "shot.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        inp = LLMInput(
            context=[ContextMessage(role=ContextMessageRole.TOOL, tool_call_id="t0", text="captured", images=[File(path=str(png))])],
            text="done",
            role=ContextMessageRole.TOOL,
            tool_call_id="t1",
        )
        _, items = meta_llm.build_input(inp)
        assert [i.get("type") or i.get("role") for i in items] == ["function_call_output", "user", "function_call_output"]
        assert items[0]["output"] == "captured"
        assert items[1]["content"][0] == {"type": "input_text", "text": "Attachments returned by tool call t0"}
        assert items[1]["content"][1]["type"] == "input_image" and items[1]["content"][1]["image_url"].startswith("data:image/png;base64,")

    def test_assistant_calls_replayed_without_item_id(self):
        call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": {"a": 1}}}
        inp = LLMInput(
            context=[ContextMessage(role=ContextMessageRole.ASSISTANT, text="", tool_calls=[call])],
            text="ok", role=ContextMessageRole.TOOL, tool_call_id="c1",
        )
        _, items = meta_llm.build_input(inp)
        assert items[0] == {"type": "function_call", "call_id": "c1", "name": "f", "arguments": '{"a": 1}'}


class TestStream:
    def test_text_and_summary_deltas(self):
        state, deltas = fold([
            {"type": "response.reasoning_summary_part.added"},
            {"type": "response.reasoning_summary_text.delta", "delta": "think"},
            {"type": "response.reasoning_summary_part.added"},
            {"type": "response.reasoning_summary_text.delta", "delta": "more"},
            {"type": "response.output_text.delta", "delta": "hi"},
        ])
        assert state["reasoning"] == "think\n\nmore" and state["response"] == "hi"
        assert deltas[0] == {"reasoning": "think"} and deltas[-1] == {"response": "hi"}

    def test_streamed_function_call_arguments(self):
        state, deltas = fold([
            {"type": "response.output_item.added", "item": {"type": "function_call", "id": "fc1", "call_id": "c1", "name": "f", "arguments": ""}},
            {"type": "response.function_call_arguments.delta", "item_id": "fc1", "delta": '{"a":'},
            {"type": "response.function_call_arguments.delta", "item_id": "fc1", "delta": "1}"},
            {"type": "response.function_call_arguments.done", "item_id": "fc1", "arguments": '{"a":1}'},
            {"type": "response.output_item.done", "item": {"type": "function_call", "id": "fc1", "call_id": "c1", "name": "f", "arguments": '{"a":1}', "status": "completed"}},
        ])
        assert state["tool_calls"] == [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}}]
        assert [d["tool_calls"][0]["function"]["arguments"] for d in deltas] == ["", '{"a":', "1}"]

    def test_call_without_item_id_uses_latest_call(self):
        state, _ = fold([
            {"type": "response.output_item.added", "item": {"type": "function_call", "call_id": "c1", "name": "f", "arguments": ""}},
            {"type": "response.function_call_arguments.delta", "delta": "{}"},
        ])
        assert state["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_whole_function_call_only_in_done_event(self):
        state, deltas = fold([{"type": "response.output_item.done", "item": {
            "type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"}}])
        assert len(state["tool_calls"]) == 1 and deltas[0]["tool_calls"][0]["id"] == "c1"

    def test_incomplete_sets_length(self):
        state, _ = fold([{"type": "response.incomplete", "response": {"id": "r", "status": "incomplete",
                                                                        "incomplete_details": {"reason": "max_output_tokens"}}}])
        assert meta_llm._build_output(state, final=True)["usage"].stop_reason == "length"

    def test_failed_raises(self):
        with pytest.raises(RuntimeError, match="server_error"):
            fold([{"type": "response.failed", "response": {"error": {"code": "server_error", "message": "boom"}}}])

    def test_shutdown_before_output_is_retryable(self):
        with pytest.raises(meta_llm._RetryStream):
            fold([{"type": "error", "code": "server_shutting_down", "message": "bye"}])

    def test_shutdown_after_output_raises(self):
        with pytest.raises(RuntimeError, match="server_shutting_down"):
            fold([
                {"type": "response.output_text.delta", "delta": "partial"},
                {"type": "error", "code": "server_shutting_down", "message": "bye"},
            ])


class TestMetering:
    def test_usage_and_output_meta(self):
        state, _ = fold([completed(input_tokens=100, output_tokens=50, total_tokens=150,
                                   input_tokens_details={"cached_tokens": 40},
                                   output_tokens_details={"reasoning_tokens": 30})])
        out = meta_llm._build_output(state, final=True)
        usage, meta = out["usage"], out["output_meta"]
        assert usage.prompt_tokens == 100 and usage.cached_tokens == 40
        assert usage.completion_tokens == 50 and usage.reasoning_tokens == 30
        assert meta.inputs[0].tokens == 100 and meta.inputs[0].cached_tokens == 40
        assert meta.outputs[0].tokens == 50

    def test_reasoning_reported_separately_is_added(self):
        state, _ = fold([completed(input_tokens=32, output_tokens=9, total_tokens=151,
                                   output_tokens_details={"reasoning_tokens": 110})])
        assert meta_llm.billable_output_tokens(state) == 119

    def test_no_usage_no_meta(self):
        state = meta_llm._create_initial_state()
        assert "output_meta" not in meta_llm._build_output(state, final=True)


class TestReasoningState:
    ITEM = {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "think"}],
            "encrypted_content": "gAAA-enc"}

    def test_items_captured_without_status_and_with_summary(self):
        state = meta_llm._create_initial_state()
        bare = {"type": "reasoning", "id": "rs_2", "encrypted_content": "x", "status": "completed"}
        for item in (dict(self.ITEM, status="completed"), bare):
            assert meta_llm._handle_event({"type": "response.output_item.done", "item": item}, state) is None
        out = meta_llm._build_output(state, final=True)
        assert out["reasoning_state"] == {"meta": {"items": [self.ITEM, {"type": "reasoning", "id": "rs_2", "encrypted_content": "x", "summary": []}]}}
        assert "reasoning_state" not in meta_llm._build_output(state)

    def test_replayed_before_the_turns_text_and_calls(self):
        call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        inp = LLMInput(
            context=[
                ContextMessage(role=ContextMessageRole.USER, text="go"),
                ContextMessage(role=ContextMessageRole.ASSISTANT, text="calling", tool_calls=[call],
                               reasoning_state={"meta": {"items": [{"type": "reasoning", "id": "rs_1", "encrypted_content": "e"}]}}),
            ],
            text="42", role=ContextMessageRole.TOOL, tool_call_id="c1",
        )
        _, items = meta_llm.build_input(inp)
        assert [i.get("type") or i.get("role") for i in items] == [
            "user", "reasoning", "assistant", "function_call", "function_call_output"]
        assert items[1] == {"type": "reasoning", "id": "rs_1", "encrypted_content": "e", "summary": []}

    def test_another_providers_entry_is_not_sent(self):
        inp = LLMInput(
            context=[ContextMessage(role=ContextMessageRole.ASSISTANT, text="hi", reasoning_state={
                "openai": {"items": [self.ITEM]},
                "anthropic": [{"type": "thinking", "thinking": "t", "signature": "s"}],
            })],
            text="next",
        )
        _, items = meta_llm.build_input(inp)
        assert all(i.get("type") != "reasoning" for i in items)


class TestCapacityRejection:
    """Meta sheds load on its newest model as a 404 model_not_found; it is a 503."""

    BODY = {"error": {"code": "model_not_found", "message": "The requested model was not found.", "param": None, "type": "invalid_request_error"}}

    def test_model_not_found_404_is_a_503(self):
        status, body = meta_llm.normalize_capacity_rejection(404, self.BODY)
        assert status == 503 and status in meta_llm.RETRYABLE_STATUS
        assert body["error"]["code"] == "service_overloaded" and body["error"]["type"] == "server_error"
        assert "404 model_not_found" in body["error"]["message"]

    def test_error_reads_as_503(self):
        status, body = meta_llm.normalize_capacity_rejection(404, self.BODY)
        assert meta_llm._format_error(status, body, {}).startswith("Meta API error (503 server_error service_overloaded): Meta has no capacity")

    @pytest.mark.parametrize("status,body", [
        (404, {"error": {"code": "file_not_found", "message": "no"}}),
        (400, {"error": {"code": "model_not_found"}}),
        (404, "not json"),
        (404, {}),
    ])
    def test_other_errors_untouched(self, status, body):
        assert meta_llm.normalize_capacity_rejection(status, body) == (status, body)


class TestErrors:
    def test_format_error_names_type_code_param(self):
        text = meta_llm._format_error(400, {"error": {"type": "invalid_request_error", "code": "content_policy_violation",
                                                     "param": "input", "message": "no"}}, {"x-request-id": "abc"})
        assert text == "Meta API error (400 invalid_request_error content_policy_violation input): no [req:abc]"

    def test_summary_rejection_detected(self):
        assert meta_llm._is_summary_rejection(400, {"error": {"param": "reasoning.summary", "message": "unknown"}})
        assert not meta_llm._is_summary_rejection(400, {"error": {"param": "input", "message": "bad"}})
