import anthropic
import httpx
import pytest

import llm_client


class FakeBlock:
    def __init__(self, type_, name=None, input_=None):
        self.type = type_
        self.name = name
        self.input = input_


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeMessages:
    def __init__(self, side_effects):
        self.side_effects = list(side_effects)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        effect = self.side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


class FakeClient:
    def __init__(self, side_effects):
        self.messages = FakeMessages(side_effects)


def _rate_limit_error():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request)
    return anthropic.RateLimitError("rate limited", response=response, body=None)


def test_call_extraction_returns_tool_input_on_success():
    result_payload = {"rows": [], "summary_paragraph": "the answer"}
    response = FakeResponse([FakeBlock("tool_use", name="record_extraction", input_=result_payload)])
    client = FakeClient([response])

    result = llm_client.call_extraction(client, "claude-sonnet-5", "paper text", "pmc_fulltext", "instructions")
    assert result == result_payload
    assert client.messages.calls == 1


def test_call_extraction_retries_on_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm_client.time, "sleep", lambda _: None)
    result_payload = {"rows": [], "summary_paragraph": "ok after retry"}
    response = FakeResponse([FakeBlock("tool_use", name="record_extraction", input_=result_payload)])
    client = FakeClient([_rate_limit_error(), _rate_limit_error(), response])

    result = llm_client.call_extraction(client, "claude-sonnet-5", "text", "pubmed_abstract", "instructions")
    assert result == result_payload
    assert client.messages.calls == 3


def test_call_extraction_raises_after_max_retries(monkeypatch):
    monkeypatch.setattr(llm_client.time, "sleep", lambda _: None)
    client = FakeClient([_rate_limit_error()] * llm_client.MAX_RETRIES)

    with pytest.raises(RuntimeError):
        llm_client.call_extraction(client, "claude-sonnet-5", "text", "pmc_fulltext", "instructions")
    assert client.messages.calls == llm_client.MAX_RETRIES


def test_call_extraction_raises_if_no_tool_use_block_returned():
    response = FakeResponse([FakeBlock("text")])
    client = FakeClient([response])

    with pytest.raises(RuntimeError):
        llm_client.call_extraction(client, "claude-sonnet-5", "text", "pmc_fulltext", "instructions")


def test_build_user_message_flags_abstract_only_source():
    msg = llm_client.build_user_message("some abstract", "pubmed_abstract", "do the thing")
    assert "ABSTRACT ONLY" in msg
    assert "some abstract" in msg
    assert "do the thing" in msg
