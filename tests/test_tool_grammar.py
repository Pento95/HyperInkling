"""Tool-call grammar (HYPERINKLING_TOOL_GRAMMAR) and the arguments of a broken call. Needs the
tokenizer of a downloaded checkpoint; no GPU.

    venv/bin/python -m pytest -q tests/test_tool_grammar.py
"""

import json
import re
from pathlib import Path

import pytest
import xgrammar as xgr

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.inkling import InklingParser, _inkling_arg_converter
from vllm.tokenizers import get_tokenizer
from vllm.v1.structured_output.backend_xgrammar import _without_json_whitespace

CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "Inkling-2-Mini-W4A16-g128-gptq"
SCHEMA = {
    "type": "object", "required": ["customer", "items", "vat"],
    "properties": {"customer": {"type": "string"}, "vat": {"type": "number"}, "items": {
        "type": "array", "items": {"type": "object", "required": ["qty"],
                                   "properties": {"qty": {"type": "integer"}}}}},
}
GOOD = '{"name":"inv","args":{"customer":"J","items":[{"qty":1},{"qty":2}],"vat":0}}'
MISPLACED = GOOD.replace('}],"vat"', '}},"vat"')  # what the model writes
OUTPUT = ('<|message_model|><|content_thinking|>free {"a": [1}}<|end_message|>'
          "<|message_model|>inv<|content_invoke_tool_json|>%s<|end_message|>"
          '<|message_model|>f<|content_invoke_tool_json|>{"name":"f","args":{}}<|end_message|>'
          "<|message_model|><|content_text|>Done<|end_message|><|content_model_end_sampling|>")


@pytest.fixture(scope="module")
def tokenizer():
    if not CHECKPOINT.is_dir():
        pytest.skip(f"{CHECKPOINT} is not there")
    return get_tokenizer(str(CHECKPOINT))


def request(strict=False, **kw):
    inv = {"name": "inv", "parameters": SCHEMA, **({"strict": True} if strict else {})}
    tools = [{"type": "function", "function": f} for f in (inv, {"name": "f", "parameters": {}})]
    return ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "x"}], tools=tools, **kw)


def adjust(tokenizer, req):
    parser = InklingParser(tokenizer, None)
    return parser.adjust_request(parser.adjust_request(req))  # serving calls it twice


@pytest.fixture(scope="module")
def accepts(tokenizer):
    vocab = tokenizer.get_vocab()
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer))

    def check(req, text):
        tag = xgr.StructuralTag.model_validate(json.loads(req.structured_outputs.structural_tag))
        matcher = xgr.GrammarMatcher(compiler.compile_structural_tag(_without_json_whitespace(tag)))
        ids = []
        for part in filter(None, re.split(r"(<\|[a-z_]+\|>)", text)):
            ids += [vocab[part]] if part in vocab else tokenizer.encode(
                part, add_special_tokens=False)
        return all(matcher.accept_token(i) for i in ids)
    return check


@pytest.mark.parametrize("strict", [False, True])
def test_grammar(tokenizer, accepts, monkeypatch, strict):
    monkeypatch.setenv("HYPERINKLING_TOOL_GRAMMAR", "1")
    req = adjust(tokenizer, request(strict))
    assert req._grammar_from_parser
    assert accepts(req, OUTPUT % GOOD)
    assert accepts(req, OUTPUT % '{"name":"inv","args":{"vat":0,"items":[],"customer":"J"}}')
    for bad in (MISPLACED, GOOD[:-1], GOOD + "}"):
        assert not accepts(req, OUTPUT % bad)
    assert accepts(req, OUTPUT % GOOD.replace('"qty":1', '"qty":"x"')) is not strict


THINK = "<|message_model|><|content_thinking|>why<|end_message|>"
CALL = "<|message_model|>inv<|content_invoke_tool_json|>" + GOOD + "<|end_message|>"
F = '<|message_model|>f<|content_invoke_tool_json|>{"name":"f","args":{}}<|end_message|>'
TEXT = "<|message_model|><|content_text|>Hi<|end_message|>"
EOS = "<|content_model_end_sampling|>"
NAMED = {"type": "function", "function": {"name": "inv"}}


@pytest.mark.parametrize("kw,yes,no", [
    ({"tool_choice": "required"}, [CALL, THINK + THINK + CALL + F],
     ["", THINK, TEXT, TEXT + CALL, CALL + TEXT]),
    ({"tool_choice": "required", "parallel_tool_calls": False}, [THINK + CALL, F], [CALL + F]),
    ({"tool_choice": NAMED}, [CALL, THINK + CALL], [F, CALL + CALL, TEXT]),
    ({"tool_choice": "required", "response_format": {"type": "json_object"}}, [CALL], [TEXT]),
    ({"parallel_tool_calls": False}, [TEXT, THINK + CALL, TEXT + CALL], [CALL + F]),
    ({}, [TEXT, CALL + F + TEXT], [CALL.replace(GOOD, MISPLACED)]),
])
def test_tool_choice(tokenizer, accepts, monkeypatch, kw, yes, no):
    monkeypatch.setenv("HYPERINKLING_TOOL_GRAMMAR", "1")
    req = adjust(tokenizer, request(**kw))
    for text in yes:
        assert accepts(req, text + EOS), text
    for text in no:
        assert not accepts(req, text + EOS), text


def test_left_alone(tokenizer, monkeypatch):
    monkeypatch.setenv("HYPERINKLING_TOOL_GRAMMAR", "1")
    assert adjust(tokenizer, request(tool_choice="none")).structured_outputs is None
    schema = adjust(tokenizer, request(response_format={"type": "json_object"}))
    assert "token_triggered_tags" not in schema.structured_outputs.structural_tag
    monkeypatch.setenv("HYPERINKLING_TOOL_GRAMMAR", "0")
    assert adjust(tokenizer, request()).structured_outputs is None


@pytest.mark.parametrize("wrapper,args", [
    (GOOD, GOOD[len('{"name":"inv","args":'):-1]),
    (MISPLACED, MISPLACED[len('{"name":"inv","args":'):-1]),
    # an extra brace elsewhere leaves the arguments as they were
    ('{"name":"f","args":{"a":{"b":1}},"c":2}}', '{"a":{"b":1}}'),
    ('{"name":"f","args":{"a":1},"id":3}', '{"a":1}'),
    ('{"name":"f","args":{"a":1},"id":3', '{"a":1}'),
    ('{"name":"f","args":{"a":[{"b":1}},"c":2}', '{"a":[{"b":1}},"c":2}'),
    ('{"name":"f","args":{"a":1}', '{"a":1}'),
    ('{"name":"f"}', "{}"),
])
def test_arguments(wrapper, args):
    final = _inkling_arg_converter(wrapper, False)
    assert final == args
    for n in range(len(wrapper)):  # streamed deltas only ever extend
        assert final.startswith(_inkling_arg_converter(wrapper[:n], True)) or final == "{}"
