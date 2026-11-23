"""disable_any_whitespace inside structural tags, which is how Inkling's parser asks for JSON
(after the reasoning): JSON comes out compact. No GPU.

    venv/bin/python -m pytest -q tests/test_json_whitespace.py
"""

import json

import pytest
import xgrammar as xgr
from xgrammar.structural_tag import (
    AnyTextFormat,
    JSONSchemaFormat,
    SequenceFormat,
    StarFormat,
    TagFormat,
)
from xgrammar.testing import _is_grammar_accept_string as accepts

from vllm.v1.structured_output.backend_xgrammar import _without_json_whitespace

SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "string"}, "b": {"type": "array", "items": {"type": "integer"}},
                   "c": {"type": "object", "properties": {"d": {"type": "boolean"}},
                         "required": ["d"]}},
    "required": ["a", "b", "c"], "additionalProperties": False,
}
VALUE = {"a": "due  spazi, e: \n", "b": [1, 2, 3], "c": {"d": True}}
COMPACT = json.dumps(VALUE, separators=(",", ":"))


def _tag(schema):
    think = TagFormat(begin="<think>", content=AnyTextFormat(excludes=["</think>"]), end="</think>")
    return xgr.StructuralTag(format=SequenceFormat(
        elements=[StarFormat(content=think), JSONSchemaFormat(json_schema=schema)]))


@pytest.mark.parametrize("schema", [SCHEMA, {"type": "object"}])
@pytest.mark.parametrize("prefix", ["", "<think>perche' no</think>"])
def test_compact_json_after_reasoning(schema, prefix):
    grammar = xgr.Grammar.from_structural_tag(_without_json_whitespace(_tag(schema)))
    assert accepts(grammar, prefix + COMPACT)
    for text in (json.dumps(VALUE), json.dumps(VALUE, indent=2), " " + COMPACT, COMPACT + " "):
        assert not accepts(grammar, prefix + text)
    assert accepts(xgr.Grammar.from_structural_tag(_tag(schema)), prefix + json.dumps(VALUE))


def test_several_schemas_do_not_clash():
    """Two converted nodes in one tag: their grammars' rule names must not collide."""
    other = {"type": "object", "properties": {"z": {"type": "number"}}, "required": ["z"],
             "additionalProperties": False}
    tag = xgr.StructuralTag(format=SequenceFormat(
        elements=[JSONSchemaFormat(json_schema=SCHEMA), JSONSchemaFormat(json_schema=other)]))
    grammar = xgr.Grammar.from_structural_tag(_without_json_whitespace(tag))
    assert accepts(grammar, COMPACT + '{"z":1.5}')
    assert not accepts(grammar, '{"z":1.5}' + COMPACT)
