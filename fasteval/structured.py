"""Helpers for converting the compact CLI schema syntax to JSON Schema."""

import re
from typing import Any


_TYPE_MAP: dict[str, dict[str, str]] = {
    "str": {"type": "string"},
    "string": {"type": "string"},
    "int": {"type": "integer"},
    "integer": {"type": "integer"},
    "float": {"type": "number"},
    "number": {"type": "number"},
    "bool": {"type": "boolean"},
    "boolean": {"type": "boolean"},
    "object": {"type": "object"},
    "any": {},
}


def shorthand_to_schema(spec: str) -> dict[str, Any]:
    """Convert ``name:type,name:type`` into a strict JSON Schema.

    ``?`` marks an optional field and ``[]`` marks an array, e.g.
    ``title:str,tags:str[],score:float?,metadata:object``.
    """
    if not spec or not spec.strip():
        raise ValueError("Structured output schema cannot be empty")

    properties: dict[str, Any] = {}
    required: list[str] = []
    for raw_field in _split_fields(spec):
        field = raw_field.strip()
        if not field:
            raise ValueError("Structured output schema contains an empty field")
        if ":" not in field:
            raise ValueError(f"Invalid field '{field}'. Expected name:type")
        name, type_name = (part.strip() for part in field.split(":", 1))
        if not name:
            raise ValueError("Structured output field name cannot be empty")
        if name in properties:
            raise ValueError(f"Duplicate structured output field: '{name}'")
        description = None
        description_match = re.fullmatch(r"(.+?)\s*\(\s*(['\"])(.*?)\2\s*\)\s*(\?)?", type_name)
        if description_match:
            type_name, _, description, optional_suffix = description_match.groups()
            if optional_suffix:
                type_name = f"{type_name}?"
        type_name = type_name.strip()
        optional = type_name.endswith("?")
        type_name = type_name.removesuffix("?").strip()
        is_array = type_name.endswith("[]")
        type_name = type_name.removesuffix("[]").strip().lower()
        if type_name not in _TYPE_MAP:
            supported = ", ".join(sorted(_TYPE_MAP))
            raise ValueError(f"Unknown type '{type_name}' for field '{name}'. Supported: {supported}")
        schema = dict(_TYPE_MAP[type_name])
        if is_array:
            schema = {"type": "array", "items": schema}
        if description:
            schema["description"] = description
        properties[name] = schema
        if not optional:
            required.append(name)

    if not properties:
        raise ValueError("Structured output schema has no fields")
    result: dict[str, Any] = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        result["required"] = required
    return result


def _split_fields(spec: str) -> list[str]:
    """Split fields on commas outside description parentheses."""
    fields: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    for index, char in enumerate(spec):
        if char in "'\"":
            quote = None if quote == char else char if quote is None else quote
        elif quote is None and char == "(":
            depth += 1
        elif quote is None and char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced description parentheses")
        elif quote is None and char == "," and depth == 0:
            fields.append(spec[start:index])
            start = index + 1
    if depth != 0 or quote is not None:
        raise ValueError("Unclosed description in structured output schema")
    fields.append(spec[start:])
    return fields
