"""JSON Schema validation for tool invocation arguments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from jsonschema import Draft202012Validator
from jsonschema.validators import validator_for


class ToolSchemaValidationError(ValueError):
    """Raised when a tool parameter schema cannot be evaluated safely."""


@dataclass(frozen=True, slots=True)
class ToolArgumentIssue:
    """A safe, structured argument-validation issue."""

    path: str
    keyword: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "keyword": self.keyword,
            "message": self.message,
        }


def _json_pointer(path: Iterable[Any]) -> str:
    parts = [str(part).replace("~", "~0").replace("/", "~1") for part in path]
    return "/" + "/".join(parts) if parts else "/"


def _required_property(error: Any) -> str | None:
    if error.validator != "required" or not isinstance(error.instance, dict):
        return None
    required = error.validator_value
    if not isinstance(required, list):
        return None
    missing = [name for name in required if name not in error.instance]
    return next(
        (
            name
            for name in missing
            if error.message == f"{name!r} is a required property"
        ),
        missing[0] if missing else None,
    )


def _additional_property(error: Any) -> str | None:
    if error.validator != "additionalProperties" or not isinstance(
        error.instance, dict
    ):
        return None
    schema = error.schema if isinstance(error.schema, dict) else {}
    declared = schema.get("properties", {})
    declared_names = set(declared) if isinstance(declared, dict) else set()
    patterns = schema.get("patternProperties", {})
    compiled_patterns = []
    if isinstance(patterns, dict):
        for pattern in patterns:
            try:
                compiled_patterns.append(re.compile(pattern))
            except re.error:
                continue
    return next(
        (
            str(name)
            for name in sorted(error.instance, key=str)
            if name not in declared_names
            and not any(pattern.search(str(name)) for pattern in compiled_patterns)
        ),
        None,
    )


def _safe_issue(error: Any) -> ToolArgumentIssue:
    keyword = str(error.validator or "schema")
    path = list(error.absolute_path)
    missing_property = _required_property(error)
    additional_property = _additional_property(error)
    if missing_property is not None:
        path.append(missing_property)
        message = f"required property {missing_property!r} is missing"
    elif additional_property is not None:
        path.append(additional_property)
        message = f"property {additional_property!r} is not declared"
    elif keyword == "type":
        message = f"expected type {error.validator_value!r}"
    elif keyword == "enum":
        message = f"must be one of {error.validator_value!r}"
    elif keyword == "const":
        message = "must equal the schema's constant value"
    elif keyword == "additionalProperties":
        message = "contains undeclared properties"
    elif keyword == "minLength":
        message = f"must contain at least {error.validator_value} characters"
    elif keyword == "maxLength":
        message = f"must contain at most {error.validator_value} characters"
    elif keyword == "minItems":
        message = f"must contain at least {error.validator_value} items"
    elif keyword == "maxItems":
        message = f"must contain at most {error.validator_value} items"
    elif keyword == "minimum":
        message = f"must be greater than or equal to {error.validator_value}"
    elif keyword == "maximum":
        message = f"must be less than or equal to {error.validator_value}"
    elif keyword in {"oneOf", "anyOf"}:
        message = "does not match any allowed schema"
    elif keyword == "allOf":
        message = "does not satisfy all required schemas"
    elif keyword == "pattern":
        message = "does not match the required pattern"
    else:
        message = f"does not satisfy schema rule {keyword!r}"
    return ToolArgumentIssue(
        path=_json_pointer(path),
        keyword=keyword,
        message=message,
    )


def validate_tool_arguments(
    schema: dict[str, Any],
    arguments: dict[str, Any],
    *,
    max_issues: int = 10,
) -> tuple[ToolArgumentIssue, ...]:
    """Return deterministic, value-redacted validation issues."""

    try:
        validator_class = validator_for(schema, default=Draft202012Validator)
        validator_class.check_schema(schema)
        validator = validator_class(schema)
        errors = sorted(
            validator.iter_errors(arguments),
            key=lambda error: (
                _json_pointer(error.absolute_path),
                str(error.validator or ""),
                error.message,
            ),
        )
        return tuple(_safe_issue(error) for error in errors[:max_issues])
    except Exception:
        # jsonschema exceptions can render the complete instance, including
        # secrets from tool arguments. Never let those details cross the
        # invocation boundary.
        raise ToolSchemaValidationError(
            "Tool parameter schema is invalid."
        ) from None
