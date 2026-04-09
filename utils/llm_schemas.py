from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    entry_id: str
    status: Literal["valid", "partially_valid", "invalid"]
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    suggested_fixes: dict[str, Any] = Field(default_factory=dict)


class ValidationEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[ValidationResult] = Field(default_factory=list)


class CorrectionChange(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    field: str
    original: Any = ""
    corrected: Any = ""


class CorrectionEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entry_id: str
    corrected_bibtex: str = ""
    corrections_applied: list[CorrectionChange] = Field(default_factory=list)


class CorrectionEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore")

    markdown_summary: str = ""
    corrected_entries: list[CorrectionEntry] = Field(default_factory=list)


def parse_validation_results(payload: dict[str, Any]) -> tuple[list[dict], list[str]]:
    """Validate validation JSON payload and return normalized result dicts."""
    errors: list[str] = []
    validated: list[dict] = []

    try:
        envelope = ValidationEnvelope.model_validate(payload)
        return [item.model_dump() for item in envelope.results], errors
    except ValidationError as exc:
        errors.append(f"ValidationEnvelope error: {exc}")

    raw_results = payload.get("results", [])
    if not isinstance(raw_results, list):
        return [], errors

    for idx, item in enumerate(raw_results):
        if not isinstance(item, dict):
            errors.append(f"results[{idx}] is not an object")
            continue
        try:
            normalized = ValidationResult.model_validate(item)
            validated.append(normalized.model_dump())
        except ValidationError as exc:
            errors.append(f"results[{idx}] error: {exc}")

    return validated, errors


def parse_correction_payload(payload: dict[str, Any]) -> tuple[str, list[dict], list[str]]:
    """
    Validate and normalize correction payload.
    Accepts common alternative top-level and entry-level key names.
    """
    errors: list[str] = []

    markdown = payload.get("markdown_summary", "")
    if not isinstance(markdown, str):
        markdown = ""

    raw_entries = payload.get("corrected_entries", [])
    if not isinstance(raw_entries, list):
        for key in ("entries", "results", "corrections"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                raw_entries = candidate
                break
        else:
            raw_entries = []

    normalized_entries: list[dict] = []
    for idx, item in enumerate(raw_entries):
        if not isinstance(item, dict):
            errors.append(f"entries[{idx}] is not an object")
            continue

        mapped = {
            "entry_id": item.get("entry_id") or item.get("id"),
            "corrected_bibtex": item.get("corrected_bibtex") or item.get("bibtex") or "",
            "corrections_applied": (
                item.get("corrections_applied")
                or item.get("changes")
                or item.get("corrections")
                or []
            ),
        }

        try:
            parsed = CorrectionEntry.model_validate(mapped)
            normalized_entries.append(parsed.model_dump())
        except ValidationError as exc:
            errors.append(f"entries[{idx}] error: {exc}")

    return markdown, normalized_entries, errors
