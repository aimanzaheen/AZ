"""Anthropic API wrapper that forces structured JSON output via tool-use.

Kept separate from reextract.py/verify.py so the calls themselves (and their
retry/backoff behavior) can be unit-tested with a fake client, without
hitting the network.
"""

from __future__ import annotations

import time

import anthropic

DEFAULT_MODEL = "claude-sonnet-5"
MAX_RETRIES = 5

RETRYABLE_ERRORS = (
    anthropic.RateLimitError,
    anthropic.OverloadedError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)

EXTRACTION_TOOL = {
    "name": "record_extraction",
    "description": "Record the structured result of the extraction instructions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "description": "One entry per distinct item the instructions ask you to "
                "enumerate (e.g. one row per traced pathway, per population-per-epoch, or "
                "per cell type/group). Leave empty only if the instructions ask a single "
                "question with no table.",
                "items": {
                    "type": "object",
                    "properties": {
                        "row_label": {
                            "type": "string",
                            "description": "Short human-readable label for what this row "
                            "covers, e.g. 'ZI (Vgat+) -> PVT (anterograde AAV)'.",
                        },
                        "figure_ref": {
                            "type": "string",
                            "description": "Figure/table reference for this row's data, or "
                            "'not explicitly stated'.",
                        },
                        "source_quote": {
                            "type": "string",
                            "description": "A short verbatim quote from the provided text "
                            "supporting this row, or 'not explicitly stated'.",
                        },
                        "fields": {
                            "type": "object",
                            "description": "Every itemized sub-field the instructions "
                            "requested for this row, as key: value string pairs, using the "
                            "instructions' own field names as keys.",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                    "required": ["row_label", "fields"],
                },
            },
            "summary_paragraph": {
                "type": "string",
                "description": "The free-text summary/paragraph the instructions ask for "
                "(if any), or the direct answer when the instructions pose a single question "
                "rather than requesting a table.",
            },
        },
        "required": ["rows", "summary_paragraph"],
    },
}

EXTRACTION_SYSTEM_PROMPT = (
    "You are a meticulous scientific data extraction assistant reviewing a neuroscience "
    "paper. You will be given the paper's text (full text, or an abstract only if that is "
    "all that was available - this will be labeled) followed by extraction instructions "
    "written by the research team. Follow the instructions exactly, including their "
    "row/table structure. If a value is not explicitly stated in the provided text, write "
    "'not explicitly stated' - never estimate or infer a number that isn't given. Call the "
    "record_extraction tool exactly once with your complete answer."
)

VERIFICATION_TOOL = {
    "name": "record_verification",
    "description": "Record a verdict on whether each of Otto's original extracted values "
    "is accurate and reasonably complete given the paper text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "description": "One entry per Otto field being audited.",
                "items": {
                    "type": "object",
                    "properties": {
                        "otto_field": {
                            "type": "string",
                            "description": "Exactly one of the field names given in "
                            "<otto_original_values>.",
                        },
                        "verdict": {
                            "type": "string",
                            "enum": ["match", "partial", "mismatch", "unverifiable"],
                            "description": "match = accurate and reasonably complete. "
                            "partial = correct as far as it goes but missing something the "
                            "paper reports. mismatch = wrong, unsupported, or contradicted by "
                            "the text. unverifiable = the provided text doesn't contain enough "
                            "information to judge (e.g. abstract-only).",
                        },
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "explanation": {
                            "type": "string",
                            "description": "One or two sentences justifying the verdict.",
                        },
                        "evidence_quote": {
                            "type": "string",
                            "description": "A short verbatim quote from the paper text that "
                            "most directly supports or contradicts Otto's value, or 'not "
                            "explicitly stated'.",
                        },
                    },
                    "required": ["otto_field", "verdict", "confidence", "explanation"],
                },
            }
        },
        "required": ["verdicts"],
    },
}

VERIFICATION_SYSTEM_PROMPT = (
    "You are auditing a research-extraction team's work on a neuroscience paper. You will "
    "be given the paper's text (full text, or an abstract only if that is all that was "
    "available - this will be labeled), a fresh independent re-extraction performed "
    "separately by another reviewer, and the original team's ('Otto's') recorded value for "
    "one or more fields. For EACH field listed in <otto_original_values>, decide whether "
    "Otto's value is accurate and reasonably complete given the actual paper text - use the "
    "fresh re-extraction as a helpful cross-check, but base your verdict on the paper text "
    "itself, not merely on agreement with the re-extraction (the re-extraction can also be "
    "wrong). An empty or blank Otto value is 'match' only if the paper genuinely reports "
    "nothing for that field; otherwise it's 'mismatch' or 'partial'. Call record_verification "
    "exactly once, with one verdict entry per field in <otto_original_values>."
)


def _source_note(text_source: str) -> str:
    return {
        "pmc_fulltext": "FULL TEXT (from PubMed Central)",
        "pubmed_abstract": "ABSTRACT ONLY (full text was not available - treat missing "
        "details as 'not explicitly stated' rather than guessing from an abstract-limited view)",
        "local_pdf": "FULL TEXT (extracted from a locally supplied PDF)",
        "local_txt": "FULL TEXT (from a locally supplied text file)",
    }.get(text_source, text_source or "UNKNOWN SOURCE")


def build_user_message(paper_text: str, text_source: str, instructions: str) -> str:
    return (
        f'<paper_text source="{_source_note(text_source)}">\n{paper_text}\n</paper_text>\n\n'
        f"<extraction_instructions>\n{instructions}\n</extraction_instructions>"
    )


def build_verification_user_message(
    paper_text: str,
    text_source: str,
    reextraction_summary: str,
    otto_values: dict[str, str],
) -> str:
    otto_block = "\n".join(f'{name}: "{value}"' for name, value in otto_values.items())
    return (
        f'<paper_text source="{_source_note(text_source)}">\n{paper_text}\n</paper_text>\n\n'
        f"<fresh_independent_reextraction>\n{reextraction_summary}\n</fresh_independent_reextraction>\n\n"
        f"<otto_original_values>\n{otto_block}\n</otto_original_values>"
    )


def _call_tool(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    tool: dict,
    user_message: str,
    max_tokens: int,
) -> dict:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                tools=[tool],
                tool_choice={"type": "tool", "name": tool["name"]},
                messages=[{"role": "user", "content": user_message}],
            )
            for block in response.content:
                if block.type == "tool_use" and block.name == tool["name"]:
                    return block.input
            raise RuntimeError(f"Model response did not include a {tool['name']} tool call.")
        except RETRYABLE_ERRORS as exc:
            last_exc = exc
            if attempt == MAX_RETRIES - 1:
                break
            time.sleep(2**attempt)

    raise RuntimeError(f"{tool['name']} call failed after {MAX_RETRIES} attempts: {last_exc}") from last_exc


def call_extraction(
    client: anthropic.Anthropic,
    model: str,
    paper_text: str,
    text_source: str,
    instructions: str,
    max_tokens: int = 8000,
) -> dict:
    """Call the Anthropic API and return the parsed record_extraction tool input."""
    user_message = build_user_message(paper_text, text_source, instructions)
    return _call_tool(client, model, EXTRACTION_SYSTEM_PROMPT, EXTRACTION_TOOL, user_message, max_tokens)


def call_verification(
    client: anthropic.Anthropic,
    model: str,
    paper_text: str,
    text_source: str,
    reextraction_summary: str,
    otto_values: dict[str, str],
    max_tokens: int = 4000,
) -> dict:
    """Call the Anthropic API and return the parsed record_verification tool input."""
    user_message = build_verification_user_message(paper_text, text_source, reextraction_summary, otto_values)
    return _call_tool(client, model, VERIFICATION_SYSTEM_PROMPT, VERIFICATION_TOOL, user_message, max_tokens)
