"""Anthropic API wrapper that forces structured JSON output via tool-use.

Kept separate from reextract.py so the call itself (and its retry/backoff
behavior) can be unit-tested with a fake client, without hitting the network.
"""

from __future__ import annotations

import time

import anthropic

DEFAULT_MODEL = "claude-sonnet-5"
MAX_RETRIES = 5

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

SYSTEM_PROMPT = (
    "You are a meticulous scientific data extraction assistant reviewing a neuroscience "
    "paper. You will be given the paper's text (full text, or an abstract only if that is "
    "all that was available - this will be labeled) followed by extraction instructions "
    "written by the research team. Follow the instructions exactly, including their "
    "row/table structure. If a value is not explicitly stated in the provided text, write "
    "'not explicitly stated' - never estimate or infer a number that isn't given. Call the "
    "record_extraction tool exactly once with your complete answer."
)


def build_user_message(paper_text: str, text_source: str, instructions: str) -> str:
    source_note = {
        "pmc_fulltext": "FULL TEXT (from PubMed Central)",
        "pubmed_abstract": "ABSTRACT ONLY (full text was not available - treat missing "
        "details as 'not explicitly stated' rather than guessing from an abstract-limited view)",
    }.get(text_source, text_source or "UNKNOWN SOURCE")

    return (
        f"<paper_text source=\"{source_note}\">\n{paper_text}\n</paper_text>\n\n"
        f"<extraction_instructions>\n{instructions}\n</extraction_instructions>"
    )


def call_extraction(
    client: anthropic.Anthropic,
    model: str,
    paper_text: str,
    text_source: str,
    instructions: str,
    max_tokens: int = 8000,
) -> dict:
    """Call the Anthropic API and return the parsed record_extraction tool input.

    Retries on rate-limit/overload/connection errors with exponential backoff.
    """
    user_message = build_user_message(paper_text, text_source, instructions)

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                tools=[EXTRACTION_TOOL],
                tool_choice={"type": "tool", "name": "record_extraction"},
                messages=[{"role": "user", "content": user_message}],
            )
            for block in response.content:
                if block.type == "tool_use" and block.name == "record_extraction":
                    return block.input
            raise RuntimeError("Model response did not include a record_extraction tool call.")
        except (
            anthropic.RateLimitError,
            anthropic.OverloadedError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
        ) as exc:
            last_exc = exc
            if attempt == MAX_RETRIES - 1:
                break
            time.sleep(2**attempt)

    raise RuntimeError(f"Extraction call failed after {MAX_RETRIES} attempts: {last_exc}") from last_exc
