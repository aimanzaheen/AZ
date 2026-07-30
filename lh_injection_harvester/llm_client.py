"""Anthropic API wrapper that forces structured JSON output via tool-use.

Kept separate from extract.py so the calls themselves (and their retry/backoff behavior) can be
unit-tested with a fake client, without hitting the network.
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

EXPERIMENT_FIELDS = [
    "volume_injected_lh",
    "stereotaxic_coordinates",
    "tracer",
    "projections_found",
    "survival_time",
]

RECORD_LH_INJECTIONS_TOOL = {
    "name": "record_lh_injections",
    "description": "Record whether this paper reports retrograde tracer injection(s) into the "
    "mouse lateral hypothalamus (LH), and the extracted data for each such experiment.",
    "input_schema": {
        "type": "object",
        "properties": {
            "lh_retrograde_injection_present": {
                "type": "boolean",
                "description": "True if the paper reports at least one retrograde tracer "
                "injection placed in the mouse LH.",
            },
            "experiments": {
                "type": "array",
                "description": "One entry per distinct LH retrograde injection experiment "
                "(different tracer, coordinates, or mouse line). Empty if "
                "lh_retrograde_injection_present is false.",
                "items": {
                    "type": "object",
                    "properties": {
                        "volume_injected_lh": {
                            "type": "string",
                            "description": "Injection volume placed in the LH, or 'not "
                            "explicitly stated'.",
                        },
                        "stereotaxic_coordinates": {
                            "type": "string",
                            "description": "AP/ML/DV stereotaxic coordinates used to target "
                            "the LH, or 'not explicitly stated'.",
                        },
                        "tracer": {
                            "type": "string",
                            "description": "The specific tracer/virus and conjugate/serotype "
                            "used, or 'not explicitly stated'.",
                        },
                        "projections_found": {
                            "type": "string",
                            "description": "Regions reported as projecting to the LH based on "
                            "this retrograde tracer, or 'not explicitly stated'.",
                        },
                        "survival_time": {
                            "type": "string",
                            "description": "Survival time between injection and perfusion, or "
                            "'not explicitly stated'.",
                        },
                        "figure_ref": {
                            "type": "string",
                            "description": "Figure/table reference, or 'not explicitly stated'.",
                        },
                        "source_quote": {
                            "type": "string",
                            "description": "Short verbatim supporting quote, or 'not explicitly "
                            "stated'.",
                        },
                    },
                    "required": EXPERIMENT_FIELDS + ["figure_ref", "source_quote"],
                },
            },
            "notes": {
                "type": "string",
                "description": "One sentence: a caveat for a genuine match, or the reason this "
                "paper doesn't fit for a non-match.",
            },
        },
        "required": ["lh_retrograde_injection_present", "experiments", "notes"],
    },
}

EXTRACTION_SYSTEM_PROMPT = (
    "You are a meticulous scientific data extraction assistant reviewing a neuroscience paper. "
    "You will be given the paper's text (full text, or an abstract only if that is all that was "
    "available - this will be labeled) followed by extraction instructions written by the "
    "research team. Follow the instructions exactly. If a value is not explicitly stated in the "
    "provided text, write 'not explicitly stated' - never estimate or infer a number that isn't "
    "given. Call the record_lh_injections tool exactly once with your complete answer."
)


def _source_note(text_source: str) -> str:
    return {
        "pmc_fulltext": "FULL TEXT (from PubMed Central)",
        "pubmed_abstract": "ABSTRACT ONLY (full text was not available - treat missing details "
        "as 'not explicitly stated' rather than guessing from an abstract-limited view)",
    }.get(text_source, text_source or "UNKNOWN SOURCE")


def build_user_message(paper_text: str, text_source: str, instructions: str) -> str:
    return (
        f'<paper_text source="{_source_note(text_source)}">\n{paper_text}\n</paper_text>\n\n'
        f"<extraction_instructions>\n{instructions}\n</extraction_instructions>"
    )


def call_extraction(
    client: anthropic.Anthropic,
    model: str,
    paper_text: str,
    text_source: str,
    instructions: str,
    max_tokens: int = 4000,
) -> dict:
    """Call the Anthropic API and return the parsed record_lh_injections tool input."""
    user_message = build_user_message(paper_text, text_source, instructions)

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=EXTRACTION_SYSTEM_PROMPT,
                tools=[RECORD_LH_INJECTIONS_TOOL],
                tool_choice={"type": "tool", "name": RECORD_LH_INJECTIONS_TOOL["name"]},
                messages=[{"role": "user", "content": user_message}],
            )
            for block in response.content:
                if block.type == "tool_use" and block.name == RECORD_LH_INJECTIONS_TOOL["name"]:
                    return block.input
            raise RuntimeError("Model response did not include a record_lh_injections tool call.")
        except RETRYABLE_ERRORS as exc:
            last_exc = exc
            if attempt == MAX_RETRIES - 1:
                break
            time.sleep(2**attempt)

    raise RuntimeError(f"record_lh_injections call failed after {MAX_RETRIES} attempts: {last_exc}") from last_exc
