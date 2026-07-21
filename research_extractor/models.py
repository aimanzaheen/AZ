"""Data structures returned by the extractor."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Article:
    source_file: str
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    affiliations: list[str] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)
    abstract: str | None = None
    keywords: list[str] = field(default_factory=list)
    doi: str | None = None
    journal: str | None = None
    publication_year: str | None = None
    sections: dict[str, str] = field(default_factory=dict)
    references: list[str] = field(default_factory=list)
    word_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
