"""Extract structured information from research article files (PDF or text)."""

from research_extractor.models import Article
from research_extractor.parser import parse_article

__all__ = ["Article", "parse_article"]
__version__ = "0.1.0"
