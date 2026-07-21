from pathlib import Path

from research_extractor.parser import parse_article, parse_text

FIXTURE = Path(__file__).parent / "fixtures" / "sample_article.txt"


def test_parse_text_extracts_core_metadata():
    text = FIXTURE.read_text()
    article = parse_text(text, source_file="sample_article.txt")

    assert article.title == (
        "Deep Learning Approaches for Automated Extraction of Scientific Knowledge"
    )
    assert "Aiman Zaheen" in article.authors
    assert article.doi == "10.1234/jamlr.2024.00567"
    assert article.publication_year == "2024"
    assert "aiman.zaheen@example.edu" in article.emails
    assert "information extraction" in article.keywords
    assert article.abstract is not None
    assert "lightweight pipeline" in article.abstract


def test_parse_text_extracts_sections():
    text = FIXTURE.read_text()
    article = parse_text(text, source_file="sample_article.txt")

    assert set(["introduction", "methods", "results", "discussion", "conclusion"]).issubset(
        article.sections
    )
    assert "rule-based parser" in article.sections["methods"]
    assert "92%" in article.sections["results"]


def test_parse_text_extracts_references():
    text = FIXTURE.read_text()
    article = parse_text(text, source_file="sample_article.txt")

    assert len(article.references) == 3
    assert any("Text mining for scientific literature" in ref for ref in article.references)


def test_parse_article_reads_txt_file():
    article = parse_article(FIXTURE)
    assert article.source_file == str(FIXTURE)
    assert article.word_count > 50
