"""Tests for src/demo_texts.py: both languages complete, placeholders match, Turkish dates and numbers."""

from datetime import date
from string import Formatter

from src.demo_texts import LANGUAGES, TEXTS, hour_text, long_date, number, percent, short_date, text


def placeholders(template: str) -> set:
    return {field for _, field, _, _ in Formatter().parse(template) if field}


def test_every_language_has_the_same_keys_and_placeholders():
    english = TEXTS["en"]
    for lang in LANGUAGES:
        assert TEXTS[lang].keys() == english.keys()
        for key, template in TEXTS[lang].items():
            assert placeholders(template) == placeholders(english[key]), (lang, key)


def test_every_text_formats_with_its_placeholders():
    for lang in LANGUAGES:
        for key, template in TEXTS[lang].items():
            text(lang, key, **{name: 1 for name in placeholders(template)})   # raises on a stray brace


def test_plotly_placeholders_survive_formatting():
    assert text("en", "hover_hour") == "%{x}:00 — %{y:.1f} USD/MWh"


def test_text_fills_placeholders():
    assert text("tr", "hits", n=4) == "En ucuz 4 saati bulma"
    assert text("en", "hours_used", hours="10:00–14:00") == "hours: 10:00–14:00"


def test_percent_and_hours():
    assert percent(0.775, "en") == "78%" and percent(0.775, "tr") == "%78"
    assert hour_text([(10, 14), (18, 19)]) == "10:00–14:00, 18:00–19:00"


def test_dates_per_language():
    sunday = date(2026, 10, 11)
    assert long_date(sunday, "en") == "Sunday 11 October 2026"
    assert long_date(sunday, "tr") == "11 Ekim 2026 Pazar"
    assert long_date(sunday, "tr", year=False) == "11 Ekim Pazar"
    assert short_date(sunday, "tr", year=True) == "11 Eki 2026"


def test_numbers_per_language():
    assert number(1308.46, "en") == "1,308"
    assert number(1308.46, "tr") == "1.308"
    assert number(7.735, "tr", decimals=2) == "7,74"
    assert number(-146, "tr") == "-146"
