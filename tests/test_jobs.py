"""Tests for hh.ru helper parsing."""

from app.tools.jobs import _parse_count


def test_parse_count_plain_number():
    assert _parse_count("10") == 10


def test_parse_count_number_inside_text():
    assert _parse_count("покажи 2 вакансии") == 2


def test_parse_count_empty_string_returns_default():
    assert _parse_count("", default=7) == 7


def test_parse_count_limits_large_number():
    assert _parse_count("999", default=10, max_val=20) == 20


def test_parse_count_invalid_string_returns_default():
    assert _parse_count("без числа", default=6) == 6
