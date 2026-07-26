"""Tests for the safe calculator tool."""

from app.tools.basic import calculator


def calculate(expression: str) -> str:
    return calculator.invoke({"expression": expression})


def test_calculator_addition():
    assert calculate("2 + 2") == "Результат: 4"


def test_calculator_parentheses_and_multiplication():
    assert calculate("2 * (3 + 4)") == "Результат: 14"


def test_calculator_division():
    assert calculate("10 / 2") == "Результат: 5.0"


def test_calculator_division_by_zero_returns_clear_error():
    result = calculate("10 / 0")
    assert result.startswith("Ошибка вычисления:")
    assert "деление на ноль" in result


def test_calculator_blocks_imports():
    result = calculate("__import__('os').system('echo hacked')")
    assert result.startswith("Ошибка вычисления:")


def test_calculator_blocks_open_call():
    result = calculate("open('x')")
    assert result.startswith("Ошибка вычисления:")


def test_calculator_blocks_variables():
    result = calculate("x + 1")
    assert result.startswith("Ошибка вычисления:")


def test_calculator_blocks_strings():
    result = calculate("'hello'")
    assert result.startswith("Ошибка вычисления:")
