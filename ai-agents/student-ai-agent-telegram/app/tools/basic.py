"""Basic tools: calculator and current time."""

import ast
import datetime as dt

from langchain_core.tools import tool


class SafeCalculatorError(ValueError):
    """Понятная ошибка для недопустимых математических выражений."""


def _safe_eval_math(expression: str) -> int | float:
    """Вычисляет только простую арифметику без доступа к Python-объектам."""
    if not isinstance(expression, str) or not expression.strip():
        raise SafeCalculatorError("пустое выражение")
    if len(expression) > 200:
        raise SafeCalculatorError("выражение слишком длинное")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise SafeCalculatorError(f"некорректный синтаксис: {e.msg}") from None

    def eval_node(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise SafeCalculatorError("разрешены только числа")
            return node.value

        if isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise SafeCalculatorError("разрешены только унарные + и -")

        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)

            try:
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.FloorDiv):
                    return left // right
                if isinstance(node.op, ast.Mod):
                    return left % right
                if isinstance(node.op, ast.Pow):
                    if abs(right) > 1000:
                        raise SafeCalculatorError("слишком большая степень")
                    return left ** right
            except ZeroDivisionError:
                raise SafeCalculatorError("деление на ноль") from None
            except OverflowError:
                raise SafeCalculatorError("результат слишком большой") from None

            raise SafeCalculatorError("разрешены только +, -, *, /, //, %, **")

        raise SafeCalculatorError("разрешены только числа, скобки и математические операторы")

    return eval_node(tree)


@tool
def calculator(expression: str) -> str:
    """Выполняет математические вычисления. Принимает выражение в виде строки."""
    try:
        result = _safe_eval_math(expression)
        return f"Результат: {result}"
    except SafeCalculatorError as e:
        return f"Ошибка вычисления: {e}."
    except Exception:
        return "Ошибка вычисления: выражение не удалось обработать."


@tool
def get_current_time() -> str:
    """Возвращает текущее время."""
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
