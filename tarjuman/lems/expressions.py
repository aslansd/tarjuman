"""A parser for LEMS expressions, compiling them to JAX closures.

NeuroML's standard component types cover the textbook cases, but real models —
ChannelML conversions, most of the Open Source Brain library, and every
parameter set of c302 — define their own ``<ComponentType>`` elements whose
``<Dynamics>`` blocks contain arbitrary infix expressions::

    <DerivedVariable name="inf" value="1 / (1 + (exp((ca_half - caConc) / k)))"/>
    <TimeDerivative variable="concentration"
                    value="(iCa/surfaceArea) * rho - ((concentration - restingConc) / decayConstant)"/>

This module turns such a string into a function of a context dictionary, built
out of ``jax.numpy`` operations so that it stays differentiable and
``jit``-able.

Supported syntax:

* numbers (including ``1e-8`` and ``.5``), identifiers, parentheses
* ``+ - * / ^`` with the usual precedence, ``^`` right-associative
* comparisons ``.gt. .lt. .geq. .leq. .eq. .ne.`` and their symbolic forms
* boolean ``.and. .or.``, negation ``!``
* the LEMS function set: ``exp ln log sin cos tan asin acos atan sinh cosh
  tanh sqrt abs ceil floor H random``

Comparisons return 1.0/0.0 rather than booleans, which is what LEMS means by
them and what ``jnp.where`` wants.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable

import jax.numpy as jnp

from ..errors import ParseError

__all__ = ["compile_expression", "parse_expression", "expression_symbols", "FUNCTIONS"]


def _heaviside(x):
    return jnp.where(x > 0, 1.0, 0.0)


#: LEMS built-in functions, as JAX callables.
FUNCTIONS: dict[str, Callable] = {
    "exp": jnp.exp,
    "ln": jnp.log,
    "log": jnp.log10,
    "log10": jnp.log10,
    "sin": jnp.sin,
    "cos": jnp.cos,
    "tan": jnp.tan,
    "asin": jnp.arcsin,
    "acos": jnp.arccos,
    "atan": jnp.arctan,
    "sinh": jnp.sinh,
    "cosh": jnp.cosh,
    "tanh": jnp.tanh,
    "sqrt": jnp.sqrt,
    "abs": jnp.abs,
    "ceil": jnp.ceil,
    "floor": jnp.floor,
    "H": _heaviside,
}

_WORD_OPERATORS = {
    ".gt.": ">",
    ".lt.": "<",
    ".geq.": ">=",
    ".leq.": "<=",
    ".eq.": "==",
    ".ne.": "!=",
    ".neq.": "!=",
    ".and.": "and",
    ".or.": "or",
}

_TOKEN_RE = re.compile(
    r"""
    (?P<number>(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)
  | (?P<word_op>\.(?:gt|lt|geq|leq|eq|ne|neq|and|or)\.)
  | (?P<name>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<op><=|>=|==|!=|&&|\|\||[-+*/^()<>,!])
  | (?P<space>\s+)
    """,
    re.VERBOSE,
)

_COMPARISONS = {
    ">": jnp.greater,
    "<": jnp.less,
    ">=": jnp.greater_equal,
    "<=": jnp.less_equal,
    "==": jnp.equal,
    "!=": jnp.not_equal,
}


@dataclass
class _Token:
    kind: str
    text: str
    position: int


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    position = 0
    while position < len(text):
        match = _TOKEN_RE.match(text, position)
        if match is None:
            raise ParseError(
                f"Cannot parse LEMS expression {text!r}: unexpected character "
                f"{text[position]!r} at position {position}."
            )
        kind = match.lastgroup
        value = match.group()
        position = match.end()
        if kind == "space":
            continue
        if kind == "word_op":
            value = _WORD_OPERATORS[value.lower()]
            kind = "op"
        elif kind == "op" and value in ("&&", "||"):
            value = "and" if value == "&&" else "or"
        tokens.append(_Token(kind, value, match.start()))
    return tokens


class _Parser:
    """A small precedence-climbing parser producing JAX closures directly."""

    def __init__(self, text: str):
        self.text = text
        self.tokens = _tokenize(text)
        self.index = 0
        self.symbols: set[str] = set()

    # -- token helpers ---------------------------------------------------- #
    def peek(self) -> _Token | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def take(self) -> _Token:
        token = self.peek()
        if token is None:
            raise ParseError(f"Unexpected end of LEMS expression {self.text!r}.")
        self.index += 1
        return token

    def accept(self, *texts: str) -> _Token | None:
        token = self.peek()
        if token is not None and token.kind == "op" and token.text in texts:
            self.index += 1
            return token
        return None

    def expect(self, text: str) -> None:
        if self.accept(text) is None:
            token = self.peek()
            found = token.text if token else "end of expression"
            raise ParseError(
                f"Expected {text!r} but found {found!r} in LEMS expression "
                f"{self.text!r}."
            )

    # -- grammar ---------------------------------------------------------- #
    def parse(self) -> Callable:
        node = self.parse_or()
        if self.peek() is not None:
            raise ParseError(
                f"Trailing input {self.peek().text!r} in LEMS expression "
                f"{self.text!r}."
            )
        return node

    def parse_or(self) -> Callable:
        left = self.parse_and()
        while self.accept("or"):
            right = self.parse_and()
            left = self._binary(left, right, lambda a, b: jnp.maximum(a, b))
        return left

    def parse_and(self) -> Callable:
        left = self.parse_comparison()
        while self.accept("and"):
            right = self.parse_comparison()
            left = self._binary(left, right, lambda a, b: jnp.minimum(a, b))
        return left

    def parse_comparison(self) -> Callable:
        left = self.parse_sum()
        token = self.accept(*_COMPARISONS)
        while token is not None:
            operation = _COMPARISONS[token.text]
            right = self.parse_sum()
            left = self._binary(
                left, right, lambda a, b, op=operation: op(a, b).astype(float)
            )
            token = self.accept(*_COMPARISONS)
        return left

    def parse_sum(self) -> Callable:
        left = self.parse_product()
        while True:
            token = self.accept("+", "-")
            if token is None:
                return left
            right = self.parse_product()
            if token.text == "+":
                left = self._binary(left, right, jnp.add)
            else:
                left = self._binary(left, right, jnp.subtract)

    def parse_product(self) -> Callable:
        left = self.parse_unary()
        while True:
            token = self.accept("*", "/")
            if token is None:
                return left
            right = self.parse_unary()
            if token.text == "*":
                left = self._binary(left, right, jnp.multiply)
            else:
                left = self._binary(left, right, jnp.divide)

    def parse_unary(self) -> Callable:
        token = self.accept("-", "+", "!")
        if token is None:
            return self.parse_power()
        operand = self.parse_unary()
        if token.text == "-":
            return lambda context: -operand(context)
        if token.text == "!":
            return lambda context: 1.0 - operand(context)
        return operand

    def parse_power(self) -> Callable:
        base = self.parse_atom()
        if self.accept("^"):
            exponent = self.parse_unary()  # right associative
            return self._binary(base, exponent, jnp.power)
        return base

    def parse_atom(self) -> Callable:
        token = self.take()
        if token.kind == "number":
            value = float(token.text)
            return lambda context, value=value: value
        if token.kind == "op" and token.text == "(":
            inner = self.parse_or()
            self.expect(")")
            return inner
        if token.kind == "name":
            name = token.text
            if self.accept("("):
                arguments = [self.parse_or()]
                while self.accept(","):
                    arguments.append(self.parse_or())
                self.expect(")")
                return self._call(name, arguments)
            self.symbols.add(name)
            return lambda context, name=name: _lookup(context, name)
        raise ParseError(
            f"Unexpected token {token.text!r} in LEMS expression {self.text!r}."
        )

    # -- helpers ---------------------------------------------------------- #
    def _call(self, name: str, arguments: list[Callable]) -> Callable:
        if name == "random":
            raise ParseError(
                "The LEMS function 'random' is not supported: tarjuman compiles "
                "expressions into deterministic JAX functions."
            )
        function = FUNCTIONS.get(name)
        if function is None:
            raise ParseError(
                f"Unknown function {name!r} in LEMS expression {self.text!r}. "
                f"Known functions: {', '.join(sorted(FUNCTIONS))}."
            )
        if len(arguments) != 1:
            raise ParseError(
                f"Function {name!r} takes one argument, got {len(arguments)}."
            )
        argument = arguments[0]
        return lambda context: function(argument(context))

    @staticmethod
    def _binary(left: Callable, right: Callable, operation: Callable) -> Callable:
        return lambda context: operation(left(context), right(context))


def _lookup(context: dict, name: str):
    try:
        return context[name]
    except KeyError:
        raise ParseError(
            f"LEMS expression refers to {name!r}, which is not defined here. "
            f"Available: {', '.join(sorted(str(key) for key in context))}."
        ) from None


def parse_expression(text: str) -> tuple[Callable, set[str]]:
    """Compile a LEMS expression, returning ``(function, symbols)``.

    Args:
        text: The infix expression.

    Returns:
        A callable taking a context dictionary, and the set of free symbols it
        reads from that context.
    """
    parser = _Parser(text)
    function = parser.parse()
    return function, parser.symbols


def compile_expression(text: str) -> Callable:
    """Compile a LEMS expression into a function of a context dictionary."""
    return parse_expression(text)[0]


def expression_symbols(text: str) -> set[str]:
    """The free symbols a LEMS expression reads."""
    return parse_expression(text)[1]
