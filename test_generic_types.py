"""
Demonstrates that Array[T] preserves its type parameter through generic functions.

Before the fix, type checkers resolved Array[Element] as bare Array, losing T.
After the fix, __class_getitem__ is hidden from type checkers so PEP 695 generic
alias machinery kicks in and T flows through correctly.

Run with: pyright test_generic_types.py
"""

from typing import TYPE_CHECKING, reveal_type

import typespark as ts
from typespark import functions as tsf


# --- 1. Annotation sites: T is preserved in variable annotations ---


def demo_variable_annotation(arr: ts.Array[ts.String]) -> ts.Array[ts.Int]:
    # arr should be Array[String], return should be Array[Int]
    # Both should carry their element type without manual annotation
    result: ts.Array[ts.Int] = arr.getItem(0)  # type: ignore[assignment]
    return result


# --- 2. Generic function: T flows from parameter to return ---


def identity[T: ts.TypedColumn](arr: ts.Array[T]) -> ts.Array[T]:
    return arr.__class__.wrap(arr.to_spark())  # type: ignore[return-value]


# --- 3. DataFrame field: Array[T] as field annotation resolves at class-def time ---


class MyFrame(ts.DataFrame):
    tags: ts.Array[ts.String]
    scores: ts.Array[ts.Int]


def make(x: ts.String):
    return tsf.array(x)


a = make(ts.String("hej"))


# --- 4. Nested arrays: Array[Array[T]] ---


class NestedFrame(ts.DataFrame):
    matrix: ts.Array[ts.Array[ts.Float]]


# --- 5. ArrayOf factory: same concrete class as subscript ---

StringArray = tsf.array  # callable reference, not instantiation
assert ts.Array[ts.String] is ts.ArrayOf(ts.String), (
    "ArrayOf and subscript must return same cached class"
)
assert ts.Array[ts.Int] is ts.ArrayOf(ts.Int)


# --- 6. getItem preserves element type ---


def demo_getitem(arr: ts.Array[ts.Long]) -> ts.Long:
    return arr.getItem(0)


# --- 7. reveal_type calls (Pyright/mypy will print inferred types) ---
# Expected: Array[String], Array[Int], type[Array[String]]

if TYPE_CHECKING:
    x: ts.Array[ts.String]
    reveal_type(x)  # should be: TypedArrayType[String]

    y: ts.Array[ts.Int]
    reveal_type(y)  # should be: TypedArrayType[Int]

    reveal_type(ts.Array[ts.String])  # should be: type[TypedArrayType[String]]

print("Runtime assertions passed.")
