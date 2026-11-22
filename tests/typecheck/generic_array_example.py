# pyright: reportUnnecessaryTypeIgnoreComment=true
# mypy: warn-unused-ignores

"""Array[T] keeps its element type through annotations and generic functions,
checked with pyright by tests/test_typecheck.py.

Lines marked `# type: ignore` must produce an error.
"""

from typing import assert_type

import typespark as ts
from typespark import functions as tsf


# Annotation sites keep the element type.
def element_type(arr: ts.Array[ts.String], ints: ts.Array[ts.Int]) -> None:
    assert_type(arr, ts.Array[ts.String])
    assert_type(ints, ts.Array[ts.Int])
    assert_type(arr.getItem(0), ts.String)


# getItem returns the element type, so a mismatched element is an error.
def wrong_element(arr: ts.Array[ts.String]) -> ts.Int:
    return arr.getItem(0)  # type: ignore


# T flows from parameter to return in a generic function.
def identity[T: ts.Column](arr: ts.Array[T]) -> ts.Array[T]:
    return arr


def generic_call(arr: ts.Array[ts.Long]) -> None:
    assert_type(identity(arr), ts.Array[ts.Long])


# array() infers the element type from its arguments.
def make(x: ts.String) -> None:
    assert_type(tsf.array(x), ts.Array[ts.String])


# Array fields, nested arrays included, resolve at class-definition time.
class Frame(ts.DataFrame):
    tags: ts.Array[ts.String]
    matrix: ts.Array[ts.Array[ts.Float]]


def fields(df: Frame) -> None:
    assert_type(df.tags, ts.Array[ts.String])
    assert_type(df.matrix.getItem(0), ts.Array[ts.Float])
