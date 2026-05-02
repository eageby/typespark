from typing import TypeVar

from pyspark.sql import functions as F

from typespark.columns import TypedColumn
from typespark.columns.groups import _make_aggregate
from typespark.type_alias import Integer

C = TypeVar("C", bound=TypedColumn)


def sum(col: C) -> C:
    return _make_aggregate(type(col), F.sum(col.to_spark()))  # type: ignore[return-value]


def first(col: C) -> C:
    return _make_aggregate(type(col), F.first(col.to_spark()))  # type: ignore[return-value]


def max(col: C) -> C:
    return _make_aggregate(type(col), F.max(col.to_spark()))  # type: ignore[return-value]


def max_by(col: C, ord: TypedColumn) -> C:
    return _make_aggregate(type(col), F.max_by(col.to_spark(), ord.to_spark()))  # type: ignore[return-value]


def min(col: C) -> C:
    return _make_aggregate(type(col), F.min(col.to_spark()))  # type: ignore[return-value]


def count(col: TypedColumn) -> Integer:
    return _make_aggregate(Integer, F.count(col.to_spark()))  # type: ignore[return-value]
