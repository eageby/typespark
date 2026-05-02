"""Typed aggregate functions for use with the typed ``DataFrame`` constructor.

Pass aggregate columns alongside ``.group()`` columns when constructing a typed
``DataFrame`` — see ``BaseDataFrame`` for usage.

Implemented
-----------
count, countDistinct, approx_count_distinct,
sum (with Integer/Long → Long overloads),
avg, mean, stddev, stddev_samp, stddev_pop,
corr, covar_pop, covar_samp,
collect_list, collect_set,
min, max, min_by, max_by, first, last

Not yet implemented (contributions welcome)
--------------------------------------------
variance / var_samp / var_pop  — TypedColumn → Double
kurtosis                       — TypedColumn → Double
skewness                       — TypedColumn → Double
median                         — TypedColumn → Double
mode                           — TypedColumn → C
percentile_approx              — (TypedColumn, percentage, accuracy) → Double
bit_and / bit_or / bit_xor     — C → C  (integer columns)
nth_value                      — (C, offset) → C
any_value                      — C → C
sumDistinct                    — same overloads as sum
"""
from typing import TypeVar, overload

from pyspark.sql import functions as F

from typespark.columns import TypedColumn
from typespark.columns.array import ArrayOf, TypedArrayType
from typespark.columns.groups import _make_aggregate
from typespark.type_alias import Double, Integer, Long

C = TypeVar("C", bound=TypedColumn)


# ---------------------------------------------------------------------------
# Type-preserving  (C → C)
# ---------------------------------------------------------------------------


def min(col: C) -> C:
    return _make_aggregate(type(col), F.min(col.to_spark()))  # type: ignore[return-value]


def max(col: C) -> C:
    return _make_aggregate(type(col), F.max(col.to_spark()))  # type: ignore[return-value]


def first(col: C) -> C:
    return _make_aggregate(type(col), F.first(col.to_spark()))  # type: ignore[return-value]


def last(col: C) -> C:
    return _make_aggregate(type(col), F.last(col.to_spark()))  # type: ignore[return-value]


def min_by(col: C, ord: TypedColumn) -> C:
    return _make_aggregate(type(col), F.min_by(col.to_spark(), ord.to_spark()))  # type: ignore[return-value]


def max_by(col: C, ord: TypedColumn) -> C:
    return _make_aggregate(type(col), F.max_by(col.to_spark(), ord.to_spark()))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Always Long
# ---------------------------------------------------------------------------


def count(col: TypedColumn) -> Long:
    return _make_aggregate(Long, F.count(col.to_spark()))  # type: ignore[return-value]


def countDistinct(*cols: TypedColumn) -> Long:
    return _make_aggregate(Long, F.countDistinct(*[c.to_spark() for c in cols]))  # type: ignore[return-value]


def approx_count_distinct(col: TypedColumn, rsd: float = 0.05) -> Long:
    return _make_aggregate(Long, F.approx_count_distinct(col.to_spark(), rsd))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# sum — Integer|Long → Long, everything else preserved
# ---------------------------------------------------------------------------


@overload
def sum(col: Integer) -> Long: ...


@overload
def sum(col: Long) -> Long: ...


@overload
def sum(col: C) -> C: ...


def sum(col: TypedColumn) -> TypedColumn:
    target = Long if isinstance(col, (Integer, Long)) else type(col)
    return _make_aggregate(target, F.sum(col.to_spark()))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Always Double
# ---------------------------------------------------------------------------


def avg(col: TypedColumn) -> Double:
    return _make_aggregate(Double, F.avg(col.to_spark()))  # type: ignore[return-value]


mean = avg


def stddev(col: TypedColumn) -> Double:
    return _make_aggregate(Double, F.stddev(col.to_spark()))  # type: ignore[return-value]


def stddev_samp(col: TypedColumn) -> Double:
    return _make_aggregate(Double, F.stddev_samp(col.to_spark()))  # type: ignore[return-value]


def stddev_pop(col: TypedColumn) -> Double:
    return _make_aggregate(Double, F.stddev_pop(col.to_spark()))  # type: ignore[return-value]


def corr(col1: TypedColumn, col2: TypedColumn) -> Double:
    return _make_aggregate(Double, F.corr(col1.to_spark(), col2.to_spark()))  # type: ignore[return-value]


def covar_pop(col1: TypedColumn, col2: TypedColumn) -> Double:
    return _make_aggregate(Double, F.covar_pop(col1.to_spark(), col2.to_spark()))  # type: ignore[return-value]


def covar_samp(col1: TypedColumn, col2: TypedColumn) -> Double:
    return _make_aggregate(Double, F.covar_samp(col1.to_spark(), col2.to_spark()))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Collection output  (C → TypedArrayType[C])
# ---------------------------------------------------------------------------


def collect_list(col: C) -> "TypedArrayType[C]":
    return _make_aggregate(ArrayOf(type(col)), F.collect_list(col.to_spark()))  # type: ignore[return-value]


def collect_set(col: C) -> "TypedArrayType[C]":
    return _make_aggregate(ArrayOf(type(col)), F.collect_set(col.to_spark()))  # type: ignore[return-value]
