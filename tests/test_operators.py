"""Tests for typed operator overloads on TypedColumn primitives."""

import datetime
import decimal

import pytest
from pyspark.sql import Row

from tests.utils import collect_column
from typespark import BaseDataFrame
from typespark.columns.primitives import (
    Binary,
    Bool,
    Date,
    Decimal,
    Double,
    Float,
    Integer,
    Long,
    String,
    Timestamp,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class NumFrame(BaseDataFrame):
    a: Integer
    b: Integer


class FloatFrame(BaseDataFrame):
    x: Double
    y: Double


class StrFrame(BaseDataFrame):
    s: String


class BoolFrame(BaseDataFrame):
    flag: Bool


class DateFrame(BaseDataFrame):
    d: Date


class TsFrame(BaseDataFrame):
    ts: Timestamp


@pytest.fixture
def nums(spark):
    return NumFrame.from_df(spark.createDataFrame([Row(a=1, b=2), Row(a=3, b=4)]))


@pytest.fixture
def floats(spark):
    return FloatFrame.from_df(
        spark.createDataFrame(
            [Row(x=1.5, y=2.5), Row(x=3.0, y=3.0)],
            schema="x DOUBLE, y DOUBLE",
        )
    )


@pytest.fixture
def strings(spark):
    return StrFrame.from_df(spark.createDataFrame([Row(s="apple"), Row(s="banana")]))


@pytest.fixture
def bools(spark):
    return BoolFrame.from_df(spark.createDataFrame([Row(flag=True), Row(flag=False)]))


@pytest.fixture
def dates(spark):
    return DateFrame.from_df(
        spark.createDataFrame(
            [Row(d=datetime.date(2024, 1, 1)), Row(d=datetime.date(2024, 6, 1))],
            schema="d DATE",
        )
    )


@pytest.fixture
def timestamps(spark):
    return TsFrame.from_df(
        spark.createDataFrame(
            [
                Row(ts=datetime.datetime(2024, 1, 1, 0, 0, 0)),
                Row(ts=datetime.datetime(2024, 6, 1, 12, 0, 0)),
            ],
            schema="ts TIMESTAMP",
        )
    )


# ---------------------------------------------------------------------------
# Integer — comparison with literal
# ---------------------------------------------------------------------------


def test_integer_eq_literal(nums: NumFrame):
    result = collect_column(nums.filter(nums.a == 1), "a")
    assert result == [1]


def test_integer_ne_literal(nums: NumFrame):
    result = collect_column(nums.filter(nums.a != 1), "a")
    assert result == [3]


def test_integer_lt_literal(nums: NumFrame):
    result = collect_column(nums.filter(nums.a < 3), "a")
    assert result == [1]


def test_integer_le_literal(nums: NumFrame):
    result = collect_column(nums.filter(nums.a <= 1), "a")
    assert result == [1]


def test_integer_gt_literal(nums: NumFrame):
    result = collect_column(nums.filter(nums.a > 1), "a")
    assert result == [3]


def test_integer_ge_literal(nums: NumFrame):
    result = collect_column(nums.filter(nums.a >= 3), "a")
    assert result == [3]


# ---------------------------------------------------------------------------
# Integer — comparison column-to-column
# ---------------------------------------------------------------------------


def test_integer_eq_column(nums: NumFrame):
    result = collect_column(nums.filter(nums.a == nums.b), "a")
    assert result == []


def test_integer_lt_column(nums: NumFrame):
    result = collect_column(nums.filter(nums.a < nums.b), "a")
    assert result == [1, 3]


# ---------------------------------------------------------------------------
# Integer — arithmetic with literal
# ---------------------------------------------------------------------------


def test_integer_add_literal(nums: NumFrame):
    result = collect_column(nums.select((nums.a + 10).alias("v")), "v")
    assert result == [11, 13]


def test_integer_sub_literal(nums: NumFrame):
    result = collect_column(nums.select((nums.a - 1).alias("v")), "v")
    assert result == [0, 2]


def test_integer_mul_literal(nums: NumFrame):
    result = collect_column(nums.select((nums.a * 3).alias("v")), "v")
    assert result == [3, 9]


def test_integer_truediv_literal(nums: NumFrame):
    result = collect_column(nums.select((nums.b / 2).alias("v")), "v")
    assert result == [1.0, 2.0]


def test_integer_mod_literal(nums: NumFrame):
    result = collect_column(nums.select((nums.a % 2).alias("v")), "v")
    assert result == [1, 1]


def test_integer_pow_literal(nums: NumFrame):
    result = collect_column(nums.select((nums.a**2).alias("v")), "v")
    assert result == [1, 9]


def test_integer_neg(nums: NumFrame):
    result = collect_column(nums.select((-nums.a).alias("v")), "v")
    assert result == [-1, -3]


# ---------------------------------------------------------------------------
# Double — comparison and arithmetic with float literal
# ---------------------------------------------------------------------------


def test_double_eq_float_literal(floats: FloatFrame):
    result = collect_column(floats.filter(floats.x == 1.5), "x")
    assert result == [1.5]


def test_double_gt_int_literal(floats: FloatFrame):
    result = collect_column(floats.filter(floats.x > 2), "x")
    assert result == [3.0]


def test_double_add_float_literal(floats: FloatFrame):
    result = collect_column(floats.select((floats.x + 0.5).alias("v")), "v")
    assert result == [2.0, 3.5]


def test_double_eq_column(floats: FloatFrame):
    result = collect_column(floats.filter(floats.x == floats.y), "x")
    assert result == [3.0]


# ---------------------------------------------------------------------------
# String — comparison with literal
# ---------------------------------------------------------------------------


def test_string_eq_literal(strings: StrFrame):
    result = collect_column(strings.filter(strings.s == "apple"), "s")
    assert result == ["apple"]


def test_string_ne_literal(strings: StrFrame):
    result = collect_column(strings.filter(strings.s != "apple"), "s")
    assert result == ["banana"]


def test_string_lt_literal(strings: StrFrame):
    # "apple" < "banana" alphabetically
    result = collect_column(strings.filter(strings.s < "banana"), "s")
    assert result == ["apple"]


def test_string_gt_literal(strings: StrFrame):
    result = collect_column(strings.filter(strings.s > "apple"), "s")
    assert result == ["banana"]


def test_string_eq_column(strings: StrFrame):
    result = collect_column(strings.filter(strings.s == strings.s), "s")
    assert sorted(result) == ["apple", "banana"]


# ---------------------------------------------------------------------------
# Bool — logical operators
# ---------------------------------------------------------------------------


def test_bool_and_literal(bools: BoolFrame):
    result = collect_column(bools.filter(bools.flag & True), "flag")
    assert result == [True]


def test_bool_or_literal(bools: BoolFrame):
    result = collect_column(bools.filter(bools.flag | True), "flag")
    assert sorted(result) == [False, True]


def test_bool_invert(bools: BoolFrame):
    result = collect_column(bools.select((~bools.flag).alias("v")), "v")
    assert sorted(result) == [False, True]


def test_bool_eq_literal(bools: BoolFrame):
    result = collect_column(bools.filter(bools.flag == True), "flag")  # noqa: E712
    assert result == [True]


def test_bool_and_column(bools: BoolFrame):
    result = collect_column(bools.filter(bools.flag & bools.flag), "flag")
    assert result == [True]


# ---------------------------------------------------------------------------
# Date — comparison with datetime.date literal
# ---------------------------------------------------------------------------


def test_date_eq_literal(dates: DateFrame):
    result = collect_column(dates.filter(dates.d == datetime.date(2024, 1, 1)), "d")
    assert result == [datetime.date(2024, 1, 1)]


def test_date_lt_literal(dates: DateFrame):
    result = collect_column(dates.filter(dates.d < datetime.date(2024, 6, 1)), "d")
    assert result == [datetime.date(2024, 1, 1)]


def test_date_gt_literal(dates: DateFrame):
    result = collect_column(dates.filter(dates.d > datetime.date(2024, 1, 1)), "d")
    assert result == [datetime.date(2024, 6, 1)]


def test_date_eq_column(dates: DateFrame):
    result = collect_column(dates.filter(dates.d == dates.d), "d")
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Timestamp — comparison with datetime.datetime literal
# ---------------------------------------------------------------------------


def test_timestamp_eq_literal(timestamps: TsFrame):
    result = collect_column(
        timestamps.filter(timestamps.ts == datetime.datetime(2024, 1, 1, 0, 0, 0)),
        "ts",
    )
    assert len(result) == 1
    assert result[0].year == 2024 and result[0].month == 1


def test_timestamp_lt_literal(timestamps: TsFrame):
    result = collect_column(
        timestamps.filter(timestamps.ts < datetime.datetime(2024, 6, 1, 0, 0, 0)),
        "ts",
    )
    assert len(result) == 1


def test_timestamp_gt_literal(timestamps: TsFrame):
    result = collect_column(
        timestamps.filter(timestamps.ts > datetime.datetime(2024, 1, 1, 12, 0, 0)),
        "ts",
    )
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Negation preserves the primitive type
# ---------------------------------------------------------------------------


def test_neg_returns_same_class(nums: NumFrame):
    result = -nums.a
    assert isinstance(result, Integer)


def test_neg_double_returns_same_class(floats: FloatFrame):
    result = -floats.x
    assert isinstance(result, Double)


# ---------------------------------------------------------------------------
# between — inclusive range, literal and column bounds
# ---------------------------------------------------------------------------


def test_between_returns_bool(nums: NumFrame):
    assert isinstance(nums.a.between(1, 2), Bool)


def test_integer_between_literals(nums: NumFrame):
    result = collect_column(nums.filter(nums.a.between(1, 2)), "a")
    assert result == [1]


def test_between_is_inclusive(nums: NumFrame):
    result = collect_column(nums.filter(nums.a.between(1, 3)), "a")
    assert result == [1, 3]


def test_integer_between_columns(nums: NumFrame):
    result = collect_column(nums.filter(nums.a.between(nums.a, nums.b)), "a")
    assert result == [1, 3]


def test_double_between_literals(floats: FloatFrame):
    result = collect_column(floats.filter(floats.x.between(1.0, 2.0)), "x")
    assert result == [1.5]


def test_string_between_literals(strings: StrFrame):
    result = collect_column(strings.filter(strings.s.between("a", "b")), "s")
    assert result == ["apple"]


def test_date_between_literals(dates: DateFrame):
    result = collect_column(
        dates.filter(
            dates.d.between(datetime.date(2024, 1, 1), datetime.date(2024, 3, 1))
        ),
        "d",
    )
    assert result == [datetime.date(2024, 1, 1)]


def test_timestamp_between_literals(timestamps: TsFrame):
    result = collect_column(
        timestamps.filter(
            timestamps.ts.between(
                datetime.datetime(2024, 1, 1, 0, 0, 0),
                datetime.datetime(2024, 3, 1, 0, 0, 0),
            )
        ),
        "ts",
    )
    assert len(result) == 1
