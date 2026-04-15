"""Verify that operator return types match actual Spark schema types.

Uses check_op_types() which asserts both:
  - the TypedColumn wrapper class (via _data_type) matches the Spark schema type
  - catching annotation lies AND Spark version changes in one call

If Spark changes type promotion rules these tests will fail.
"""

import datetime
import decimal

import pytest
from pyspark.sql import functions as F

from typespark import BaseDataFrame
from typespark.columns.columns import Bool, TypedColumn
from typespark.columns.primitives import (
    Binary,
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
# Helper
# ---------------------------------------------------------------------------


def check_op_types(df, col: TypedColumn) -> None:
    """Assert col's Python class _data_type matches the actual Spark result schema type.

    Fails if:
    - the operator returns the wrong wrapper class (annotation lie)
    - Spark changes its type promotion rules
    """
    expected = type(col)._data_type
    assert expected is not None, (
        f"{type(col).__name__} has no _data_type — "
        "use TypedColumn for intentionally untyped ops"
    )
    actual = type(df.select(col.alias("v")).schema["v"].dataType)
    assert actual is expected, (
        f"Spark returned {actual.__name__} but "
        f"{type(col).__name__}._data_type = {expected.__name__}"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class IntFrame(BaseDataFrame):
    a: Integer
    b: Integer


class LongFrame(BaseDataFrame):
    a: Long
    b: Long


class FloatFrame(BaseDataFrame):
    a: Float
    b: Float


class DoubleFrame(BaseDataFrame):
    a: Double
    b: Double


class DecimalFrame(BaseDataFrame):
    a: Decimal
    b: Decimal


class BoolFrame(BaseDataFrame):
    a: Bool
    b: Bool


class StringFrame(BaseDataFrame):
    a: String
    b: String


class DateFrame(BaseDataFrame):
    a: Date
    b: Date


class TimestampFrame(BaseDataFrame):
    a: Timestamp
    b: Timestamp


class BinaryFrame(BaseDataFrame):
    a: Binary
    b: Binary


@pytest.fixture
def ints(spark):
    return IntFrame.from_df(spark.createDataFrame([(2, 3)], schema="a INT, b INT"))


@pytest.fixture
def longs(spark):
    return LongFrame.from_df(spark.createDataFrame([(2, 3)], schema="a LONG, b LONG"))


@pytest.fixture
def floats(spark):
    return FloatFrame.from_df(spark.createDataFrame([(1.0, 2.0)], schema="a FLOAT, b FLOAT"))


@pytest.fixture
def doubles(spark):
    return DoubleFrame.from_df(spark.createDataFrame([(1.0, 2.0)], schema="a DOUBLE, b DOUBLE"))


@pytest.fixture
def decimals(spark):
    df = spark.range(1).select(
        F.lit(decimal.Decimal("1.50")).cast("DECIMAL(10,2)").alias("a"),
        F.lit(decimal.Decimal("2.50")).cast("DECIMAL(10,2)").alias("b"),
    )
    return DecimalFrame.from_df(df)


@pytest.fixture
def bools(spark):
    return BoolFrame.from_df(
        spark.createDataFrame([(True, False), (False, True)], schema="a BOOLEAN, b BOOLEAN")
    )


@pytest.fixture
def strings(spark):
    return StringFrame.from_df(
        spark.createDataFrame([("foo", "bar")], schema="a STRING, b STRING")
    )


@pytest.fixture
def dates(spark):
    return DateFrame.from_df(
        spark.createDataFrame(
            [(datetime.date(2024, 1, 1), datetime.date(2024, 6, 1))],
            schema="a DATE, b DATE",
        )
    )


@pytest.fixture
def timestamps(spark):
    return TimestampFrame.from_df(
        spark.createDataFrame(
            [(datetime.datetime(2024, 1, 1), datetime.datetime(2024, 6, 1))],
            schema="a TIMESTAMP, b TIMESTAMP",
        )
    )


@pytest.fixture
def binaries(spark):
    return BinaryFrame.from_df(
        spark.createDataFrame([(bytearray(b"x"), bytearray(b"y"))], schema="a BINARY, b BINARY")
    )


# ---------------------------------------------------------------------------
# Integer arithmetic
# ---------------------------------------------------------------------------


def test_integer_add(ints):
    check_op_types(ints, ints.a + ints.b)


def test_integer_sub(ints):
    check_op_types(ints, ints.a - ints.b)


def test_integer_mul(ints):
    check_op_types(ints, ints.a * ints.b)


def test_integer_mod(ints):
    check_op_types(ints, ints.a % ints.b)


def test_integer_neg(ints):
    check_op_types(ints, -ints.a)


def test_integer_truediv(ints):
    check_op_types(ints, ints.a / ints.b)


def test_integer_pow(ints):
    check_op_types(ints, ints.a ** ints.b)


# ---------------------------------------------------------------------------
# Long arithmetic
# ---------------------------------------------------------------------------


def test_long_add(longs):
    check_op_types(longs, longs.a + longs.b)


def test_long_sub(longs):
    check_op_types(longs, longs.a - longs.b)


def test_long_mul(longs):
    check_op_types(longs, longs.a * longs.b)


def test_long_mod(longs):
    check_op_types(longs, longs.a % longs.b)


def test_long_neg(longs):
    check_op_types(longs, -longs.a)


def test_long_truediv(longs):
    check_op_types(longs, longs.a / longs.b)


def test_long_pow(longs):
    check_op_types(longs, longs.a ** longs.b)


# ---------------------------------------------------------------------------
# Float arithmetic
# ---------------------------------------------------------------------------


def test_float_add(floats):
    check_op_types(floats, floats.a + floats.b)


def test_float_sub(floats):
    check_op_types(floats, floats.a - floats.b)


def test_float_mul(floats):
    check_op_types(floats, floats.a * floats.b)


def test_float_mod(floats):
    check_op_types(floats, floats.a % floats.b)


def test_float_neg(floats):
    check_op_types(floats, -floats.a)


def test_float_truediv(floats):
    check_op_types(floats, floats.a / floats.b)


def test_float_pow(floats):
    check_op_types(floats, floats.a ** floats.b)


# ---------------------------------------------------------------------------
# Double arithmetic
# ---------------------------------------------------------------------------


def test_double_add(doubles):
    check_op_types(doubles, doubles.a + doubles.b)


def test_double_sub(doubles):
    check_op_types(doubles, doubles.a - doubles.b)


def test_double_mul(doubles):
    check_op_types(doubles, doubles.a * doubles.b)


def test_double_mod(doubles):
    check_op_types(doubles, doubles.a % doubles.b)


def test_double_neg(doubles):
    check_op_types(doubles, -doubles.a)


def test_double_truediv(doubles):
    check_op_types(doubles, doubles.a / doubles.b)


def test_double_pow(doubles):
    check_op_types(doubles, doubles.a ** doubles.b)


# ---------------------------------------------------------------------------
# Decimal arithmetic
# ---------------------------------------------------------------------------


def test_decimal_add(decimals):
    check_op_types(decimals, decimals.a + decimals.b)


def test_decimal_sub(decimals):
    check_op_types(decimals, decimals.a - decimals.b)


def test_decimal_mul(decimals):
    check_op_types(decimals, decimals.a * decimals.b)


def test_decimal_truediv(decimals):
    check_op_types(decimals, decimals.a / decimals.b)


def test_decimal_mod(decimals):
    check_op_types(decimals, decimals.a % decimals.b)


def test_decimal_neg(decimals):
    check_op_types(decimals, -decimals.a)


def test_decimal_pow(decimals):
    check_op_types(decimals, decimals.a ** decimals.b)


# ---------------------------------------------------------------------------
# Bool logical ops
# ---------------------------------------------------------------------------


def test_bool_and(bools):
    check_op_types(bools, bools.a & bools.b)


def test_bool_or(bools):
    check_op_types(bools, bools.a | bools.b)


def test_bool_invert(bools):
    check_op_types(bools, ~bools.a)


def test_bool_eq(bools):
    check_op_types(bools, bools.a == bools.b)


def test_bool_ne(bools):
    check_op_types(bools, bools.a != bools.b)


# ---------------------------------------------------------------------------
# Comparison ops — all typed primitives must return Bool
# ---------------------------------------------------------------------------


def test_integer_eq_returns_bool(ints):
    check_op_types(ints, ints.a == ints.b)


def test_integer_lt_returns_bool(ints):
    check_op_types(ints, ints.a < ints.b)


def test_long_eq_returns_bool(longs):
    check_op_types(longs, longs.a == longs.b)


def test_long_lt_returns_bool(longs):
    check_op_types(longs, longs.a < longs.b)


def test_float_eq_returns_bool(floats):
    check_op_types(floats, floats.a == floats.b)


def test_double_eq_returns_bool(doubles):
    check_op_types(doubles, doubles.a == doubles.b)


def test_decimal_eq_returns_bool(decimals):
    check_op_types(decimals, decimals.a == decimals.b)


def test_string_eq_returns_bool(strings):
    check_op_types(strings, strings.a == strings.b)


def test_string_lt_returns_bool(strings):
    check_op_types(strings, strings.a < strings.b)


def test_date_eq_returns_bool(dates):
    check_op_types(dates, dates.a == dates.b)


def test_date_lt_returns_bool(dates):
    check_op_types(dates, dates.a < dates.b)


def test_timestamp_eq_returns_bool(timestamps):
    check_op_types(timestamps, timestamps.a == timestamps.b)


def test_timestamp_lt_returns_bool(timestamps):
    check_op_types(timestamps, timestamps.a < timestamps.b)


def test_binary_eq_returns_bool(binaries):
    check_op_types(binaries, binaries.a == binaries.b)
