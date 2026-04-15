"""Tests for primitive TypedColumn construction from Python literals and columns."""
import datetime
import decimal

import pytest
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    TimestampType,
)

from typespark.columns.primitives import (
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
# Helpers
# ---------------------------------------------------------------------------


def schema_type(spark, col):
    """Return the Spark DataType of a column expression in a single-row DataFrame."""
    return spark.createDataFrame([Row(x=1)]).select(col.to_spark().alias("v")).schema["v"].dataType


def collect_one(spark, col):
    """Evaluate a column expression against a single dummy row."""
    return spark.createDataFrame([Row(x=1)]).select(col.to_spark().alias("v")).collect()[0]["v"]


# ---------------------------------------------------------------------------
# String
# ---------------------------------------------------------------------------


def test_string_from_str_literal(spark):
    assert collect_one(spark, String("hello")) == "hello"


def test_string_from_str_casts_to_string_type(spark):
    assert isinstance(schema_type(spark, String("hello")), StringType)


def test_string_from_none(spark):
    assert collect_one(spark, String(None)) is None


def test_string_from_typed_column(spark):
    col = String.wrap(F.lit("world"))
    assert collect_one(spark, String(col)) == "world"


# ---------------------------------------------------------------------------
# Integer
# ---------------------------------------------------------------------------


def test_integer_from_int_literal(spark):
    assert collect_one(spark, Integer(42)) == 42


def test_integer_from_int_casts_to_integer_type(spark):
    assert isinstance(schema_type(spark, Integer(42)), IntegerType)


def test_integer_from_none(spark):
    assert collect_one(spark, Integer(None)) is None


def test_integer_from_typed_column(spark):
    col = Integer.wrap(F.lit(7).cast(IntegerType()))
    assert collect_one(spark, Integer(col)) == 7


# ---------------------------------------------------------------------------
# Long
# ---------------------------------------------------------------------------


def test_long_from_int_literal(spark):
    assert collect_one(spark, Long(99)) == 99


def test_long_from_int_casts_to_long_type(spark):
    assert isinstance(schema_type(spark, Long(99)), LongType)


# ---------------------------------------------------------------------------
# Float
# ---------------------------------------------------------------------------


def test_float_from_float_literal(spark):
    assert abs(collect_one(spark, Float(1.5)) - 1.5) < 1e-5


def test_float_from_int_literal(spark):
    assert abs(collect_one(spark, Float(2)) - 2.0) < 1e-5


def test_float_casts_to_float_type(spark):
    assert isinstance(schema_type(spark, Float(1.5)), FloatType)


# ---------------------------------------------------------------------------
# Double
# ---------------------------------------------------------------------------


def test_double_from_float_literal(spark):
    assert collect_one(spark, Double(3.14)) == pytest.approx(3.14)


def test_double_from_int_literal(spark):
    assert collect_one(spark, Double(5)) == pytest.approx(5.0)


def test_double_casts_to_double_type(spark):
    assert isinstance(schema_type(spark, Double(3.14)), DoubleType)


# ---------------------------------------------------------------------------
# Date
# ---------------------------------------------------------------------------


def test_date_from_date_literal(spark):
    d = datetime.date(2024, 3, 15)
    assert collect_one(spark, Date(d)) == d


def test_date_casts_to_date_type(spark):
    assert isinstance(schema_type(spark, Date(datetime.date(2024, 1, 1))), DateType)


def test_date_from_none(spark):
    assert collect_one(spark, Date(None)) is None


# ---------------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------------


def test_timestamp_from_datetime_literal(spark):
    ts = datetime.datetime(2024, 6, 15, 12, 0, 0)
    result = collect_one(spark, Timestamp(ts))
    assert result.year == 2024 and result.month == 6 and result.day == 15


def test_timestamp_casts_to_timestamp_type(spark):
    ts = datetime.datetime(2024, 1, 1)
    assert isinstance(schema_type(spark, Timestamp(ts)), TimestampType)


# ---------------------------------------------------------------------------
# Decimal — literal modes
# ---------------------------------------------------------------------------


def test_decimal_from_int_literal_default_precision(spark):
    result = collect_one(spark, Decimal(42))
    assert result == decimal.Decimal("42")


def test_decimal_from_int_literal_default_schema(spark):
    assert schema_type(spark, Decimal(42)) == DecimalType(10, 0)


def test_decimal_from_int_literal_explicit_precision(spark):
    assert schema_type(spark, Decimal(42, 12, 4)) == DecimalType(12, 4)


def test_decimal_from_float_literal(spark):
    assert schema_type(spark, Decimal(3.14, 10, 2)) == DecimalType(10, 2)


def test_decimal_from_decimal_literal(spark):
    assert schema_type(spark, Decimal(decimal.Decimal("1.5"), 10, 2)) == DecimalType(10, 2)


def test_decimal_from_none(spark):
    assert collect_one(spark, Decimal(None)) is None


# ---------------------------------------------------------------------------
# Decimal — wrap mode (no cast)
# ---------------------------------------------------------------------------


def test_decimal_wrap_preserves_type(spark):
    raw = F.lit(decimal.Decimal("1.23")).cast(DecimalType(8, 3))
    existing = Decimal.wrap(raw)
    wrapped = Decimal(existing)
    # schema should remain DecimalType(8, 3) — no re-cast
    assert schema_type(spark, wrapped) == DecimalType(8, 3)


# ---------------------------------------------------------------------------
# Decimal — cast mode
# ---------------------------------------------------------------------------


def test_decimal_cast_mode_from_integer_column(spark):
    int_col = Integer.wrap(F.lit(5).cast(IntegerType()))
    result = Decimal(int_col, 14, 6)
    assert schema_type(spark, result) == DecimalType(14, 6)


def test_decimal_cast_mode_from_raw_column(spark):
    raw = F.lit(3)
    result = Decimal(raw, 10, 2)
    assert schema_type(spark, result) == DecimalType(10, 2)


def test_decimal_cast_mode_default_scale(spark):
    int_col = Integer.wrap(F.lit(5).cast(IntegerType()))
    result = Decimal(int_col, precision=12)
    assert schema_type(spark, result) == DecimalType(12, 0)
