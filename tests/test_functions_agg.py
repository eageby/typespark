"""Return-type and value tests for typespark.functions.aggregates."""
import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import ArrayType, DoubleType, LongType

import typespark as ts
from typespark import BaseDataFrame
from typespark.columns.array import TypedArrayType
from typespark.columns.primitives import Double, Integer, Long, String
from typespark.columns.columns import Bool
from typespark.functions.aggregates import (
    approx_count_distinct,
    avg,
    collect_list,
    collect_set,
    corr,
    count,
    count_distinct,
    count_if,
    countDistinct,
    covar_pop,
    covar_samp,
    first,
    last,
    mean,
    min_by,
    stddev,
    stddev_pop,
    stddev_samp,
    sum,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class NumFrame(BaseDataFrame):
    name: ts.String
    age: ts.Int
    score: ts.Double


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.Builder().master("local").appName("pytest-pyspark-agg").getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(scope="function")
def num_frame(spark):
    data = [
        Row(name="Alice", age=30, score=1.5),
        Row(name="Alice", age=28, score=2.5),
        Row(name="Bob", age=25, score=3.0),
    ]
    yield NumFrame.from_df(spark.createDataFrame(data))


# ---------------------------------------------------------------------------
# Always-Long return type
# ---------------------------------------------------------------------------


def test_count_returns_long(num_frame: NumFrame) -> None:
    result = count(num_frame.age)
    assert isinstance(result, Long)
    assert not isinstance(result, Integer)


def test_count_value(num_frame: NumFrame) -> None:
    class Agg(ts.DataFrame):
        name: ts.String
        n: ts.Long

    tf = Agg(num_frame.to_df(), name=num_frame.name.group(), n=count(num_frame.age))
    rows = {r["name"]: r["n"] for r in tf.to_df().collect()}
    assert rows["Alice"] == 2
    assert rows["Bob"] == 1


def test_count_spark_type(num_frame: NumFrame) -> None:
    col = count(num_frame.age)
    actual = type(num_frame.to_df().select(col.to_spark().alias("v")).schema["v"].dataType)
    assert actual is LongType


def test_count_distinct_returns_long(num_frame: NumFrame) -> None:
    result = countDistinct(num_frame.age)
    assert isinstance(result, Long)


def test_count_distinct_value(num_frame: NumFrame) -> None:
    col = countDistinct(num_frame.age)
    row = num_frame.to_df().agg(col.to_spark().alias("v")).collect()[0]
    assert row["v"] == 3


def test_approx_count_distinct_returns_long(num_frame: NumFrame) -> None:
    result = approx_count_distinct(num_frame.name)
    assert isinstance(result, Long)


def test_count_distinct_returns_long(num_frame: NumFrame) -> None:
    result = count_distinct(num_frame.age)
    assert isinstance(result, Long)


def test_count_distinct_value(num_frame: NumFrame) -> None:
    col = count_distinct(num_frame.age)
    row = num_frame.to_df().agg(col.to_spark().alias("v")).collect()[0]
    assert row["v"] == 3


def test_count_if_returns_long(num_frame: NumFrame) -> None:
    result = count_if(num_frame.age > 26)
    assert isinstance(result, Long)


def test_count_if_value(num_frame: NumFrame) -> None:
    col = count_if(num_frame.age > 26)
    row = num_frame.to_df().agg(col.to_spark().alias("v")).collect()[0]
    assert row["v"] == 2


# ---------------------------------------------------------------------------
# sum overloads
# ---------------------------------------------------------------------------


def test_sum_integer_returns_long(num_frame: NumFrame) -> None:
    result = sum(num_frame.age)
    assert isinstance(result, Long)
    assert not isinstance(result, Integer)


def test_sum_long_returns_long(num_frame: NumFrame) -> None:
    class LFrame(BaseDataFrame):
        v: ts.Long

    spark = num_frame.to_df().sparkSession
    lf = LFrame.from_df(spark.createDataFrame([Row(v=1)]))
    result = sum(lf.v)
    assert isinstance(result, Long)


def test_sum_double_preserves_type(num_frame: NumFrame) -> None:
    result = sum(num_frame.score)
    assert isinstance(result, Double)
    assert not isinstance(result, Long)


def test_sum_integer_spark_type(num_frame: NumFrame) -> None:
    col = sum(num_frame.age)
    actual = type(num_frame.to_df().select(col.to_spark().alias("v")).schema["v"].dataType)
    assert actual is LongType


# ---------------------------------------------------------------------------
# Always-Double
# ---------------------------------------------------------------------------


def test_avg_returns_double(num_frame: NumFrame) -> None:
    result = avg(num_frame.age)
    assert isinstance(result, Double)


def test_avg_spark_type(num_frame: NumFrame) -> None:
    col = avg(num_frame.age)
    actual = type(num_frame.to_df().select(col.to_spark().alias("v")).schema["v"].dataType)
    assert actual is DoubleType


def test_mean_is_avg(num_frame: NumFrame) -> None:
    assert mean is avg


def test_stddev_returns_double(num_frame: NumFrame) -> None:
    assert isinstance(stddev(num_frame.age), Double)


def test_stddev_samp_returns_double(num_frame: NumFrame) -> None:
    assert isinstance(stddev_samp(num_frame.age), Double)


def test_stddev_pop_returns_double(num_frame: NumFrame) -> None:
    assert isinstance(stddev_pop(num_frame.age), Double)


def test_corr_returns_double(num_frame: NumFrame) -> None:
    result = corr(num_frame.age, num_frame.score)
    assert isinstance(result, Double)


def test_corr_self_is_one(num_frame: NumFrame) -> None:
    col = corr(num_frame.age, num_frame.age)
    row = num_frame.to_df().agg(col.to_spark().alias("v")).collect()[0]
    assert abs(row["v"] - 1.0) < 1e-9


def test_covar_pop_returns_double(num_frame: NumFrame) -> None:
    assert isinstance(covar_pop(num_frame.age, num_frame.score), Double)


def test_covar_samp_returns_double(num_frame: NumFrame) -> None:
    assert isinstance(covar_samp(num_frame.age, num_frame.score), Double)


# ---------------------------------------------------------------------------
# Collection output
# ---------------------------------------------------------------------------


def test_collect_list_returns_array(num_frame: NumFrame) -> None:
    result = collect_list(num_frame.age)
    assert isinstance(result, TypedArrayType)


def test_collect_list_elem_type(num_frame: NumFrame) -> None:
    result = collect_list(num_frame.age)
    assert result._elem_type is Integer


def test_collect_list_spark_type(num_frame: NumFrame) -> None:
    col = collect_list(num_frame.age)
    field_type = num_frame.to_df().select(col.to_spark().alias("v")).schema["v"].dataType
    assert isinstance(field_type, ArrayType)


def test_collect_list_value(num_frame: NumFrame) -> None:
    class Agg(ts.DataFrame):
        name: ts.String
        ages: ts.Array

    tf = Agg(
        num_frame.to_df(),
        name=num_frame.name.group(),
        ages=collect_list(num_frame.age),
    )
    rows = {r["name"]: sorted(r["ages"]) for r in tf.to_df().collect()}
    assert rows["Alice"] == [28, 30]
    assert rows["Bob"] == [25]


def test_collect_set_returns_array(num_frame: NumFrame) -> None:
    result = collect_set(num_frame.age)
    assert isinstance(result, TypedArrayType)


def test_collect_set_elem_type(num_frame: NumFrame) -> None:
    result = collect_set(num_frame.age)
    assert result._elem_type is Integer


# ---------------------------------------------------------------------------
# Type-preserving: first, last, min_by
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def null_frame(spark):
    """Single partition so first/last are deterministic; nulls at both ends."""
    data = [
        Row(name=None, age=30, score=1.5),
        Row(name="Alice", age=28, score=2.5),
        Row(name="Bob", age=25, score=3.0),
        Row(name=None, age=40, score=4.0),
    ]
    df = spark.createDataFrame(data, NumFrame.generate_schema()).coalesce(1)
    yield NumFrame.from_df(df)


def test_first_returns_same_type(num_frame: NumFrame) -> None:
    result = first(num_frame.name)
    assert isinstance(result, String)


def test_first_keeps_nulls_by_default(null_frame: NumFrame) -> None:
    col = first(null_frame.name)
    row = null_frame.to_df().agg(col.to_spark().alias("v")).collect()[0]
    assert row["v"] is None


def test_first_ignorenulls(null_frame: NumFrame) -> None:
    col = first(null_frame.name, ignorenulls=True)
    row = null_frame.to_df().agg(col.to_spark().alias("v")).collect()[0]
    assert row["v"] == "Alice"


def test_last_keeps_nulls_by_default(null_frame: NumFrame) -> None:
    col = last(null_frame.name)
    row = null_frame.to_df().agg(col.to_spark().alias("v")).collect()[0]
    assert row["v"] is None


def test_last_ignorenulls(null_frame: NumFrame) -> None:
    col = last(null_frame.name, ignorenulls=True)
    row = null_frame.to_df().agg(col.to_spark().alias("v")).collect()[0]
    assert row["v"] == "Bob"


def test_last_returns_same_type(num_frame: NumFrame) -> None:
    result = last(num_frame.name)
    assert isinstance(result, String)


def test_min_by_returns_same_type(num_frame: NumFrame) -> None:
    result = min_by(num_frame.name, num_frame.age)
    assert isinstance(result, String)


def test_min_by_value(num_frame: NumFrame) -> None:
    col = min_by(num_frame.name, num_frame.age)
    row = num_frame.to_df().agg(col.to_spark().alias("v")).collect()[0]
    assert row["v"] == "Bob"
