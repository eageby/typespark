import json

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from tests.conftest import Id, Range
from tests.utils import collect_column, collect_values
from typespark import (
    Binary,
    DataFrame,
    Double,
    Int,
    String,
    string_literal,
)
from typespark import functions as tsf


def test_broadcast(id: Id, small_range: Range):
    sm = tsf.broadcast(small_range)

    df = id.join(sm, id.value == sm.id)

    result = collect_values(df)

    assert result[0]["value"] == 1
    assert result[1]["value"] == 2
    assert result[0]["id"] == 1
    assert result[1]["id"] == 2


def test_coalesce_prefers_left(id: Id, small_range: Range):
    df = id.join(small_range, id.value == small_range.id, "fullouter").select(
        tsf.coalesce(id.value, small_range.id).alias("coalesced"),
        id.value.alias("value"),
        small_range.id.alias("rid"),
    )

    rows = collect_values(df)

    assert rows[0]["coalesced"] == rows[0]["rid"] == 0
    assert rows[1]["coalesced"] == rows[1]["value"] == 1
    assert rows[2]["coalesced"] == rows[2]["value"] == 2
    assert rows[3]["coalesced"] == rows[3]["value"] == 3
    assert rows[4]["coalesced"] == rows[4]["value"] == 3
    assert rows[5]["coalesced"] == rows[5]["value"] == 4

    assert isinstance(df.to_spark().schema["coalesced"].dataType, IntegerType)


def test_when(spark: SparkSession):
    class IntTestData(DataFrame):
        a: Int
        b: Int

    data = [
        (1, 2),
        (1, 3),
        (1, 1),
        (5, 4),
    ]
    integers = IntTestData.from_df(spark.createDataFrame(data, schema=IntTestData.generate_schema()))

    a = tsf.when(integers.a > integers.b, string_literal("A"))
    b = a.when(integers.a == integers.b, string_literal("eq"))
    c = b.otherwise(string_literal("B"))

    result = integers.select(c.alias("when"))

    values = collect_column(result, "when")

    assert values == ["B", "B", "eq", "A"]

    type = result.to_spark().schema["when"].dataType
    assert isinstance(type, StringType)


def test_from_json_with_struct_type(spark: SparkSession):
    """Test from_json overload with StructType schema."""
    class JsonData(DataFrame):
        json_str: String

    data = [
        (json.dumps({"id": 1, "name": "Alice"}),),
        (json.dumps({"id": 2, "name": "Bob"}),),
    ]

    schema = StructType(
        [
            StructField("id", IntegerType()),
            StructField("name", StringType()),
        ]
    )

    df = JsonData.from_df(spark.createDataFrame(data, schema=JsonData.generate_schema()))
    parsed = df.select(tsf.from_json(df.json_str, schema).alias("parsed"))

    result = collect_column(parsed, "parsed")

    assert result[0]["id"] == 1
    assert result[0]["name"] == "Alice"
    assert result[1]["id"] == 2
    assert result[1]["name"] == "Bob"

    parsed_schema = parsed.to_spark().schema["parsed"].dataType
    assert isinstance(parsed_schema, StructType)


def test_bitwise_not(spark: SparkSession):
    class Nums(DataFrame):
        x: Int

    df = Nums.from_df(spark.createDataFrame([(0,), (1,)], schema=Nums.generate_schema()))
    result = df.select(tsf.bitwise_not(df.x).alias("v"))
    vals = collect_column(result, "v")
    assert vals == [-1, -2]
    assert isinstance(result.to_spark().schema["v"].dataType, IntegerType)


def test_crc32(spark: SparkSession):
    class BinaryData(DataFrame):
        b: Binary

    df = BinaryData.from_df(spark.createDataFrame([(bytearray(b"hello"),)], schema=BinaryData.generate_schema()))
    result = df.select(tsf.crc32(df.b).alias("v"))
    vals = collect_column(result, "v")
    assert isinstance(vals[0], int)
    assert isinstance(result.to_spark().schema["v"].dataType, LongType)


def test_equal_null(spark: SparkSession):
    class Nums(DataFrame):
        a: Int
        b: Int

    df = Nums.from_df(spark.createDataFrame([(1, 1), (1, 2), (None, None)], schema=Nums.generate_schema()))
    result = df.select(tsf.equal_null(df.a, df.b).alias("v"))
    assert collect_column(result, "v") == [True, False, True]
    assert isinstance(result.to_spark().schema["v"].dataType, BooleanType)


def test_isnan_isnull_isnotnull(spark: SparkSession):
    class Nums(DataFrame):
        x: Double

    df = Nums.from_df(spark.createDataFrame([(float("nan"),), (None,), (1.0,)], schema=Nums.generate_schema()))
    result = df.select(
        tsf.isnan(df.x).alias("nan"),
        tsf.isnull(df.x).alias("null"),
        tsf.isnotnull(df.x).alias("notnull"),
    )
    rows = collect_values(result)
    assert rows[0]["nan"] is True
    assert rows[1]["null"] is True
    assert rows[2]["notnull"] is True
    for col in ["nan", "null", "notnull"]:
        assert isinstance(result.to_spark().schema[col].dataType, BooleanType)


def test_monotonically_increasing_id(spark: SparkSession):
    class Dummy(DataFrame):
        x: Int

    df = Dummy.from_df(spark.createDataFrame([(1,), (2,), (3,)], schema=Dummy.generate_schema()))
    result = df.select(tsf.monotonically_increasing_id().alias("v"))
    vals = collect_column(result, "v")
    assert len(vals) == 3
    assert isinstance(result.to_spark().schema["v"].dataType, LongType)


def test_nanvl(spark: SparkSession):
    class Nums(DataFrame):
        a: Double
        b: Double

    df = Nums.from_df(
        spark.createDataFrame(
            [(float("nan"), 1.0), (2.0, float("nan"))],
            schema=Nums.generate_schema(),
        )
    )
    result = df.select(tsf.nanvl(df.a, df.b).alias("v"))
    vals = collect_column(result, "v")
    assert vals[0] == pytest.approx(1.0)
    assert vals[1] == pytest.approx(2.0)
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_nullif(spark: SparkSession):
    class Nums(DataFrame):
        a: Int
        b: Int

    df = Nums.from_df(spark.createDataFrame([(1, 1), (2, 3)], schema=Nums.generate_schema()))
    result = df.select(tsf.nullif(df.a, df.b).alias("v"))
    vals = collect_column(result, "v")
    assert vals == [None, 2]
    assert isinstance(result.to_spark().schema["v"].dataType, IntegerType)


def test_nvl(spark: SparkSession):
    class Nums(DataFrame):
        a: Int
        b: Int

    df = Nums.from_df(spark.createDataFrame([(None, 5), (3, 99)], schema=Nums.generate_schema()))
    result = df.select(tsf.nvl(df.a, df.b).alias("v"))
    assert collect_column(result, "v") == [5, 3]
    assert isinstance(result.to_spark().schema["v"].dataType, IntegerType)


def test_nvl2(spark: SparkSession):
    class Nums(DataFrame):
        a: Int
        b: Int
        c: Int

    df = Nums.from_df(spark.createDataFrame([(None, 1, 2), (5, 10, 20)], schema=Nums.generate_schema()))
    result = df.select(tsf.nvl2(df.a, df.b, df.c).alias("v"))
    # nvl2(null, 1, 2) = 2; nvl2(5, 10, 20) = 10
    assert collect_column(result, "v") == [2, 10]
    assert isinstance(result.to_spark().schema["v"].dataType, IntegerType)


def test_spark_partition_id(spark: SparkSession):
    class Dummy(DataFrame):
        x: Int

    df = Dummy.from_df(spark.createDataFrame([(1,)], schema=Dummy.generate_schema()))
    result = df.select(tsf.spark_partition_id().alias("v"))
    vals = collect_column(result, "v")
    assert isinstance(vals[0], int)
    assert isinstance(result.to_spark().schema["v"].dataType, IntegerType)


def test_typeof(spark: SparkSession):
    class Nums(DataFrame):
        x: Int

    df = Nums.from_df(spark.createDataFrame([(1,)], schema=Nums.generate_schema()))
    result = df.select(tsf.typeof(df.x).alias("v"))
    assert collect_column(result, "v") == ["int"]
    assert isinstance(result.to_spark().schema["v"].dataType, StringType)


def test_version(spark: SparkSession):
    class Dummy(DataFrame):
        x: Int

    df = Dummy.from_df(spark.createDataFrame([(1,)], schema=Dummy.generate_schema()))
    result = df.select(tsf.version().alias("v"))
    vals = collect_column(result, "v")
    assert len(vals) == 1
    assert isinstance(vals[0], str)
    assert isinstance(result.to_spark().schema["v"].dataType, StringType)


def test_xxhash64(spark: SparkSession):
    class Nums(DataFrame):
        x: Int

    df = Nums.from_df(spark.createDataFrame([(1,), (2,)], schema=Nums.generate_schema()))
    result = df.select(tsf.xxhash64(df.x).alias("v"))
    vals = collect_column(result, "v")
    assert len(vals) == 2
    assert isinstance(vals[0], int)
    assert isinstance(result.to_spark().schema["v"].dataType, LongType)
