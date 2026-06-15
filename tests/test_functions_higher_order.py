from pyspark.sql import SparkSession

import typespark as ts
from tests.utils import collect_column
from typespark import functions as tsf
from typespark.columns.array import TypedArrayType


def test_filter(spark: SparkSession):
    class Data(ts.DataFrame):
        a: ts.Array[ts.Int]

    df = Data.from_df(
        spark.createDataFrame([([1, 2, 3, 4, 5],)], schema=Data.generate_schema())
    )
    result = df.select(tsf.filter(df.a, lambda x: x > ts.Integer(2)).alias("v"))

    assert collect_column(result, "v") == [[3, 4, 5]]
    assert isinstance(tsf.filter(df.a, lambda x: x > ts.Integer(2)), TypedArrayType)


def test_aggregate(spark: SparkSession):
    class Data(ts.DataFrame):
        a: ts.Array[ts.Int]

    df = Data.from_df(
        spark.createDataFrame([([1, 2, 3, 4],)], schema=Data.generate_schema())
    )
    result = df.select(
        tsf.aggregate(df.a, ts.Integer(0), lambda acc, x: acc + x).alias("v")
    )

    assert collect_column(result, "v") == [10]
    assert isinstance(
        tsf.aggregate(df.a, ts.Integer(0), lambda acc, x: acc + x), ts.Int
    )


def test_transform(spark: SparkSession):
    class Data(ts.DataFrame):
        a: ts.Array[ts.Int]

    df = Data.from_df(
        spark.createDataFrame([([1, 2, 3],)], schema=Data.generate_schema())
    )
    result = df.select(tsf.transform(df.a, lambda x: x * ts.Integer(2)).alias("v"))

    assert collect_column(result, "v") == [[2, 4, 6]]
    assert isinstance(tsf.transform(df.a, lambda x: x * ts.Integer(2)), TypedArrayType)


def test_transform_changes_type(spark: SparkSession):
    class Data(ts.DataFrame):
        a: ts.Array[ts.Int]

    df = Data.from_df(
        spark.createDataFrame([([1, 2, 3],)], schema=Data.generate_schema())
    )
    result = df.select(tsf.transform(df.a, lambda x: ts.Long(x)).alias("v"))

    import pyspark.sql.types as pt

    a: pt.ArrayType = result.to_spark().schema["v"].dataType  # type: ignore

    assert isinstance(a.elementType, pt.LongType)
