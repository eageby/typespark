import pyspark.sql.types
from pyspark.sql import Row, SparkSession

import typespark as ts
from tests.utils import collect_column, collect_values
from typespark import functions as tsf


def test_array(spark: SparkSession):
    class IntTestData(ts.DataFrame):
        a: ts.Int
        b: ts.Int

    data = [
        (1, 2),
        (1, 3),
        (3, 4),
    ]
    integers = IntTestData.from_df(spark.createDataFrame(data, schema=IntTestData.generate_schema()))

    col: ts.Array[ts.Int] = tsf.array(integers.a, integers.b)

    result = integers.select(col.alias("array"))

    values = collect_column(result, "array")

    assert values == [[1, 2], [1, 3], [3, 4]]

    type = result.to_spark().schema["array"].dataType
    assert isinstance(type, pyspark.sql.types.ArrayType)
    assert isinstance(type.elementType, pyspark.sql.types.IntegerType)


def test_array_append(spark: SparkSession):
    from typespark.columns.array import TypedArrayType

    class Arrays(ts.DataFrame):
        a: TypedArrayType[ts.Int]

    data = [
        Row(a=[1, 2]),
        Row(a=[3, 4]),
    ]

    df = spark.createDataFrame(data, schema=Arrays.generate_schema())

    arrays = Arrays.from_df(df)
    result = arrays.select(tsf.array_append(arrays.a, ts.int_literal(10)).alias("array"))

    values = collect_column(result, "array")

    assert values == [
        [1, 2, 10],
        [3, 4, 10],
    ]

    assert result.to_spark().schema["array"].dataType == pyspark.sql.types.ArrayType(
        pyspark.sql.types.IntegerType(), True
    )


def test_array_contains(spark: SparkSession):
    from typespark.columns.array import TypedArrayType

    class Arrays(ts.DataFrame):
        a: TypedArrayType[ts.Int]

    data = [Row(a=[1, 2]), Row(a=[3, 4])]

    arrays = Arrays.from_df(spark.createDataFrame(data, schema=Arrays.generate_schema()))
    result = arrays.select(
        tsf.array_contains(arrays.a, ts.int_literal(2)).alias("2"),
        tsf.array_contains(arrays.a, ts.int_literal(10)).alias("10"),
    )

    values = collect_values(result)

    assert values[0]["2"]
    assert not values[0]["10"]
    assert not values[1]["2"]
    assert not values[1]["10"]

    assert result.to_spark().schema["2"].dataType == pyspark.sql.types.BooleanType()
    assert result.to_spark().schema["10"].dataType == pyspark.sql.types.BooleanType()


def test_collect_list(spark: SparkSession):
    """collect_list over column `a` should collect values into a single-row array."""

    class IntTestData(ts.DataFrame):
        a: ts.Int
        b: ts.Int

    data = [
        (1, 2),
        (1, 3),
        (3, 4),
    ]
    integers = IntTestData.from_df(spark.createDataFrame(data, schema=IntTestData.generate_schema()))
    result = integers.select(tsf.collect_list(integers.a).alias("collected"))

    values = collect_column(result, "collected")

    assert values == [[1, 1, 3]]

    dtype = result.to_spark().schema["collected"].dataType
    assert isinstance(dtype, pyspark.sql.types.ArrayType)
    assert isinstance(dtype.elementType, pyspark.sql.types.IntegerType)


def test_collect_set(spark: SparkSession):
    """collect_set over column should collect values into a single-row array and remove duplicates."""

    class IntTestData(ts.DataFrame):
        a: ts.Int
        b: ts.Int

    data = [
        (1, 2),
        (1, 3),
        (3, 4),
    ]
    integers = IntTestData.from_df(spark.createDataFrame(data, schema=IntTestData.generate_schema()))
    result = integers.select(
        tsf.collect_set(integers.a).alias("collected_duplicates"),
        tsf.collect_set(integers.b).alias("collected_unique"),
    )

    duplicate_values = collect_column(result, "collected_duplicates")
    unique_values = collect_column(result, "collected_unique")

    assert duplicate_values == [[1, 3]]
    assert unique_values == [[2, 3, 4]]

    dtype = result.to_spark().schema["collected_duplicates"].dataType
    assert isinstance(dtype, pyspark.sql.types.ArrayType)
    assert isinstance(dtype.elementType, pyspark.sql.types.IntegerType)

    dtype = result.to_spark().schema["collected_unique"].dataType
    assert isinstance(dtype, pyspark.sql.types.ArrayType)
    assert isinstance(dtype.elementType, pyspark.sql.types.IntegerType)


def test_explode(spark: SparkSession):
    class ArrayData(ts.DataFrame):
        elements: ts.Array[ts.Int]

    data = [([1, 2, 3],)]

    df = ArrayData.from_df(spark.createDataFrame(data, schema=ArrayData.generate_schema()))

    results = df.select(tsf.explode(df.elements).alias("elems"))
    values = collect_column(results, "elems")
    assert values == [1, 2, 3]

    assert isinstance(results.to_spark().schema["elems"].dataType, pyspark.sql.types.IntegerType)


def test_array_compact(spark: SparkSession):
    class Arrays(ts.DataFrame):
        a: ts.TypedArrayType[ts.Int]

    df = Arrays.from_df(spark.createDataFrame([Row(a=[1, None, 2, None, 3])], schema=Arrays.generate_schema()))
    result = df.select(tsf.array_compact(df.a).alias("v"))
    assert collect_column(result, "v") == [[1, 2, 3]]
    schema_type = result.to_spark().schema["v"].dataType
    assert isinstance(schema_type, pyspark.sql.types.ArrayType)
    assert isinstance(schema_type.elementType, pyspark.sql.types.IntegerType)


def test_array_distinct(spark: SparkSession):
    class Arrays(ts.DataFrame):
        a: ts.TypedArrayType[ts.Int]

    df = Arrays.from_df(spark.createDataFrame([Row(a=[1, 2, 2, 3, 3, 3])], schema=Arrays.generate_schema()))
    result = df.select(tsf.array_distinct(df.a).alias("v"))
    vals = collect_column(result, "v")
    assert sorted(vals[0]) == [1, 2, 3]
    schema_type = result.to_spark().schema["v"].dataType
    assert isinstance(schema_type, pyspark.sql.types.ArrayType)


def test_array_except(spark: SparkSession):
    class Arrays(ts.DataFrame):
        a: ts.TypedArrayType[ts.Int]
        b: ts.TypedArrayType[ts.Int]

    df = Arrays.from_df(spark.createDataFrame([Row(a=[1, 2, 3, 4], b=[2, 4])], schema=Arrays.generate_schema()))
    result = df.select(tsf.array_except(df.a, df.b).alias("v"))
    assert sorted(collect_column(result, "v")[0]) == [1, 3]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.ArrayType)


def test_array_insert(spark: SparkSession):
    class Arrays(ts.DataFrame):
        a: ts.TypedArrayType[ts.Int]

    df = Arrays.from_df(spark.createDataFrame([Row(a=[1, 2, 3])], schema=Arrays.generate_schema()))
    result = df.select(tsf.array_insert(df.a, 2, ts.int_literal(10)).alias("v"))
    assert collect_column(result, "v") == [[1, 10, 2, 3]]


def test_array_intersect(spark: SparkSession):
    class Arrays(ts.DataFrame):
        a: ts.TypedArrayType[ts.Int]
        b: ts.TypedArrayType[ts.Int]

    df = Arrays.from_df(spark.createDataFrame([Row(a=[1, 2, 3, 4], b=[2, 4, 5])], schema=Arrays.generate_schema()))
    result = df.select(tsf.array_intersect(df.a, df.b).alias("v"))
    assert sorted(collect_column(result, "v")[0]) == [2, 4]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.ArrayType)


def test_array_join(spark: SparkSession):
    class Arrays(ts.DataFrame):
        a: ts.TypedArrayType[ts.String]

    df = Arrays.from_df(spark.createDataFrame([Row(a=["a", "b", "c"])], schema=Arrays.generate_schema()))
    result = df.select(tsf.array_join(df.a, ", ").alias("v"))
    assert collect_column(result, "v") == ["a, b, c"]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.StringType)


def test_array_max_min(spark: SparkSession):
    class Arrays(ts.DataFrame):
        a: ts.TypedArrayType[ts.Int]

    df = Arrays.from_df(spark.createDataFrame([Row(a=[3, 1, 4, 1, 5, 9, 2])], schema=Arrays.generate_schema()))
    result = df.select(
        tsf.array_max(df.a).alias("mx"),
        tsf.array_min(df.a).alias("mn"),
    )
    vals = collect_values(result)[0]
    assert vals["mx"] == 9
    assert vals["mn"] == 1
    assert isinstance(result.to_spark().schema["mx"].dataType, pyspark.sql.types.IntegerType)
    assert isinstance(result.to_spark().schema["mn"].dataType, pyspark.sql.types.IntegerType)


def test_array_position(spark: SparkSession):
    class Arrays(ts.DataFrame):
        a: ts.TypedArrayType[ts.Int]

    df = Arrays.from_df(spark.createDataFrame([Row(a=[10, 20, 30])], schema=Arrays.generate_schema()))
    result = df.select(tsf.array_position(df.a, ts.int_literal(20)).alias("v"))
    assert collect_column(result, "v") == [2]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.LongType)


def test_array_prepend(spark: SparkSession):
    class Arrays(ts.DataFrame):
        a: ts.TypedArrayType[ts.Int]

    df = Arrays.from_df(spark.createDataFrame([Row(a=[2, 3])], schema=Arrays.generate_schema()))
    result = df.select(tsf.array_prepend(df.a, ts.int_literal(1)).alias("v"))
    assert collect_column(result, "v") == [[1, 2, 3]]


def test_array_remove(spark: SparkSession):
    class Arrays(ts.DataFrame):
        a: ts.TypedArrayType[ts.Int]

    df = Arrays.from_df(spark.createDataFrame([Row(a=[1, 2, 3, 2, 1])], schema=Arrays.generate_schema()))
    result = df.select(tsf.array_remove(df.a, ts.int_literal(2)).alias("v"))
    assert collect_column(result, "v") == [[1, 3, 1]]


def test_array_repeat(spark: SparkSession):
    class Nums(ts.DataFrame):
        x: ts.Int

    df = Nums.from_df(spark.createDataFrame([(5,)], schema=Nums.generate_schema()))
    result = df.select(tsf.array_repeat(df.x, 3).alias("v"))
    assert collect_column(result, "v") == [[5, 5, 5]]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.ArrayType)


def test_array_size(spark: SparkSession):
    class Arrays(ts.DataFrame):
        a: ts.TypedArrayType[ts.Int]

    df = Arrays.from_df(spark.createDataFrame([Row(a=[1, 2, 3, 4, 5])], schema=Arrays.generate_schema()))
    result = df.select(tsf.array_size(df.a).alias("v"))
    assert collect_column(result, "v") == [5]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.IntegerType)


def test_array_sort(spark: SparkSession):
    class Arrays(ts.DataFrame):
        a: ts.TypedArrayType[ts.Int]

    df = Arrays.from_df(spark.createDataFrame([Row(a=[3, 1, 4, 1, 5])], schema=Arrays.generate_schema()))
    result = df.select(tsf.array_sort(df.a).alias("v"))
    assert collect_column(result, "v") == [[1, 1, 3, 4, 5]]


def test_array_union(spark: SparkSession):
    class Arrays(ts.DataFrame):
        a: ts.TypedArrayType[ts.Int]
        b: ts.TypedArrayType[ts.Int]

    df = Arrays.from_df(spark.createDataFrame([Row(a=[1, 2, 3], b=[3, 4, 5])], schema=Arrays.generate_schema()))
    result = df.select(tsf.array_union(df.a, df.b).alias("v"))
    assert sorted(collect_column(result, "v")[0]) == [1, 2, 3, 4, 5]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.ArrayType)


def test_arrays_overlap(spark: SparkSession):
    class Arrays(ts.DataFrame):
        a: ts.TypedArrayType[ts.Int]
        b: ts.TypedArrayType[ts.Int]

    df = Arrays.from_df(
        spark.createDataFrame(
            [Row(a=[1, 2, 3], b=[3, 4, 5]), Row(a=[1, 2], b=[4, 5])],
            schema=Arrays.generate_schema(),
        )
    )
    result = df.select(tsf.arrays_overlap(df.a, df.b).alias("v"))
    assert collect_column(result, "v") == [True, False]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.BooleanType)


def test_cardinality_size(spark: SparkSession):
    class Arrays(ts.DataFrame):
        a: ts.TypedArrayType[ts.Int]

    df = Arrays.from_df(spark.createDataFrame([Row(a=[1, 2, 3])], schema=Arrays.generate_schema()))
    result = df.select(
        tsf.cardinality(df.a).alias("card"),
        tsf.size(df.a).alias("sz"),
    )
    vals = collect_values(result)[0]
    assert vals["card"] == 3
    assert vals["sz"] == 3
    assert isinstance(result.to_spark().schema["card"].dataType, pyspark.sql.types.IntegerType)


def test_element_at(spark: SparkSession):
    class Arrays(ts.DataFrame):
        a: ts.TypedArrayType[ts.Int]

    df = Arrays.from_df(spark.createDataFrame([Row(a=[10, 20, 30])], schema=Arrays.generate_schema()))
    result = df.select(tsf.element_at(df.a, 2).alias("v"))
    assert collect_column(result, "v") == [20]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.IntegerType)


def test_explode_outer(spark: SparkSession):
    class Arrays(ts.DataFrame):
        a: ts.TypedArrayType[ts.Int]

    df = Arrays.from_df(spark.createDataFrame([Row(a=[1, 2]), Row(a=None)], schema=Arrays.generate_schema()))
    result = df.select(tsf.explode_outer(df.a).alias("v"))
    vals = collect_column(result, "v")
    assert vals == [1, 2, None]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.IntegerType)


def test_flatten(spark: SparkSession):
    class Arrays(ts.DataFrame):
        a: ts.TypedArrayType[ts.TypedArrayType[ts.Int]]

    df = Arrays.from_df(
        spark.createDataFrame(
            [Row(a=[[1, 2], [3, 4]])],
            "a array<array<int>>",
        )
    )

    result = df.select(tsf.flatten(df.a).alias("v"))
    assert collect_column(result, "v") == [[1, 2, 3, 4]]
    schema_type = result.schema["v"].dataType
    assert isinstance(schema_type, pyspark.sql.types.ArrayType)
    assert isinstance(schema_type.elementType, pyspark.sql.types.IntegerType)


def test_shuffle(spark: SparkSession):
    class Arrays(ts.DataFrame):
        a: ts.TypedArrayType[ts.Int]

    df = Arrays.from_df(spark.createDataFrame([Row(a=[1, 2, 3, 4, 5])], schema=Arrays.generate_schema()))
    result = df.select(tsf.shuffle(df.a).alias("v"))
    vals = collect_column(result, "v")
    assert sorted(vals[0]) == [1, 2, 3, 4, 5]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.ArrayType)


def test_slice(spark: SparkSession):
    class Arrays(ts.DataFrame):
        a: ts.TypedArrayType[ts.Int]

    df = Arrays.from_df(spark.createDataFrame([Row(a=[1, 2, 3, 4, 5])], schema=Arrays.generate_schema()))
    result = df.select(tsf.slice(df.a, 2, 3).alias("v"))
    assert collect_column(result, "v") == [[2, 3, 4]]
    schema_type = result.to_spark().schema["v"].dataType
    assert isinstance(schema_type, pyspark.sql.types.ArrayType)
    assert isinstance(schema_type.elementType, pyspark.sql.types.IntegerType)


def test_sort_array(spark: SparkSession):
    class Arrays(ts.DataFrame):
        a: ts.TypedArrayType[ts.Int]

    df = Arrays.from_df(spark.createDataFrame([Row(a=[3, 1, 4, 1, 5])], schema=Arrays.generate_schema()))
    result_asc = df.select(tsf.sort_array(df.a).alias("v"))
    result_desc = df.select(tsf.sort_array(df.a, asc=False).alias("v"))
    assert collect_column(result_asc, "v") == [[1, 1, 3, 4, 5]]
    assert collect_column(result_desc, "v") == [[5, 4, 3, 1, 1]]


def test_sequence_int(spark: SparkSession):
    class Nums(ts.DataFrame):
        a: ts.Int
        b: ts.Int

    df = Nums.from_df(spark.createDataFrame([(1, 5)], schema=Nums.generate_schema()))
    result = df.select(tsf.sequence(df.a, df.b).alias("v"))
    assert collect_column(result, "v") == [[1, 2, 3, 4, 5]]
    schema_type = result.to_spark().schema["v"].dataType
    assert isinstance(schema_type, pyspark.sql.types.ArrayType)
    assert isinstance(schema_type.elementType, pyspark.sql.types.IntegerType)
