import datetime
import math
from decimal import Decimal as PyDecimal

import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DateType,
    DecimalType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    TimestampType,
)

from tests.conftest import Id, Range
from tests.utils import collect_column, collect_values
from typespark import (
    Array,
    Binary,
    DataFrame,
    Date,
    Decimal,
    Double,
    Float,
    Int,
    String,
    Timestamp,
    int_literal,
    string_literal,
)
from typespark import functions as tsf
from typespark.columns.array import TypedArrayType


def test_add_months_with_literal(spark: SparkSession):
    class DateTestData(DataFrame):
        d: Date

    data = [
        (datetime.date(2020, 1, 31),),
        (datetime.date(2020, 2, 29),),
    ]
    dates = DateTestData.from_df(
        spark.createDataFrame(data, schema=DateTestData.generate_schema())
    )

    col: Date = tsf.add_months(dates.d, 1)

    result = dates.select(col.alias("added"))

    values = collect_column(result, "added")

    assert values == [
        datetime.date(2020, 2, 29),
        datetime.date(2020, 3, 29),
    ]

    assert isinstance(result.to_spark().schema["added"].dataType, DateType)


def test_add_months_with_col(spark: SparkSession):

    class DateTestData(DataFrame):
        d: Date

    data = [
        (datetime.date(2020, 1, 31),),
        (datetime.date(2020, 2, 29),),
    ]
    dates = DateTestData.from_df(
        spark.createDataFrame(data, schema=DateTestData.generate_schema())
    )

    col: Date = tsf.add_months(dates.d, int_literal(1))
    result = dates.select(col.alias("added"))

    values = collect_column(result, "added")

    assert values == [
        datetime.date(2020, 2, 29),
        datetime.date(2020, 3, 29),
    ]

    assert isinstance(result.to_spark().schema["added"].dataType, DateType)


def test_array(spark: SparkSession):
    class IntTestData(DataFrame):
        a: Int
        b: Int

    data = [
        (1, 2),
        (1, 3),
        (3, 4),
    ]
    integers = IntTestData.from_df(
        spark.createDataFrame(data, schema=IntTestData.generate_schema())
    )

    col: Array[Int] = tsf.array(integers.a, integers.b)

    result = integers.select(col.alias("array"))

    values = collect_column(result, "array")

    assert values == [[1, 2], [1, 3], [3, 4]]

    type = result.to_spark().schema["array"].dataType
    assert isinstance(type, ArrayType)
    assert isinstance(type.elementType, IntegerType)


def test_array_append(spark: SparkSession):
    class Arrays(DataFrame):
        a: TypedArrayType[Int]

    data = [
        Row(a=[1, 2]),
        Row(a=[3, 4]),
    ]

    df = spark.createDataFrame(data, schema=Arrays.generate_schema())

    arrays = Arrays.from_df(df)
    result = arrays.select(tsf.array_append(arrays.a, int_literal(10)).alias("array"))

    values = collect_column(result, "array")

    assert values == [
        [1, 2, 10],
        [3, 4, 10],
    ]

    assert result.to_spark().schema["array"].dataType == ArrayType(IntegerType(), True)


def test_array_contains(spark: SparkSession):
    class Arrays(DataFrame):
        a: TypedArrayType[Int]

    data = [Row(a=[1, 2]), Row(a=[3, 4])]

    arrays = Arrays.from_df(
        spark.createDataFrame(data, schema=Arrays.generate_schema())
    )
    result = arrays.select(
        tsf.array_contains(arrays.a, int_literal(2)).alias("2"),
        tsf.array_contains(arrays.a, int_literal(10)).alias("10"),
    )

    values = collect_values(result)

    assert values[0]["2"]
    assert not values[0]["10"]
    assert not values[1]["2"]
    assert not values[1]["10"]

    assert result.to_spark().schema["2"].dataType == BooleanType()
    assert result.to_spark().schema["10"].dataType == BooleanType()


def test_atan2_float(spark: SparkSession):
    class TrigTestData(DataFrame):
        y: Float
        x: Float

    data = [
        (1.0, 1.0),  # atan2(1, 1) = pi/4
        (0.0, 1.0),  # atan2(0, 1) = 0
        (-1.0, 1.0),  # atan2(-1, 1) = -pi/4
    ]

    trig = TrigTestData.from_df(
        spark.createDataFrame(data, schema=TrigTestData.generate_schema())
    )

    result = trig.select(tsf.atan2(trig.y, trig.x).alias("angle"))

    values = collect_column(result, "angle")

    assert values == [
        pytest.approx(math.pi / 4),
        pytest.approx(0.0),
        pytest.approx(-math.pi / 4),
    ]

    assert isinstance(
        result.to_spark().schema["angle"].dataType,
        DoubleType,
    )


def test_atan2_double(spark: SparkSession):
    class TrigTestData(DataFrame):
        y: Double
        x: Double

    data = [
        (1.0, 1.0),  # atan2(1, 1) = pi/4
        (0.0, 1.0),  # atan2(0, 1) = 0
        (-1.0, 1.0),  # atan2(-1, 1) = -pi/4
    ]

    trig = TrigTestData.from_df(
        spark.createDataFrame(data, schema=TrigTestData.generate_schema())
    )

    result = trig.select(tsf.atan2(trig.y, trig.x).alias("angle"))

    values = collect_column(result, "angle")

    assert values == [
        pytest.approx(math.pi / 4),
        pytest.approx(0.0),
        pytest.approx(-math.pi / 4),
    ]

    assert isinstance(
        result.to_spark().schema["angle"].dataType,
        DoubleType,
    )


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

    from pyspark.sql.types import IntegerType

    assert isinstance(df.to_spark().schema["coalesced"].dataType, IntegerType)


def test_collect_list(spark: SparkSession):
    """collect_list over column `a` should collect values into a single-row array."""

    class IntTestData(DataFrame):
        a: Int
        b: Int

    data = [
        (1, 2),
        (1, 3),
        (3, 4),
    ]
    integers = IntTestData.from_df(
        spark.createDataFrame(data, schema=IntTestData.generate_schema())
    )
    result = integers.select(tsf.collect_list(integers.a).alias("collected"))

    values = collect_column(result, "collected")

    assert values == [[1, 1, 3]]

    dtype = result.to_spark().schema["collected"].dataType
    assert isinstance(dtype, ArrayType)
    assert isinstance(dtype.elementType, IntegerType)


def test_collect_set(spark: SparkSession):
    """collect_set over column should collect values into a single-row array and remove duplicates."""

    class IntTestData(DataFrame):
        a: Int
        b: Int

    data = [
        (1, 2),
        (1, 3),
        (3, 4),
    ]
    integers = IntTestData.from_df(
        spark.createDataFrame(data, schema=IntTestData.generate_schema())
    )
    result = integers.select(
        tsf.collect_set(integers.a).alias("collected_duplicates"),
        tsf.collect_set(integers.b).alias("collected_unique"),
    )

    duplicate_values = collect_column(result, "collected_duplicates")
    unique_values = collect_column(result, "collected_unique")

    assert duplicate_values == [[1, 3]]
    assert unique_values == [[2, 3, 4]]

    dtype = result.to_spark().schema["collected_duplicates"].dataType
    assert isinstance(dtype, ArrayType)
    assert isinstance(dtype.elementType, IntegerType)

    dtype = result.to_spark().schema["collected_unique"].dataType
    assert isinstance(dtype, ArrayType)
    assert isinstance(dtype.elementType, IntegerType)


def test_concat(spark: SparkSession):
    class Data(DataFrame):
        s: String
        t: String

    data = [
        ("hello", "lo"),
        ("world", "zz"),
    ]
    strings = Data.from_df(spark.createDataFrame(data, schema=Data.generate_schema()))

    result = strings.select(
        tsf.concat(strings.s).alias("no-op"),
        tsf.concat(strings.s, strings.t).alias("concat"),
    )

    values = collect_column(result, "no-op")
    concated_values = collect_column(result, "concat")
    assert values == ["hello", "world"]
    assert concated_values == ["hellolo", "worldzz"]

    assert isinstance(result.to_spark().schema["concat"].dataType, StringType)
    assert isinstance(result.to_spark().schema["no-op"].dataType, StringType)


def test_contains_string_column(spark: SparkSession):
    class Data(DataFrame):
        s: String
        t: String

    data = [
        ("hello", "lo"),
        ("world", "zz"),
    ]

    strings = Data.from_df(spark.createDataFrame(data, schema=Data.generate_schema()))

    result = strings.select(
        tsf.contains(strings.s, strings.t).alias("has"),
        strings.s.alias("s"),
        strings.t.alias("t"),
    )

    rows = collect_values(result)
    assert rows[0]["has"] is True
    assert rows[1]["has"] is False

    assert isinstance(result.to_spark().schema["has"].dataType, BooleanType)


def test_contains_binary_column(spark: SparkSession):
    class BinaryTestData(DataFrame):
        b: Binary
        needle: Binary

    data = [
        (bytearray(b"abcdef"), bytearray(b"bc")),
        (bytearray(b"1234"), bytearray(b"zz")),
    ]

    binaries = BinaryTestData.from_df(
        spark.createDataFrame(data, schema=BinaryTestData.generate_schema())
    )

    result = binaries.select(tsf.contains(binaries.b, binaries.needle).alias("has"))

    values = collect_column(result, "has")
    assert values == [True, False]

    assert isinstance(result.to_spark().schema["has"].dataType, BooleanType)


def test_to_date(spark: SparkSession):
    class DateString(DataFrame):
        date_str: String

    data = [
        ("2025-01-01",),
        ("2025:01:01",),
    ]

    df = DateString.from_df(
        spark.createDataFrame(data, schema=DateString.generate_schema())
    )

    result = df.select(
        tsf.to_date(df.date_str).alias("no_format"),
        tsf.to_date(df.date_str, "yyyy:MM:dd").alias("format"),
    )

    formatted = collect_column(result, "format")
    assert formatted == [None, datetime.date(2025, 1, 1)]
    not_formatted = collect_column(result, "no_format")
    assert not_formatted == [datetime.date(2025, 1, 1), None]

    assert isinstance(result.to_spark().schema["format"].dataType, DateType)
    assert isinstance(result.to_spark().schema["no_format"].dataType, DateType)


def test_year_dates(spark: SparkSession):
    class Dates(DataFrame):
        date: Date

    data = [
        (datetime.date(2025, 1, 1),),
        (datetime.date(2026, 4, 3),),
        (datetime.date(2028, 4, 3),),
    ]

    df = Dates.from_df(spark.createDataFrame(data, schema=Dates.generate_schema()))

    result = df.select(
        tsf.year(df.date).alias("year"),
    )

    values = collect_column(result, "year")
    assert values == [2025, 2026, 2028]

    assert isinstance(result.to_spark().schema["year"].dataType, IntegerType)


def test_year_timestamps(spark: SparkSession):
    class Dates(DataFrame):
        date: Timestamp

    data = [
        (datetime.datetime(2025, 1, 1),),
        (datetime.datetime(2026, 4, 3),),
        (datetime.datetime(2028, 4, 3),),
    ]

    df = Dates.from_df(spark.createDataFrame(data, schema=Dates.generate_schema()))

    result = df.select(
        tsf.year(df.date).alias("year"),
    )

    values = collect_column(result, "year")
    assert values == [2025, 2026, 2028]

    assert isinstance(result.to_spark().schema["year"].dataType, IntegerType)


def test_least_int(spark: SparkSession):
    class Numbers(DataFrame):
        a: Int
        b: Int

    data = [
        (1, 2),
        (4, 2),
        (4, 100),
    ]

    df = Numbers.from_df(spark.createDataFrame(data, schema=Numbers.generate_schema()))

    results = df.select(tsf.least(df.a, df.b).alias("least"))
    values = collect_column(results, "least")

    assert values == [1, 2, 4]
    assert isinstance(results.to_spark().schema["least"].dataType, IntegerType)


def test_greatest_int(spark: SparkSession):
    class Numbers(DataFrame):
        a: Int
        b: Int

    data = [
        (1, 2),
        (4, 2),
        (4, 100),
    ]

    df = Numbers.from_df(spark.createDataFrame(data, schema=Numbers.generate_schema()))

    results = df.select(tsf.greatest(df.a, df.b).alias("least"))
    values = collect_column(results, "least")

    assert values == [2, 4, 100]
    assert isinstance(results.to_spark().schema["least"].dataType, IntegerType)


def test_current_date(spark: SparkSession):
    class Dummy(DataFrame):
        _a: Int

    data = [(1,)]

    df = Dummy.from_df(spark.createDataFrame(data, schema=Dummy.generate_schema()))

    results = df.select(tsf.current_date().alias("date"))
    values = collect_column(results, "date")

    assert values[0] == datetime.date.today()

    assert isinstance(results.to_spark().schema["date"].dataType, DateType)


def test_current_timestamp(spark: SparkSession):
    class Dummy(DataFrame):
        _a: Int

    data = [(1,)]

    timestamp_before = datetime.datetime.now()

    df = Dummy.from_df(spark.createDataFrame(data, schema=Dummy.generate_schema()))

    results = df.select(tsf.current_timestamp().alias("date"))
    values = collect_column(results, "date")

    timestamp_after = datetime.datetime.now()

    assert timestamp_before <= values[0] <= timestamp_after

    assert isinstance(results.to_spark().schema["date"].dataType, TimestampType)


def test_explode(spark: SparkSession):
    class ArrayData(DataFrame):
        elements: Array[Int]

    data = [([1, 2, 3],)]

    df = ArrayData.from_df(
        spark.createDataFrame(data, schema=ArrayData.generate_schema())
    )

    results = df.select(tsf.explode(df.elements).alias("elems"))
    values = collect_column(results, "elems")
    assert values == [1, 2, 3]

    assert isinstance(results.to_spark().schema["elems"].dataType, IntegerType)


def test_month_dates(spark: SparkSession):
    class Dates(DataFrame):
        date: Date

    data = [
        (datetime.date(2025, 1, 1),),
        (datetime.date(2026, 4, 3),),
        (datetime.date(2028, 4, 3),),
    ]

    df = Dates.from_df(spark.createDataFrame(data, schema=Dates.generate_schema()))

    result = df.select(
        tsf.month(df.date).alias("month"),
    )

    values = collect_column(result, "month")
    assert values == [1, 4, 4]

    assert isinstance(result.to_spark().schema["month"].dataType, IntegerType)


def test_month_timestamps(spark: SparkSession):
    class Dates(DataFrame):
        date: Timestamp

    data = [
        (datetime.datetime(2025, 1, 1),),
        (datetime.datetime(2026, 4, 3),),
        (datetime.datetime(2028, 4, 3),),
    ]

    df = Dates.from_df(spark.createDataFrame(data, schema=Dates.generate_schema()))

    result = df.select(
        tsf.month(df.date).alias("month"),
    )

    values = collect_column(result, "month")
    assert values == [1, 4, 4]

    assert isinstance(result.to_spark().schema["month"].dataType, IntegerType)


def test_date_add(spark: SparkSession):
    class Dates(DataFrame):
        date: Date

    data = [
        (datetime.datetime(2025, 1, 1),),
        (datetime.datetime(2026, 4, 3),),
        (datetime.datetime(2028, 4, 3),),
    ]

    df = Dates.from_df(spark.createDataFrame(data, schema=Dates.generate_schema()))

    result = df.select(
        tsf.date_add(df.date, int_literal(1)).alias("added"),
    )

    values = collect_column(result, "added")
    assert values == [
        datetime.date(2025, 1, 2),
        datetime.date(2026, 4, 4),
        datetime.date(2028, 4, 4),
    ]

    assert isinstance(result.to_spark().schema["added"].dataType, DateType)


def test_floor_numeric_no_scale(spark: SparkSession):
    class Nums(DataFrame):
        # choose a floating annotation accepted by your helpers (Double/Float)
        x: Double

    floats = [
        (3.7,),
        (-3.7,),
        (2.1267,),
        (-2.1267,),
    ]

    df_f = Nums.from_df(spark.createDataFrame(floats, schema=Nums.generate_schema()))

    # without scale: numeric -> integer-like results (LongType)
    res = df_f.select(tsf.floor(df_f.x).alias("f_noscale"))
    vals = collect_column(res, "f_noscale")
    assert vals == [3, -4, 2, -3]
    assert isinstance(res.to_spark().schema["f_noscale"].dataType, LongType)


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
    integers = IntTestData.from_df(
        spark.createDataFrame(data, schema=IntTestData.generate_schema())
    )

    a = tsf.when(integers.a > integers.b, string_literal("A"))
    b = a.when(integers.a == integers.b, string_literal("eq"))
    c = b.otherwise(string_literal("B"))

    result = integers.select(c.alias("when"))

    values = collect_column(result, "when")

    assert values == ["B", "B", "eq", "A"]

    type = result.to_spark().schema["when"].dataType
    assert isinstance(type, StringType)


def test_lower(spark: SparkSession):
    class Data(DataFrame):
        s: String

    data = [
        ("hello",),
        ("WORLD",),
        ("MiXeD CaSe",),
    ]

    strings = Data.from_df(spark.createDataFrame(data, schema=Data.generate_schema()))

    result = strings.select(tsf.lower(strings.s).alias("lowered"))

    values = collect_column(result, "lowered")

    assert values == ["hello", "world", "mixed case"]

    assert isinstance(result.to_spark().schema["lowered"].dataType, StringType)


def test_ltrim(spark: SparkSession):
    class Data(DataFrame):
        s: String
        trim: String

    data = [
        ("xxhello", "x"),
        ("xyxworld  ", "xy"),
        ("   both  ", " "),
        ("nochange", "x"),
    ]

    df = Data.from_df(spark.createDataFrame(data, schema=Data.generate_schema()))

    result = df.select(
        tsf.ltrim(df.s).alias("default_trimmed"),
        tsf.ltrim(df.s, df.trim).alias("trimmed"),
    )

    default = collect_column(result, "default_trimmed")
    trimmed = collect_column(result, "trimmed")

    assert default == [
        "xxhello",
        "xyxworld  ",
        "both  ",
        "nochange",
    ]

    assert trimmed == [
        "hello",
        "world  ",
        "both  ",
        "nochange",
    ]

    assert isinstance(
        result.to_spark().schema["default_trimmed"].dataType,
        StringType,
    )
    assert isinstance(
        result.to_spark().schema["trimmed"].dataType,
        StringType,
    )
