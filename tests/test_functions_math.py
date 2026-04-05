import math

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType, LongType

from typespark import DataFrame, Double, Float, Int, int_literal
from typespark import functions as tsf

from .utils import collect_column


def test_atan2_float(spark: SparkSession):
    class TrigTestData(DataFrame):
        y: Float
        x: Float

    data = [
        (1.0, 1.0),  # atan2(1, 1) = pi/4
        (0.0, 1.0),  # atan2(0, 1) = 0
        (-1.0, 1.0),  # atan2(-1, 1) = -pi/4
    ]

    trig = TrigTestData.from_df(spark.createDataFrame(data, schema=TrigTestData.generate_schema()))

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

    trig = TrigTestData.from_df(spark.createDataFrame(data, schema=TrigTestData.generate_schema()))

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


def test_abs_positive(spark: SparkSession):
    class Nums(DataFrame):
        x: Double

    df = Nums.from_df(spark.createDataFrame([(-3.5,), (2.0,)], schema=Nums.generate_schema()))
    result = df.select(tsf.abs(df.x).alias("a"))
    assert collect_column(result, "a") == [pytest.approx(3.5), pytest.approx(2.0)]
    assert isinstance(result.to_spark().schema["a"].dataType, DoubleType)


def test_acos(spark: SparkSession):
    class Nums(DataFrame):
        x: Double

    df = Nums.from_df(spark.createDataFrame([(1.0,), (0.0,)], schema=Nums.generate_schema()))
    result = df.select(tsf.acos(df.x).alias("v"))
    vals = collect_column(result, "v")
    assert vals[0] == pytest.approx(0.0)
    assert vals[1] == pytest.approx(math.pi / 2)
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_asin(spark: SparkSession):
    class Nums(DataFrame):
        x: Double

    df = Nums.from_df(spark.createDataFrame([(0.0,), (1.0,)], schema=Nums.generate_schema()))
    result = df.select(tsf.asin(df.x).alias("v"))
    vals = collect_column(result, "v")
    assert vals[0] == pytest.approx(0.0)
    assert vals[1] == pytest.approx(math.pi / 2)
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_atan(spark: SparkSession):
    class Nums(DataFrame):
        x: Double

    df = Nums.from_df(spark.createDataFrame([(0.0,), (1.0,)], schema=Nums.generate_schema()))
    result = df.select(tsf.atan(df.x).alias("v"))
    vals = collect_column(result, "v")
    assert vals[0] == pytest.approx(0.0)
    assert vals[1] == pytest.approx(math.pi / 4)
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_bround(spark: SparkSession):
    class Nums(DataFrame):
        x: Double

    df = Nums.from_df(spark.createDataFrame([(2.5,), (3.5,)], schema=Nums.generate_schema()))
    result = df.select(tsf.bround(df.x, 0).alias("v"))
    vals = collect_column(result, "v")
    # banker's rounding: 2.5 → 2, 3.5 → 4
    assert vals == [pytest.approx(2.0), pytest.approx(4.0)]
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_cbrt(spark: SparkSession):
    class Nums(DataFrame):
        x: Double

    df = Nums.from_df(spark.createDataFrame([(27.0,), (-8.0,)], schema=Nums.generate_schema()))
    result = df.select(tsf.cbrt(df.x).alias("v"))
    vals = collect_column(result, "v")
    assert vals[0] == pytest.approx(3.0)
    assert vals[1] == pytest.approx(-2.0)
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_ceil(spark: SparkSession):
    class Nums(DataFrame):
        x: Double

    df = Nums.from_df(spark.createDataFrame([(1.2,), (-1.2,)], schema=Nums.generate_schema()))
    result = df.select(tsf.ceil(df.x).alias("v"))
    vals = collect_column(result, "v")
    assert vals == [2, -1]
    assert isinstance(result.to_spark().schema["v"].dataType, LongType)


def test_cosh(spark: SparkSession):
    class Nums(DataFrame):
        x: Double

    df = Nums.from_df(spark.createDataFrame([(0.0,)], schema=Nums.generate_schema()))
    result = df.select(tsf.cosh(df.x).alias("v"))
    assert collect_column(result, "v")[0] == pytest.approx(1.0)
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_degrees(spark: SparkSession):
    class Nums(DataFrame):
        x: Double

    df = Nums.from_df(spark.createDataFrame([(math.pi,), (0.0,)], schema=Nums.generate_schema()))
    result = df.select(tsf.degrees(df.x).alias("v"))
    vals = collect_column(result, "v")
    assert vals[0] == pytest.approx(180.0)
    assert vals[1] == pytest.approx(0.0)
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_e(spark: SparkSession):
    class Dummy(DataFrame):
        x: Int

    df = Dummy.from_df(spark.createDataFrame([(1,)], schema=Dummy.generate_schema()))
    result = df.select(tsf.e().alias("v"))
    assert collect_column(result, "v")[0] == pytest.approx(math.e)
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_exp(spark: SparkSession):
    class Nums(DataFrame):
        x: Double

    df = Nums.from_df(spark.createDataFrame([(0.0,), (1.0,)], schema=Nums.generate_schema()))
    result = df.select(tsf.exp(df.x).alias("v"))
    vals = collect_column(result, "v")
    assert vals[0] == pytest.approx(1.0)
    assert vals[1] == pytest.approx(math.e)
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_factorial(spark: SparkSession):
    class Nums(DataFrame):
        x: Int

    df = Nums.from_df(spark.createDataFrame([(5,), (0,)], schema=Nums.generate_schema()))
    result = df.select(tsf.factorial(df.x).alias("v"))
    vals = collect_column(result, "v")
    assert vals == [120, 1]
    assert isinstance(result.to_spark().schema["v"].dataType, LongType)


def test_hypot(spark: SparkSession):
    class Nums(DataFrame):
        a: Double
        b: Double

    df = Nums.from_df(spark.createDataFrame([(3.0, 4.0)], schema=Nums.generate_schema()))
    result = df.select(tsf.hypot(df.a, df.b).alias("v"))
    assert collect_column(result, "v")[0] == pytest.approx(5.0)
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_ln(spark: SparkSession):
    class Nums(DataFrame):
        x: Double

    df = Nums.from_df(spark.createDataFrame([(1.0,), (math.e,)], schema=Nums.generate_schema()))
    result = df.select(tsf.ln(df.x).alias("v"))
    vals = collect_column(result, "v")
    assert vals[0] == pytest.approx(0.0)
    assert vals[1] == pytest.approx(1.0)
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_log_natural(spark: SparkSession):
    class Nums(DataFrame):
        x: Double

    df = Nums.from_df(spark.createDataFrame([(1.0,), (math.e,)], schema=Nums.generate_schema()))
    result = df.select(tsf.log(df.x).alias("v"))
    vals = collect_column(result, "v")
    assert vals[0] == pytest.approx(0.0)
    assert vals[1] == pytest.approx(1.0)
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_log_with_base(spark: SparkSession):
    class Nums(DataFrame):
        x: Double

    df = Nums.from_df(spark.createDataFrame([(100.0,), (10.0,)], schema=Nums.generate_schema()))
    result = df.select(tsf.log(df.x, base=10.0).alias("v"))
    vals = collect_column(result, "v")
    assert vals[0] == pytest.approx(2.0)
    assert vals[1] == pytest.approx(1.0)
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_log2(spark: SparkSession):
    class Nums(DataFrame):
        x: Double

    df = Nums.from_df(spark.createDataFrame([(8.0,), (1.0,)], schema=Nums.generate_schema()))
    result = df.select(tsf.log2(df.x).alias("v"))
    vals = collect_column(result, "v")
    assert vals[0] == pytest.approx(3.0)
    assert vals[1] == pytest.approx(0.0)
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_log10(spark: SparkSession):
    class Nums(DataFrame):
        x: Double

    df = Nums.from_df(spark.createDataFrame([(1000.0,)], schema=Nums.generate_schema()))
    result = df.select(tsf.log10(df.x).alias("v"))
    assert collect_column(result, "v")[0] == pytest.approx(3.0)
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_negate(spark: SparkSession):
    class Nums(DataFrame):
        x: Double

    df = Nums.from_df(spark.createDataFrame([(3.0,), (-1.5,)], schema=Nums.generate_schema()))
    result = df.select(tsf.negate(df.x).alias("v"))
    vals = collect_column(result, "v")
    assert vals == [pytest.approx(-3.0), pytest.approx(1.5)]
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_pi(spark: SparkSession):
    class Dummy(DataFrame):
        x: Int

    df = Dummy.from_df(spark.createDataFrame([(1,)], schema=Dummy.generate_schema()))
    result = df.select(tsf.pi().alias("v"))
    assert collect_column(result, "v")[0] == pytest.approx(math.pi)
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_pmod(spark: SparkSession):
    class Nums(DataFrame):
        a: Int
        b: Int

    df = Nums.from_df(spark.createDataFrame([(-7, 3), (7, 3)], schema=Nums.generate_schema()))
    result = df.select(tsf.pmod(df.a, df.b).alias("v"))
    vals = collect_column(result, "v")
    # pmod always returns non-negative: -7 % 3 = 2
    assert vals == [2, 1]
    assert isinstance(result.to_spark().schema["v"].dataType, IntegerType)


def test_pow(spark: SparkSession):
    class Nums(DataFrame):
        x: Double

    df = Nums.from_df(spark.createDataFrame([(2.0,), (3.0,)], schema=Nums.generate_schema()))
    result = df.select(tsf.pow(df.x, 3.0).alias("v"))
    vals = collect_column(result, "v")
    assert vals == [pytest.approx(8.0), pytest.approx(27.0)]
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_radians(spark: SparkSession):
    class Nums(DataFrame):
        x: Double

    df = Nums.from_df(spark.createDataFrame([(180.0,), (0.0,)], schema=Nums.generate_schema()))
    result = df.select(tsf.radians(df.x).alias("v"))
    vals = collect_column(result, "v")
    assert vals[0] == pytest.approx(math.pi)
    assert vals[1] == pytest.approx(0.0)
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_rand(spark: SparkSession):
    class Dummy(DataFrame):
        x: Int

    df = Dummy.from_df(spark.createDataFrame([(1,), (2,)], schema=Dummy.generate_schema()))
    result = df.select(tsf.rand(seed=42).alias("v"))
    vals = collect_column(result, "v")
    assert len(vals) == 2
    assert all(0.0 <= v < 1.0 for v in vals)
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_round(spark: SparkSession):
    class Nums(DataFrame):
        x: Double

    df = Nums.from_df(spark.createDataFrame([(2.567,), (1.234,)], schema=Nums.generate_schema()))
    result = df.select(tsf.round(df.x, 2).alias("v"))
    vals = collect_column(result, "v")
    assert vals == [pytest.approx(2.57), pytest.approx(1.23)]


def test_sign(spark: SparkSession):
    class Nums(DataFrame):
        x: Double

    df = Nums.from_df(spark.createDataFrame([(-5.0,), (0.0,), (3.0,)], schema=Nums.generate_schema()))
    result = df.select(tsf.sign(df.x).alias("v"))
    vals = collect_column(result, "v")
    assert vals == [pytest.approx(-1.0), pytest.approx(0.0), pytest.approx(1.0)]
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_sqrt(spark: SparkSession):
    class Nums(DataFrame):
        x: Double

    df = Nums.from_df(spark.createDataFrame([(4.0,), (9.0,)], schema=Nums.generate_schema()))
    result = df.select(tsf.sqrt(df.x).alias("v"))
    vals = collect_column(result, "v")
    assert vals == [pytest.approx(2.0), pytest.approx(3.0)]
    assert isinstance(result.to_spark().schema["v"].dataType, DoubleType)


def test_width_bucket(spark: SparkSession):
    class Nums(DataFrame):
        v: Double

    df = Nums.from_df(spark.createDataFrame([(0.5,), (1.5,), (2.5,)], schema=Nums.generate_schema()))
    result = df.select(tsf.width_bucket(df.v, int_literal(0), int_literal(3), 3).alias("bucket"))
    vals = collect_column(result, "bucket")
    assert vals == [1, 2, 3]
    assert isinstance(result.to_spark().schema["bucket"].dataType, LongType)
