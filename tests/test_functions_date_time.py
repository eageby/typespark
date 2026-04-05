import datetime

import pyspark.errors
import pyspark.sql.types
import pytest
from pyspark.sql import SparkSession

import typespark
from tests.utils import collect_column, collect_values
from typespark import functions as tsf


def test_add_months_with_literal(spark: SparkSession):
    class DateTestData(typespark.DataFrame):
        d: typespark.Date

    data = [
        (datetime.date(2020, 1, 31),),
        (datetime.date(2020, 2, 29),),
    ]
    dates = DateTestData.from_df(spark.createDataFrame(data, schema=DateTestData.generate_schema()))

    col: typespark.Date = tsf.add_months(dates.d, 1)

    result = dates.select(col.alias("added"))

    values = collect_column(result, "added")

    assert values == [
        datetime.date(2020, 2, 29),
        datetime.date(2020, 3, 29),
    ]

    assert isinstance(result.to_spark().schema["added"].dataType, pyspark.sql.types.DateType)


def test_add_months_with_col(spark: SparkSession):
    class DateTestData(typespark.DataFrame):
        d: typespark.Date

    data = [
        (datetime.date(2020, 1, 31),),
        (datetime.date(2020, 2, 29),),
    ]
    dates = DateTestData.from_df(spark.createDataFrame(data, schema=DateTestData.generate_schema()))

    col: typespark.Date = tsf.add_months(dates.d, typespark.int_literal(1))
    result = dates.select(col.alias("added"))

    values = collect_column(result, "added")

    assert values == [
        datetime.date(2020, 2, 29),
        datetime.date(2020, 3, 29),
    ]

    assert isinstance(result.to_spark().schema["added"].dataType, pyspark.sql.types.DateType)


def test_date_diff(spark: SparkSession):
    class Dates(typespark.DataFrame):
        a: typespark.Date
        b: typespark.Date

    df = Dates.from_df(
        spark.createDataFrame(
            [(datetime.date(2025, 1, 10), datetime.date(2025, 1, 1))],
            schema=Dates.generate_schema(),
        )
    )
    result = df.select(tsf.date_diff(df.a, df.b).alias("v"))
    assert collect_column(result, "v") == [9]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.IntegerType)


def test_date_format(spark: SparkSession):
    class Dates(typespark.DataFrame):
        d: typespark.Date

    df = Dates.from_df(spark.createDataFrame([(datetime.date(2025, 3, 15),)], schema=Dates.generate_schema()))
    result = df.select(tsf.date_format(df.d, "yyyy/MM/dd").alias("v"))
    assert collect_column(result, "v") == ["2025/03/15"]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.StringType)


def test_date_from_unix_date(spark: SparkSession):
    class Nums(typespark.DataFrame):
        x: typespark.Int

    # Unix date 0 is 1970-01-01
    df = Nums.from_df(spark.createDataFrame([(0,), (1,)], schema=Nums.generate_schema()))
    result = df.select(tsf.date_from_unix_date(df.x).alias("v"))
    vals = collect_column(result, "v")
    assert vals == [datetime.date(1970, 1, 1), datetime.date(1970, 1, 2)]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.DateType)


def test_date_sub(spark: SparkSession):
    class Dates(typespark.DataFrame):
        d: typespark.Date

    df = Dates.from_df(spark.createDataFrame([(datetime.date(2025, 1, 10),)], schema=Dates.generate_schema()))
    result = df.select(tsf.date_sub(df.d, 5).alias("v"))
    assert collect_column(result, "v") == [datetime.date(2025, 1, 5)]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.DateType)


def test_date_trunc(spark: SparkSession):
    class Timestamps(typespark.DataFrame):
        ts: typespark.Timestamp

    df = Timestamps.from_df(
        spark.createDataFrame(
            [(datetime.datetime(2025, 3, 15, 10, 30, 45),)],
            schema=Timestamps.generate_schema(),
        )
    )
    result = df.select(tsf.date_trunc("month", df.ts).alias("v"))
    vals = collect_column(result, "v")
    assert vals[0].year == 2025
    assert vals[0].month == 3
    assert vals[0].day == 1
    assert vals[0].hour == 0
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.TimestampType)


def test_datediff(spark: SparkSession):
    class Dates(typespark.DataFrame):
        a: typespark.Date
        b: typespark.Date

    df = Dates.from_df(
        spark.createDataFrame(
            [(datetime.date(2025, 1, 11), datetime.date(2025, 1, 1))],
            schema=Dates.generate_schema(),
        )
    )
    result = df.select(tsf.datediff(df.a, df.b).alias("v"))
    assert collect_column(result, "v") == [10]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.IntegerType)


def test_dayofmonth_dayofweek_dayofyear(spark: SparkSession):
    class Dates(typespark.DataFrame):
        d: typespark.Date

    df = Dates.from_df(spark.createDataFrame([(datetime.date(2025, 3, 15),)], schema=Dates.generate_schema()))
    result = df.select(
        tsf.dayofmonth(df.d).alias("dom"),
        tsf.dayofweek(df.d).alias("dow"),
        tsf.dayofyear(df.d).alias("doy"),
        tsf.day(df.d).alias("day"),
    )
    vals = collect_values(result)[0]
    assert vals["dom"] == 15
    assert vals["dow"] == 7  # Saturday = 7 in Spark (1=Sunday)
    assert vals["doy"] == 74
    assert vals["day"] == 15
    for col in ["dom", "dow", "doy", "day"]:
        assert isinstance(result.to_spark().schema[col].dataType, pyspark.sql.types.IntegerType)


def test_from_unixtime(spark: SparkSession):
    class Nums(typespark.DataFrame):
        ts: typespark.Long

    df = Nums.from_df(spark.createDataFrame([(0,)], schema=Nums.generate_schema()))
    result = df.select(tsf.from_unixtime(df.ts, "yyyy-MM-dd").alias("v"))
    vals = collect_column(result, "v")
    # epoch 0 in local time — we just check the format is a date string
    assert len(vals[0]) == 10
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.StringType)


def test_hour_minute_second(spark: SparkSession):
    class Timestamps(typespark.DataFrame):
        ts: typespark.Timestamp

    df = Timestamps.from_df(
        spark.createDataFrame(
            [(datetime.datetime(2025, 1, 1, 13, 45, 30),)],
            schema=Timestamps.generate_schema(),
        )
    )
    result = df.select(
        tsf.hour(df.ts).alias("h"),
        tsf.minute(df.ts).alias("m"),
        tsf.second(df.ts).alias("s"),
    )
    vals = collect_values(result)[0]
    assert vals["h"] == 13
    assert vals["m"] == 45
    assert vals["s"] == 30
    for col in ["h", "m", "s"]:
        assert isinstance(result.to_spark().schema[col].dataType, pyspark.sql.types.IntegerType)


def test_last_day(spark: SparkSession):
    class Dates(typespark.DataFrame):
        d: typespark.Date

    df = Dates.from_df(
        spark.createDataFrame(
            [(datetime.date(2025, 2, 10),), (datetime.date(2025, 1, 5),)],
            schema=Dates.generate_schema(),
        )
    )
    result = df.select(tsf.last_day(df.d).alias("v"))
    assert collect_column(result, "v") == [
        datetime.date(2025, 2, 28),
        datetime.date(2025, 1, 31),
    ]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.DateType)


def test_make_date(spark: SparkSession):
    class Nums(typespark.DataFrame):
        y: typespark.Int
        m: typespark.Int
        d: typespark.Int

    df = Nums.from_df(spark.createDataFrame([(2025, 6, 15)], schema=Nums.generate_schema()))
    result = df.select(tsf.make_date(df.y, df.m, df.d).alias("v"))
    assert collect_column(result, "v") == [datetime.date(2025, 6, 15)]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.DateType)


def test_months_between(spark: SparkSession):
    class Dates(typespark.DataFrame):
        a: typespark.Date
        b: typespark.Date

    df = Dates.from_df(
        spark.createDataFrame(
            [(datetime.date(2025, 3, 1), datetime.date(2025, 1, 1))],
            schema=Dates.generate_schema(),
        )
    )
    result = df.select(tsf.months_between(df.a, df.b).alias("v"))
    assert collect_column(result, "v")[0] == pytest.approx(2.0)
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.DoubleType)


def test_next_day(spark: SparkSession):
    class Dates(typespark.DataFrame):
        d: typespark.Date

    df = Dates.from_df(spark.createDataFrame([(datetime.date(2025, 4, 1),)], schema=Dates.generate_schema()))
    result = df.select(tsf.next_day(df.d, "Monday").alias("v"))
    assert collect_column(result, "v") == [datetime.date(2025, 4, 7)]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.DateType)


def test_now(spark: SparkSession):
    class Dummy(typespark.DataFrame):
        x: typespark.Int

    df = Dummy.from_df(spark.createDataFrame([(1,)], schema=Dummy.generate_schema()))
    before = datetime.datetime.now()
    result = df.select(tsf.now().alias("v"))
    vals = collect_column(result, "v")
    after = datetime.datetime.now()
    assert before <= vals[0] <= after
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.TimestampType)


def test_quarter(spark: SparkSession):
    class Dates(typespark.DataFrame):
        d: typespark.Date

    df = Dates.from_df(
        spark.createDataFrame(
            [
                (datetime.date(2025, 1, 1),),
                (datetime.date(2025, 4, 1),),
                (datetime.date(2025, 7, 1),),
                (datetime.date(2025, 10, 1),),
            ],
            schema=Dates.generate_schema(),
        )
    )
    result = df.select(tsf.quarter(df.d).alias("v"))
    assert collect_column(result, "v") == [1, 2, 3, 4]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.IntegerType)


def test_to_timestamp(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    df = Strings.from_df(spark.createDataFrame([("2025-03-15 10:30:00",)], schema=Strings.generate_schema()))
    result = df.select(tsf.to_timestamp(df.s, "yyyy-MM-dd HH:mm:ss").alias("v"))
    vals = collect_column(result, "v")
    assert vals[0].year == 2025
    assert vals[0].month == 3
    assert vals[0].day == 15
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.TimestampType)


def test_trunc(spark: SparkSession):
    class Dates(typespark.DataFrame):
        d: typespark.Date

    df = Dates.from_df(spark.createDataFrame([(datetime.date(2025, 3, 15),)], schema=Dates.generate_schema()))
    result = df.select(tsf.trunc(df.d, "month").alias("v"))
    assert collect_column(result, "v") == [datetime.date(2025, 3, 1)]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.DateType)


def test_unix_date(spark: SparkSession):
    class Dates(typespark.DataFrame):
        d: typespark.Date

    df = Dates.from_df(
        spark.createDataFrame(
            [(datetime.date(1970, 1, 1),), (datetime.date(1970, 1, 2),)],
            schema=Dates.generate_schema(),
        )
    )
    result = df.select(tsf.unix_date(df.d).alias("v"))
    assert collect_column(result, "v") == [0, 1]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.IntegerType)


def test_timestamp_micros_millis_seconds(spark: SparkSession):
    class Nums(typespark.DataFrame):
        x: typespark.Long

    df = Nums.from_df(spark.createDataFrame([(0,)], schema=Nums.generate_schema()))
    result = df.select(
        tsf.timestamp_micros(df.x).alias("micros"),
        tsf.timestamp_millis(df.x).alias("millis"),
        tsf.timestamp_seconds(df.x).alias("seconds"),
    )
    for col in ["micros", "millis", "seconds"]:
        assert isinstance(result.to_spark().schema[col].dataType, pyspark.sql.types.TimestampType)


def test_unix_micros_millis_seconds(spark: SparkSession):
    class Timestamps(typespark.DataFrame):
        ts: typespark.Timestamp

    df = Timestamps.from_df(
        spark.createDataFrame(
            [(datetime.datetime(1970, 1, 1, 0, 0, 0),)],
            schema=Timestamps.generate_schema(),
        )
    )
    result = df.select(
        tsf.unix_micros(df.ts).alias("micros"),
        tsf.unix_millis(df.ts).alias("millis"),
        tsf.unix_seconds(df.ts).alias("seconds"),
    )
    for col in ["micros", "millis", "seconds"]:
        assert isinstance(result.to_spark().schema[col].dataType, pyspark.sql.types.LongType)


def test_weekofyear(spark: SparkSession):
    class Dates(typespark.DataFrame):
        d: typespark.Date

    df = Dates.from_df(spark.createDataFrame([(datetime.date(2025, 1, 6),)], schema=Dates.generate_schema()))
    result = df.select(tsf.weekofyear(df.d).alias("v"))
    assert collect_column(result, "v") == [2]
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.IntegerType)


def test_timestamp_add(spark: SparkSession):
    class Timestamps(typespark.DataFrame):
        ts: typespark.Timestamp

    df = Timestamps.from_df(
        spark.createDataFrame(
            [(datetime.datetime(2025, 1, 1, 0, 0, 0),)],
            schema=Timestamps.generate_schema(),
        )
    )
    result = df.select(tsf.timestamp_add("DAY", typespark.int_literal(10), df.ts).alias("v"))
    vals = collect_column(result, "v")
    assert vals[0].day == 11
    assert isinstance(result.to_spark().schema["v"].dataType, pyspark.sql.types.TimestampType)


def test_to_date_default_format(spark: SparkSession):
    class DateString(typespark.DataFrame):
        date_str: typespark.String

    data = [
        ("2025-01-01",),
    ]

    df = DateString.from_df(spark.createDataFrame(data, schema=DateString.generate_schema()))

    result = df.select(
        tsf.to_date(df.date_str).alias("parsed"),
    )

    parsed = collect_column(result, "parsed")
    assert parsed == [datetime.date(2025, 1, 1)]

    assert isinstance(result.to_spark().schema["parsed"].dataType, pyspark.sql.types.DateType)


def test_to_date_custom_format(spark: SparkSession):
    class DateString(typespark.DataFrame):
        date_str: typespark.String

    data = [
        ("2025:01:01",),
    ]

    df = DateString.from_df(spark.createDataFrame(data, schema=DateString.generate_schema()))

    result = df.select(
        tsf.to_date(df.date_str, "yyyy:MM:dd").alias("parsed"),
    )

    parsed = collect_column(result, "parsed")
    assert parsed == [datetime.date(2025, 1, 1)]

    assert isinstance(result.to_spark().schema["parsed"].dataType, pyspark.sql.types.DateType)


def test_to_date_error(spark: SparkSession):
    class DateString(typespark.DataFrame):
        date_str: typespark.String

    data = [
        ("2025-01-01",),
    ]

    df = DateString.from_df(spark.createDataFrame(data, schema=DateString.generate_schema()))

    result = df.select(
        tsf.to_date(df.date_str, "yyyy:MM:dd").alias("parsed"),
    )

    # Should raise DateTimeException when parsing fails
    with pytest.raises(pyspark.errors.DateTimeException):
        collect_column(result, "parsed")


def test_year_dates(spark: SparkSession):
    class Dates(typespark.DataFrame):
        date: typespark.Date

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

    assert isinstance(result.to_spark().schema["year"].dataType, pyspark.sql.types.IntegerType)


def test_year_timestamps(spark: SparkSession):
    class Dates(typespark.DataFrame):
        date: typespark.Timestamp

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

    assert isinstance(result.to_spark().schema["year"].dataType, pyspark.sql.types.IntegerType)


def test_month_dates(spark: SparkSession):
    class Dates(typespark.DataFrame):
        date: typespark.Date

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

    assert isinstance(result.to_spark().schema["month"].dataType, pyspark.sql.types.IntegerType)


def test_month_timestamps(spark: SparkSession):
    class Dates(typespark.DataFrame):
        date: typespark.Timestamp

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

    assert isinstance(result.to_spark().schema["month"].dataType, pyspark.sql.types.IntegerType)


def test_date_add(spark: SparkSession):
    class Dates(typespark.DataFrame):
        date: typespark.Date

    data = [
        (datetime.datetime(2025, 1, 1),),
        (datetime.datetime(2026, 4, 3),),
        (datetime.datetime(2028, 4, 3),),
    ]

    df = Dates.from_df(spark.createDataFrame(data, schema=Dates.generate_schema()))

    result = df.select(
        tsf.date_add(df.date, typespark.int_literal(1)).alias("added"),
    )

    values = collect_column(result, "added")
    assert values == [
        datetime.date(2025, 1, 2),
        datetime.date(2026, 4, 4),
        datetime.date(2028, 4, 4),
    ]

    assert isinstance(result.to_spark().schema["added"].dataType, pyspark.sql.types.DateType)


def test_current_date(spark: SparkSession):
    class Dummy(typespark.DataFrame):
        _a: typespark.Int

    data = [(1,)]

    df = Dummy.from_df(spark.createDataFrame(data, schema=Dummy.generate_schema()))

    results = df.select(tsf.current_date().alias("date"))
    values = collect_column(results, "date")

    assert values[0] == datetime.date.today()

    assert isinstance(results.to_spark().schema["date"].dataType, pyspark.sql.types.DateType)


def test_current_timestamp(spark: SparkSession):
    class Dummy(typespark.DataFrame):
        _a: typespark.Int

    data = [(1,)]

    timestamp_before = datetime.datetime.now()

    df = Dummy.from_df(spark.createDataFrame(data, schema=Dummy.generate_schema()))

    results = df.select(tsf.current_timestamp().alias("date"))
    values = collect_column(results, "date")

    timestamp_after = datetime.datetime.now()

    assert timestamp_before <= values[0] <= timestamp_after

    assert isinstance(results.to_spark().schema["date"].dataType, pyspark.sql.types.TimestampType)
