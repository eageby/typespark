"""Literal defaults: ``ts.field(default=<literal | None>)`` typed from the annotation."""

import datetime
import decimal

import attrs
import pytest
from pyspark.sql.types import (
    ArrayType,
    DateType,
    DecimalType,
    LongType,
    StringType,
    StructField,
    StructType,
)

import typespark as ts
from typespark import InvalidLiteralDefaultError


class Address(ts.Struct):
    city: ts.String
    zip: ts.String = ts.field(default=None)


class Person(ts.DataFrame):
    name: ts.String
    nickname: ts.String = ts.field(default=None)
    country: ts.String = ts.field(default="SE")
    score: ts.Long = ts.field(default=5)
    balance: ts.Decimal = ts.field(default=decimal.Decimal("1.5"))
    born: ts.Date = ts.field(default=datetime.date(2000, 1, 1))
    tags: ts.Array[ts.String] = ts.field(default=None)
    address: Address = ts.field(default=None)


def test_from_df_fills_literal_defaults_with_typed_columns(spark):
    df = Person.from_df(spark.createDataFrame([("Ada",)], "name string"))
    schema = df.to_df().schema

    assert schema["nickname"].dataType == StringType()
    assert schema["country"].dataType == StringType()
    assert schema["score"].dataType == LongType()
    assert schema["balance"].dataType == DecimalType(10, 0)
    assert schema["born"].dataType == DateType()
    assert schema["tags"].dataType == ArrayType(StringType())
    assert schema["address"].dataType == StructType(
        [StructField("city", StringType()), StructField("zip", StringType())]
    )

    row = df.to_df().collect()[0]
    assert row.nickname is None
    assert row.country == "SE"
    assert row.score == 5
    assert row.born == datetime.date(2000, 1, 1)
    assert row.tags is None
    assert row.address is None


def test_create_applies_literal_defaults(spark):
    df = Person.create([("Ada",)], spark=spark)

    row = df.to_df().collect()[0]
    assert row.nickname is None
    assert row.country == "SE"


def test_provided_column_overrides_literal_default(spark):
    df = Person.from_df(
        spark.createDataFrame([("Ada", "NO")], "name string, country string")
    )

    assert df.to_df().collect()[0].country == "NO"


def test_struct_constructor_applies_literal_default(spark):
    address = Address(city=ts.String("Lund"))

    row = spark.range(1).select(address.to_spark().alias("a")).collect()[0]
    assert row.a.city == "Lund"
    assert row.a.zip is None


def test_typed_column_default_still_supported(spark):
    class Legacy(ts.DataFrame):
        name: ts.String
        surname: ts.String = ts.field(default=ts.String("Doe"))

    df = Legacy.from_df(spark.createDataFrame([("Ada",)], "name string"))

    assert df.to_df().collect()[0].surname == "Doe"


def test_factory_accepts_attrs_factory_with_takes_self(spark):
    class Derived(ts.DataFrame):
        age: ts.Int
        next_age: ts.Int = ts.field(
            factory=attrs.Factory(lambda self: self.age + 1, takes_self=True)
        )

    df = Derived.from_df(spark.createDataFrame([(30,)], "age int"))

    assert df.to_df().collect()[0].next_age == 31


def test_default_and_factory_are_exclusive():
    with pytest.raises(ValueError, match="exclusive"):
        ts.field(default=None, factory=lambda: ts.String("x"))


def test_none_default_on_non_nullable_field_raises():
    with pytest.raises(InvalidLiteralDefaultError, match="nullable=False"):

        class Bad(ts.DataFrame):
            name: ts.String = ts.field(default=None, nullable=False)


def test_non_none_literal_default_on_struct_raises():
    with pytest.raises(InvalidLiteralDefaultError, match="struct or array"):

        class Bad(ts.DataFrame):
            address: Address = ts.field(default="Lund")


def test_non_none_literal_default_on_array_raises():
    with pytest.raises(InvalidLiteralDefaultError, match="struct or array"):

        class Bad(ts.DataFrame):
            tags: ts.Array[ts.String] = ts.field(default="a")


def test_timestamp_ntz_default_keeps_wall_clock(spark):
    """Built via TimestampNTZ(), not lit().cast(), so no session-tz shift."""

    class Seen(ts.DataFrame):
        id: ts.Long
        seen: ts.TimestampNTZ = ts.field(default=datetime.datetime(2000, 1, 1, 12, 0))

    previous = spark.conf.get("spark.sql.session.timeZone")
    spark.conf.set("spark.sql.session.timeZone", "Pacific/Kiritimati")
    try:
        row = Seen.from_df(spark.range(1)).to_df().collect()[0]
    finally:
        spark.conf.set("spark.sql.session.timeZone", previous)

    assert row.seen == datetime.datetime(2000, 1, 1, 12, 0)
