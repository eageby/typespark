"""Literal defaults: ``ts.field(default=<literal | None>)`` typed from the annotation."""

import datetime
import decimal

import attrs
import pyspark.sql.functions as F
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
import typespark.functions as tsf
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
        ts.field(default=None, factory=lambda: ts.String("x"))  # type: ignore  # invalid on purpose


def test_none_default_on_non_nullable_field_raises():
    with pytest.raises(InvalidLiteralDefaultError, match="nullable=False"):

        class Bad(ts.DataFrame):
            name: ts.String = ts.field(default=None, nullable=False)


def test_non_none_literal_default_on_struct_raises():
    with pytest.raises(InvalidLiteralDefaultError, match="struct or array"):

        class Bad(ts.DataFrame):
            address: Address = ts.field(default="Lund")  # type: ignore  # invalid on purpose


def test_non_none_literal_default_on_array_raises():
    with pytest.raises(InvalidLiteralDefaultError, match="struct or array"):

        class Bad(ts.DataFrame):
            tags: ts.Array[ts.String] = ts.field(default="a")  # type: ignore  # invalid on purpose


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


# --- converters: a default is the converter's input, as in attrs ---------------


def to_long_plus_one(x: ts.Integer) -> ts.Long:
    assert type(x) is ts.Integer, f"converter got {type(x).__name__}"
    return ts.Long(x + 1)


class Converted(ts.DataFrame):
    id: ts.Long
    total: ts.Long = ts.field(converter=to_long_plus_one, default=5)
    bumped: ts.Integer = ts.field(converter=lambda x: x + 100, default=1)
    from_factory: ts.Integer = ts.field(
        converter=lambda x: x + 100, factory=lambda: ts.Integer(2)
    )
    raw: ts.Integer = ts.field(converter=lambda x: x + 1, df_alias="raw_n")


def test_literal_default_built_as_converter_input_type(spark):
    """to_long_plus_one asserts it receives an Integer, not the annotated Long."""
    df = Converted.from_df(spark.range(1).selectExpr("id", "1 as raw_n"))

    assert df.to_df().schema["total"].dataType == LongType()
    assert df.to_df().collect()[0].total == 6


def test_from_df_converts_defaults_not_read_columns(spark):
    df = Converted.from_df(spark.range(1).selectExpr("id", "1 as raw_n"))

    row = df.to_df().collect()[0]
    assert (row.total, row.bumped, row.from_factory) == (6, 101, 102)
    # Read columns are taken as already converted.
    assert row.raw_n == 1


def test_converted_default_is_not_reapplied_on_rewrap(spark):
    df = Converted.from_df(spark.range(1).selectExpr("id", "1 as raw_n"))
    rewrapped = Converted.from_df(df.to_df())
    again = rewrapped.filter(rewrapped.id == 0)

    row = again.to_df().collect()[0]
    assert (row.total, row.bumped, row.from_factory) == (6, 101, 102)


def test_create_converts_defaults(spark):
    df = Converted.create([(0, 1)], spark=spark)

    row = df.to_df().collect()[0]
    assert (row.total, row.bumped, row.from_factory) == (6, 101, 102)


def test_struct_constructor_converts_literal_default(spark):
    class Box(ts.Struct):
        size: ts.Long = ts.field(converter=to_long_plus_one, default=5)

    row = spark.range(1).select(Box().to_spark().alias("b")).collect()[0]
    assert row.b.size == 6


def test_none_default_with_converter_skips_nullable_check(spark):
    class Filled(ts.DataFrame):
        id: ts.Long
        name: ts.String = ts.field(
            converter=lambda x: ts.String(F.coalesce(x.to_spark(), F.lit("n/a"))),
            default=None,
            nullable=False,
        )

    assert Filled.from_df(spark.range(1)).to_df().collect()[0].name == "n/a"


# --- ts.decimal: literal defaults use the declared precision/scale -----------


class Money(ts.Struct):
    amount: ts.Decimal = ts.decimal(13, 6, default=None)


def _types(schema: StructType) -> dict[str, object]:
    return {f.name: f.dataType for f in schema.fields}


class Priced(ts.DataFrame):
    id: ts.Long
    price: ts.Decimal = ts.decimal(13, 6, default=None)
    rate: ts.Decimal = ts.decimal(3, 1, default=1.5)
    prices: ts.Array[ts.Decimal] = ts.decimal(18, 4, default=None)
    money: Money = ts.field(default=None)


def test_decimal_literal_default_matches_generated_schema(spark):
    df = Priced.from_df(spark.range(1))

    assert _types(df.to_df().schema) == _types(Priced.generate_schema())
    assert df.to_df().schema["price"].dataType == DecimalType(13, 6)
    assert df.to_df().schema["rate"].dataType == DecimalType(3, 1)
    assert df.to_df().schema["prices"].dataType == ArrayType(DecimalType(18, 4))
    assert df.to_df().collect()[0].rate == decimal.Decimal("1.5")


def test_create_decimal_literal_default_matches_generated_schema(spark):
    df = Priced.create([(0,)], spark=spark)

    assert _types(df.to_df().schema) == _types(Priced.generate_schema())


def test_struct_constructor_decimal_literal_default(spark):
    row_schema = spark.range(1).select(Money().to_spark().alias("m")).schema

    assert row_schema["m"].dataType == Money.generate_schema()


def test_decimal_metadata_not_applied_to_converter_input(spark):
    """With a typed converter, the literal is the converter's input type."""

    def to_dec(x: ts.Integer) -> ts.Decimal:
        assert type(x) is ts.Integer
        return ts.Decimal(x, precision=13, scale=6)

    class Conv(ts.DataFrame):
        id: ts.Long
        amount: ts.Decimal = ts.decimal(13, 6, converter=to_dec, default=2)

    df = Conv.from_df(spark.range(1))
    assert df.to_df().schema["amount"].dataType == DecimalType(13, 6)


def test_struct_built_from_its_defaults_as_factory(spark):
    class Props(ts.Struct):
        kind: ts.String = ts.field(default="property")
        note: ts.String = ts.field(default=None)

    class Holder(ts.DataFrame):
        id: ts.Long
        props: Props = ts.field(factory=lambda: Props())

    row = Holder.from_df(spark.range(1)).to_df().collect()[0]
    assert row.props.kind == "property"
    assert row.props.note is None


# ── Defaulted columns become references after wrapping ───────────────────────


def _running(self: "_Ledger") -> ts.Long:
    return tsf.sum(self.amount).over(ts.Window().orderBy(self.day))


class _Ledger(ts.DataFrame):
    day: ts.Integer
    amount: ts.Integer
    running: ts.Long = ts.field(default=attrs.Factory(_running, takes_self=True))


def _previous(ledger: _Ledger) -> list:
    prev = tsf.lag(ledger.running).over(ts.Window().orderBy(ledger.day))
    return [r[0] for r in ledger.to_df().select(prev.to_spark()).collect()]


def test_create_default_is_reference(spark):
    """Regression: a defaulted column stayed an expression, so a window over it nested."""
    ledger = _Ledger.create([(1, 10), (2, 20), (3, 5)], spark)

    assert str(ledger.running.to_spark()) == "Column<'running'>"
    assert _previous(ledger) == [None, 10, 30]


def test_from_df_default_is_reference(spark):
    raw = spark.createDataFrame([(1, 10), (2, 20)], ["day", "amount"])
    ledger = _Ledger.from_df(raw)

    assert str(ledger.running.to_spark()) == "Column<'running'>"
    assert _previous(ledger) == [None, 10]


def test_from_df_aliased_default_is_reference(spark):
    raw = spark.createDataFrame([(1, 10), (2, 20)], ["day", "amount"])
    ledger = _Ledger.from_df(raw, alias="l")

    assert str(ledger.running.to_spark()) == "Column<'l.running'>"
    assert _previous(ledger) == [None, 10]


def test_constructor_default_is_reference(spark):
    class Raw(ts.DataFrame):
        day: ts.Integer
        amount: ts.Integer

    raw = Raw.create([(1, 10), (2, 20)], spark)
    ledger = _Ledger(raw, day=raw.day, amount=raw.amount)

    assert str(ledger.running.to_spark()) == "Column<'running'>"
    assert _previous(ledger) == [None, 10]


def test_from_df_read_column_unaffected(spark):
    raw = spark.createDataFrame([(1, 10, 99)], ["day", "amount", "running"])
    ledger = _Ledger.from_df(raw)

    assert str(ledger.running.to_spark()) == "Column<'running'>"
    assert [r["running"] for r in ledger.to_df().collect()] == [99]
