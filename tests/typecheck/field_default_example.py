# pyright: reportUnnecessaryTypeIgnoreComment=true
"""ts.field defaults: valid ones must type check, invalid ones must error.

Each invalid default carries a ``# pyright: ignore[...]``. With unnecessary
ignores reported as errors, a default that stops erroring fails the check.
"""

import datetime
import decimal
from enum import StrEnum

import attrs

import typespark as ts


class Color(StrEnum):
    RED = "red"


class Address(ts.Struct):
    city: ts.String
    zip: ts.String = ts.field(default=None)


class Defaults(ts.DataFrame):
    name: ts.String = ts.field(default="x")
    color: ts.StringOf[Color] = ts.field(default=Color.RED)
    flag: ts.Bool = ts.field(default=True)
    short: ts.Short = ts.field(default=1)
    qty: ts.Int = ts.field(default=1)
    total: ts.Long = ts.field(default=5)
    ratio: ts.Double = ts.field(default=1.5)
    ratio_int: ts.Float = ts.field(default=1)
    amount: ts.Decimal = ts.field(default=decimal.Decimal("1.5"))
    amount_float: ts.Decimal = ts.field(default=1.5)
    born: ts.Date = ts.field(default=datetime.date(2000, 1, 1))
    seen: ts.Timestamp = ts.field(default=datetime.datetime(2000, 1, 1))
    seen_ntz: ts.TimestampNTZ = ts.field(default=datetime.datetime(2000, 1, 1))
    tags: ts.Array[ts.String] = ts.field(default=None)
    address: Address = ts.field(default=None)
    nickname: ts.String = ts.field(default=None, df_alias="nick")
    legacy: ts.String = ts.field(default=ts.String("Doe"))
    created: ts.Date = ts.field(factory=lambda: ts.Date(ts.String("2000-01-01")))
    next_qty: ts.Int = ts.field(
        factory=attrs.Factory(lambda self: self.qty + 1, takes_self=True)
    )
    plain: ts.String = ts.field(df_alias="Plain")
    price: ts.Decimal = ts.decimal(18, 4, default=0)
    price_null: ts.Decimal = ts.decimal(18, 4, default=None)
    prices: ts.Array[ts.Decimal] = ts.decimal(18, 4)


def to_long(x: ts.Integer) -> ts.Long:
    return ts.Long(x)


class ConverterDefaults(ts.DataFrame):
    """With a converter, a default is the converter's input (as in attrs)."""

    plain: ts.Long = ts.field(converter=to_long)
    literal: ts.Long = ts.field(converter=to_long, default=5)
    null: ts.Long = ts.field(converter=to_long, default=None)
    column: ts.Long = ts.field(converter=to_long, default=ts.Integer(None))
    factory: ts.Long = ts.field(converter=to_long, factory=lambda: ts.Integer(None))
    lam: ts.Integer = ts.field(converter=lambda x: x + 1, default=5)
    aliased: ts.Integer = ts.field(converter=lambda x: x + 1, df_alias="raw")
    dec: ts.Decimal = ts.decimal(18, 4, converter=lambda x: x, default=0)


class InvalidDefaults(ts.DataFrame):
    a: ts.String = ts.field(default=5)  # pyright: ignore[reportAssignmentType]
    b: ts.Int = ts.field(default="x")  # pyright: ignore[reportAssignmentType]
    c: ts.Int = ts.field(default=1.5)  # pyright: ignore[reportAssignmentType]
    d: ts.Bool = ts.field(default=1)  # pyright: ignore[reportAssignmentType]
    e: ts.Date = ts.field(default="2000-01-01")  # pyright: ignore[reportAssignmentType]
    f: ts.Timestamp = ts.field(default=datetime.date(2000, 1, 1))  # pyright: ignore[reportAssignmentType]
    g: ts.String = ts.field(default=ts.Int(ts.Long(None)))  # pyright: ignore[reportAssignmentType]
    h: ts.Date = ts.field(factory=lambda: ts.String("x"))  # pyright: ignore[reportAssignmentType]
    i: ts.String = ts.field(factory=lambda: "x")  # pyright: ignore[reportCallIssue, reportArgumentType]
    j: ts.String = ts.field(default=[1])  # pyright: ignore[reportCallIssue, reportArgumentType]
    k: ts.String = ts.decimal(18, 4, default=0)  # pyright: ignore[reportAssignmentType]
    m: ts.Decimal = ts.decimal(18, 4, default="0")  # pyright: ignore[reportCallIssue, reportArgumentType]
    conv_a: ts.Long = ts.field(converter=to_long, default="x")  # pyright: ignore[reportCallIssue, reportArgumentType]
    conv_b: ts.Long = ts.field(converter=to_long, factory=lambda: ts.String("x"))  # pyright: ignore[reportArgumentType]
    conv_c: ts.String = ts.field(converter=to_long, default=5)  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    conv_d: ts.String = ts.field(converter=to_long)  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
