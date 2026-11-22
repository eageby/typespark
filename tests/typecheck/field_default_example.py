# pyright: reportUnnecessaryTypeIgnoreComment=true
# mypy: warn-unused-ignores
"""ts.field defaults: valid ones must type check, invalid ones must error.

Each invalid default carries a ``# type: ignore``. With unnecessary
ignores reported as errors, a default that stops erroring fails the check.
"""

import datetime
import decimal
from enum import StrEnum

import attrs

import typespark as ts


def _int_to_dec(x: ts.Integer) -> ts.Decimal:
    return ts.Decimal(x, precision=18, scale=4)


class Color(StrEnum):
    RED = "red"


class Address(ts.Struct):
    city: ts.String
    zip: ts.String = ts.field(default=None)


class Props(ts.Struct):
    """All fields defaulted, so the struct can be built with no arguments."""

    kind: ts.String = ts.field(default="property")
    note: ts.String = ts.field(default=None)


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
    # Struct with literal defaults as a factory: wrap it in a lambda (see below).
    props: Props = ts.field(factory=lambda: Props())
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
    dec_from_int: ts.Decimal = ts.decimal(18, 4, converter=_int_to_dec, default=2)


class InvalidDefaults(ts.DataFrame):
    a: ts.String = ts.field(default=5)  # type: ignore
    b: ts.Int = ts.field(default="x")  # type: ignore
    c: ts.Int = ts.field(default=1.5)  # type: ignore
    d: ts.Bool = ts.field(default=1)  # type: ignore
    e: ts.Date = ts.field(default="2000-01-01")  # type: ignore
    f: ts.Timestamp = ts.field(default=datetime.date(2000, 1, 1))  # type: ignore
    g: ts.String = ts.field(default=ts.Int(ts.Long(None)))  # type: ignore
    h: ts.Date = ts.field(factory=lambda: ts.String("x"))  # type: ignore
    i: ts.String = ts.field(factory=lambda: "x")  # type: ignore
    j: ts.String = ts.field(default=[1])  # type: ignore
    k: ts.String = ts.decimal(18, 4, default=0)  # type: ignore
    m: ts.Decimal = ts.decimal(18, 4, default="0")  # type: ignore
    conv_a: ts.Long = ts.field(converter=to_long, default="x")  # type: ignore
    conv_b: ts.Long = ts.field(converter=to_long, factory=lambda: ts.String("x"))  # type: ignore
    conv_c: ts.String = ts.field(converter=to_long, default=5)  # type: ignore
    conv_d: ts.String = ts.field(converter=to_long)  # type: ignore
    # Pyright copies `default="property"` into Props.__init__ as the parameter
    # default, and checks it against `String` when the class is used as a
    # `() -> Props` callable. Use `factory=lambda: Props()` instead.
    props_cls: Props = ts.field(factory=Props)  # pyright: ignore[reportCallIssue,reportArgumentType]
