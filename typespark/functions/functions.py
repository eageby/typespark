from typing import Literal, Optional, TypeAlias, Union, overload

import pyspark.sql.functions as F
from pyspark.sql.types import BooleanType, DataType, StringType, StructType

from typespark import (
    Array,
    Binary,
    Bool,
    Byte,
    Column,
    DataFrame,
    Date,
    Decimal,
    Double,
    Float,
    Int,
    Integer,
    Long,
    Short,
    String,
    Timestamp,
)
from typespark.columns.columns import TypedColumn
from typespark.columns.struct import Struct
from typespark.literals import LiteralType

Numeric: TypeAlias = Byte | Short | Int | Long | Float | Double | Decimal


def add_months(start: Date, months: Integer | int) -> Date:
    """Returns the date that is `months` months after `start`.

    Wrapper for `F.add_months` (:func:`pyspark.sql.functions.add_months`).

    See the original function for full documentation.
    """
    return Date(
        F.add_months(
            start.to_spark(),
            months.to_spark() if isinstance(months, Column) else months,
        )
    )


def array[T: DataType](*cols: Column[T]) -> Array[Column[T]]:
    return Array(F.array([c.to_spark() for c in cols]))


def array_append[T: DataType](
    col: Array[Column[T]], value: Column[T]
) -> Array[Column[T]]:
    return Array(F.array_append(col.to_spark(), value.to_spark()))


def array_contains[T: DataType](
    col: Array[Column[T]], value: Column[T]
) -> Column[BooleanType]:
    return Column(F.array_contains(col.to_spark(), value.to_spark()))


def atan2(col1: Numeric, col2: Numeric) -> Double:
    return Double(F.atan2(col1.to_spark(), col2.to_spark()))


def cos(col: Numeric) -> Double:
    return Double(F.cos(col.to_spark()))


def sin(col: Numeric) -> Double:
    return Double(F.sin(col.to_spark()))


def broadcast[T: DataFrame](df: T) -> T:
    return df.broadcast()


def coalesce[T: DataType](*cols: Column[T]) -> Column[T]:
    return Column(F.coalesce(*[c.to_spark() for c in cols]))


def collect_list[T: DataType](col: Column[T]) -> Array[Column[T]]:
    return Array(F.collect_list(col.to_spark()))


def collect_set[T: DataType](col: Column[T]) -> Array[Column[T]]:
    return Array(F.collect_set(col.to_spark()))


@overload
def concat(*cols: String) -> String: ...
@overload
def concat[T: Column](*cols: Array[T]) -> Array[T]: ...


def concat(*cols: String | Array) -> String | Array:
    if isinstance(cols[0], Array):
        elem_type = cols[0]._elem_type
        return Array(F.concat(*[c.to_spark() for c in cols]), elem_type)

    return String(F.concat(*[c.to_spark() for c in cols]))


def concat_ws(sep: str, *cols: String) -> String:
    return String(F.concat_ws(sep, *[c.to_spark() for c in cols]))


@overload
def contains(left: String, right: String) -> Bool: ...


@overload
def contains(left: Binary, right: Binary) -> Bool: ...


def contains(left: String | Binary, right: String | Binary) -> Bool:
    return Bool(F.contains(left.to_spark(), right.to_spark()))


def to_date(col: String, format: str | None = None) -> Date:
    return Date(F.to_date(col.to_spark(), format))


def year(col: Date | Timestamp) -> Int:
    return Int(F.year(col.to_spark()))


def month(col: Date | Timestamp) -> Int:
    return Int(F.month(col.to_spark()))


def least[T: Column](*cols: T) -> T:
    return Column(F.least(*[c.to_spark() for c in cols]))  # type: ignore


def lower(col: String) -> String:
    return String(F.lower(col.to_spark()))


def ltrim(col: String, trim: String | None = None) -> String:
    return String(F.ltrim(col.to_spark(), trim.to_spark() if trim else None))


def greatest[T: Column](*cols: T) -> T:
    return Column(F.greatest(*[c.to_spark() for c in cols]))  # type: ignore


def current_date() -> Date:
    return Date(F.current_date())


def current_timestamp() -> Timestamp:
    return Timestamp(F.current_timestamp())


def explode[T: DataType](col: Array[Column[T]]) -> Column[T]:
    return Column[T](F.explode(col.to_spark()))


def hash(col: Column) -> Int:
    return Int(F.hash(col.to_spark()))


def md5(col: Column) -> String:
    return String(F.md5(col.to_spark()))


def date_add(start: Date, days: Int) -> Date:
    return Date(F.date_add(start.to_spark(), days.to_spark()))


# needs better typing
def floor(col: Numeric, scale: Optional[Int | int] = None) -> Column:
    return Column(
        F.floor(
            col.to_spark(), scale.to_spark() if isinstance(scale, Column) else scale
        )
    )


class WhenStatement[T: DataType](Column[T]):
    def when(self, condition: Bool, value: Column[T]):
        return WhenStatement[T](
            self.to_spark().when(condition.to_spark(), value.to_spark())
        )

    def otherwise(self, value: Column[T]) -> Column[T]:
        return Column[T](self.to_spark().otherwise(value.to_spark()))


def when[T: DataType](condition: Bool, value: Column[T]) -> WhenStatement[T]:
    return WhenStatement[T](F.when(condition.to_spark(), value.to_spark()))


def from_json[T: Struct](
    col: Column,
    schema: StructType,
    options: Optional[dict[str, str]] = None,
) -> Column:
    return Column(F.from_json(col.to_spark(), schema, options))


def lit(value: LiteralType) -> Column:
    return Column(F.lit(value))


def upper(col: String) -> String:
    return String(F.upper(col.to_spark()))


def lpad(
    col: String,
    len: Union[Integer, int],
    pad: Union[String, str],
) -> String:
    return String(
        F.lpad(
            col.to_spark(),
            len.to_spark() if isinstance(len, TypedColumn) else len,
            pad.to_spark() if isinstance(pad, TypedColumn) else pad,
        )
    )


def trim(col: String, trim: Optional[String] = None):
    return String(F.trim(col.to_spark(), trim.to_spark() if trim else None))


def split(
    col: String,
    pattern: Union[Column, str],
    limit: Union[Int, int] = -1,
):
    return Array(
        F.split(
            col.to_spark(),
            pattern.to_spark() if isinstance(pattern, Column) else pattern,
            limit.to_spark() if isinstance(limit, Column) else limit,
        ),
        String,
    )


def substring(
    str: String,
    pos: Union[Int, int],
    len: Union[Int, int],
) -> String:
    return String(
        F.substring(
            str.to_spark(),
            pos.to_spark() if isinstance(pos, TypedColumn) else pos,
            len.to_spark() if isinstance(len, TypedColumn) else len,
        )
    )


def startswith(str: String, prefix: String) -> Bool:
    return Bool(F.startswith(str.to_spark(), prefix.to_spark()))


def regexp_replace(
    string: String, pattern: Union[str, String], replacement: Union[str, String]
) -> String:
    return String(
        F.regexp_replace(
            string.to_spark(),
            pattern.to_spark() if isinstance(pattern, Column) else pattern,
            replacement.to_spark() if isinstance(replacement, Column) else replacement,
        )
    )


def timestamp_diff(
    unit: Literal[
        "YEAR",
        "QUARTER",
        "MONTH",
        "WEEK",
        "DAY",
        "HOUR",
        "MINUTE",
        "SECOND",
        "MILLISECOND",
        "MICROSECOND",
    ],
    start: Timestamp,
    end: Timestamp,
) -> Int:
    return Int(F.timestamp_diff(unit, start.to_spark(), end.to_spark()))
