from typing import TypeAlias, overload

import pyspark.sql.functions as F
from pyspark.sql.types import BooleanType, DataType

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


def broadcast[T: DataFrame](df: T) -> T:
    return df.broadcast()


def coalesce[T: DataType](*cols: Column[T]) -> Column[T]:
    return Column(F.coalesce(*[c.to_spark() for c in cols]))


def collect_list[T: DataType](col: Column[T]) -> Array[Column[T]]:
    return Array(F.collect_list(col.to_spark()))


def collect_set[T: DataType](col: Column[T]) -> Array[Column[T]]:
    return Array(F.collect_set(col.to_spark()))


def concat(*cols: String) -> String:
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


# TODO
# F.when
# F.least
# F.lit
# F.arrays_zip
# F.create_map
# F.current_date
# F.current_timestamp
# F.date_add
# F.date_diff
# F.date_format
# F.date_sub
# F.date_trunc
# F.dayofweek
# F.dense_rank
# F.desc
# F.explode
# F.explode_outer
# F.expr
# F.first
# F.floor
# F.from_json
# F.from_unixtime
# F.from_utc_timestamp
# F.greatest
# F.hash
# F.hour
# F.initcap
# F.lag
# F.last_day
# F.lead
# F.length
# F.lower
# F.lpad
# F.ltrim
# F.max
# F.max_by
# F.md5
# F.min
# F.monotonically_increasing_id
# F.month
# F.months_between
# F.posexplode
# F.pow
# F.radians
# F.rank
# F.regexp_extract
# F.regexp_replace
# F.replace
# F.row_number
# F.sin
# F.split
# F.split_part
# F.sqrt
# F.startswith
# F.struct
# F.substring
# F.sum
# F.to_json
# F.to_timestamp
# F.to_utc_timestamp
# F.trim
# F.udf
# F.unbase64
# F.unix_timestamp
# F.upper
# F.window
