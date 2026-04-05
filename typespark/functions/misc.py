from typing import overload

import pyspark.sql.functions as F
import pyspark.sql.types as types

import typespark as ts
from typespark.literals import LiteralType

from ._type_aliases import Numeric


def bitwise_not[T: Numeric](col: T) -> T:
    """Returns the bitwise NOT of `col`.

    Wrapper for :func:`pyspark.sql.functions.bitwise_not`.
    """
    return ts.Column(F.bitwise_not(col.to_spark()))  # type: ignore


def crc32(col: ts.Binary) -> ts.Long:
    """Returns the CRC-32 checksum of `col` as an unsigned integer.

    Wrapper for :func:`pyspark.sql.functions.crc32`.
    """
    return ts.Long(F.crc32(col.to_spark()))


def current_date() -> ts.Date:
    """Returns the current date at query start time.

    Wrapper for :func:`pyspark.sql.functions.current_date`.
    """
    return ts.Date(F.current_date())


def current_timestamp() -> ts.Timestamp:
    """Returns the current timestamp at query start time.

    Wrapper for :func:`pyspark.sql.functions.current_timestamp`.
    """
    return ts.Timestamp(F.current_timestamp())


def equal_null[T: types.DataType](col1: ts.Column[T], col2: ts.Column[T]) -> ts.Bool:
    """Returns true if `col1` equals `col2`, treating null as equal to null.

    Wrapper for :func:`pyspark.sql.functions.equal_null`.
    """
    return ts.Bool(F.equal_null(col1.to_spark(), col2.to_spark()))


@overload
def from_json[T: ts.Struct](
    col: ts.String,
    schema: type[T],
    options: dict[str, str] | None = None,
) -> T: ...


@overload
def from_json(
    col: ts.String,
    schema: types.StructType,
    options: dict[str, str] | None = None,
) -> ts.Column: ...


def from_json[T: ts.Struct](
    col: ts.String, schema: type[T] | types.StructType, options: dict[str, str] | None = None
) -> ts.Column | T:
    """Parses a JSON string column into a struct.

    Pass a TypeSpark Struct subclass for a typed result, or a PySpark StructType
    for an untyped Column.

    Wrapper for :func:`pyspark.sql.functions.from_json`.
    """
    if isinstance(schema, type) and issubclass(schema, ts.Struct):
        return schema.from_json(col, options)
    return ts.Column(F.from_json(col.to_spark(), schema, options))


def greatest[T: ts.Column](*cols: T) -> T:
    """Returns the largest value among the given columns, skipping nulls.

    Wrapper for :func:`pyspark.sql.functions.greatest`.
    """
    return ts.Column(F.greatest(*[c.to_spark() for c in cols]))  # type: ignore


def hash(col: ts.Column) -> ts.Int:
    """Returns a hash value for `col` using the default hash algorithm.

    Wrapper for :func:`pyspark.sql.functions.hash`.
    """
    return ts.Int(F.hash(col.to_spark()))


def isnan(col: Numeric) -> ts.Bool:
    """Returns true if `col` is NaN.

    Wrapper for :func:`pyspark.sql.functions.isnan`.
    """
    return ts.Bool(F.isnan(col.to_spark()))


def isnull(col: ts.Column) -> ts.Bool:
    """Returns true if `col` is null.

    Wrapper for :func:`pyspark.sql.functions.isnull`.
    """
    return ts.Bool(F.isnull(col.to_spark()))


def isnotnull(col: ts.Column) -> ts.Bool:
    """Returns true if `col` is not null.

    Wrapper for :func:`pyspark.sql.functions.isnotnull`.
    """
    return ts.Bool(F.isnotnull(col.to_spark()))


def least[T: ts.Column](*cols: T) -> T:
    """Returns the smallest value among the given columns, skipping nulls.

    Wrapper for :func:`pyspark.sql.functions.least`.
    """
    return ts.Column(F.least(*[c.to_spark() for c in cols]))  # type: ignore


@overload
def lit(value: str) -> ts.String: ...


@overload
def lit(value: bool) -> ts.Bool: ...  # type: ignore[overload-overlap]


@overload
def lit(value: int) -> ts.Long: ...


@overload
def lit(value: float) -> ts.Double: ...


def lit(value: LiteralType) -> ts.Column:
    """Creates a column from a Python literal value.

    Wrapper for :func:`pyspark.sql.functions.lit`.
    """
    return ts.Column(F.lit(value))


def md5(col: ts.Column) -> ts.String:
    """Returns the MD5 hex digest of `col`.

    Wrapper for :func:`pyspark.sql.functions.md5`.
    """
    return ts.String(F.md5(col.to_spark()))


def monotonically_increasing_id() -> ts.Long:
    """Returns a monotonically increasing 64-bit integer that is unique per row, but not consecutive.

    Wrapper for :func:`pyspark.sql.functions.monotonically_increasing_id`.
    """
    return ts.Long(F.monotonically_increasing_id())


def nanvl[T: Numeric](col1: T, col2: T) -> T:
    """Returns `col1` if it is not NaN, otherwise returns `col2`.

    Wrapper for :func:`pyspark.sql.functions.nanvl`.
    """
    return ts.Column(F.nanvl(col1.to_spark(), col2.to_spark()))  # type: ignore


def nullif[T: types.DataType](col1: ts.Column[T], col2: ts.Column[T]) -> ts.Column[T]:
    """Returns null if `col1` equals `col2`, otherwise returns `col1`.

    Wrapper for :func:`pyspark.sql.functions.nullif`.
    """
    return ts.Column(F.nullif(col1.to_spark(), col2.to_spark()))


def nvl[T: types.DataType](col1: ts.Column[T], col2: ts.Column[T]) -> ts.Column[T]:
    """Returns `col1` if it is not null, otherwise returns `col2`.

    Wrapper for :func:`pyspark.sql.functions.nvl`.
    """
    return ts.Column(F.nvl(col1.to_spark(), col2.to_spark()))


def nvl2[T: types.DataType](col1: ts.Column, col2: ts.Column[T], col3: ts.Column[T]) -> ts.Column[T]:
    """Returns `col2` if `col1` is not null, otherwise returns `col3`.

    Wrapper for :func:`pyspark.sql.functions.nvl2`.
    """
    return ts.Column(F.nvl2(col1.to_spark(), col2.to_spark(), col3.to_spark()))


def spark_partition_id() -> ts.Int:
    """Returns the partition ID of the current row.

    Wrapper for :func:`pyspark.sql.functions.spark_partition_id`.
    """
    return ts.Int(F.spark_partition_id())


def typeof(col: ts.Column) -> ts.String:
    """Returns the Spark SQL data type name of `col` as a string.

    Wrapper for :func:`pyspark.sql.functions.typeof`.
    """
    return ts.String(F.typeof(col.to_spark()))


def base64(col: ts.Binary) -> ts.String:
    """Encodes binary column `col` as a Base64 string.

    Wrapper for :func:`pyspark.sql.functions.base64`.
    """
    return ts.String(F.base64(col.to_spark()))


def version() -> ts.String:
    """Returns the Spark version as a string.

    Wrapper for :func:`pyspark.sql.functions.version`.
    """
    return ts.String(F.version())


class WhenStatement[T: types.DataType](ts.TypedColumn[T]):
    """Represents a chained `CASE WHEN` expression. Use `.when()` to add branches and `.otherwise()` for a default."""

    def when(self, condition: ts.Bool, value: ts.TypedColumn[T]) -> "WhenStatement[T]":
        """Adds another `WHEN condition THEN value` branch to the expression.

        Wrapper for :meth:`pyspark.sql.Column.when`.
        """
        return WhenStatement[T](self.to_spark().when(condition.to_spark(), value.to_spark()))

    def otherwise(self, value: ts.TypedColumn[T]) -> ts.TypedColumn[T]:
        """Adds a final `ELSE value` clause to the expression.

        Wrapper for :meth:`pyspark.sql.Column.otherwise`.
        """
        return ts.TypedColumn(self.to_spark().otherwise(value.to_spark()))


def when[T: types.DataType](condition: ts.Bool, value: ts.TypedColumn[T]) -> WhenStatement[T]:
    """Starts a `CASE WHEN condition THEN value` expression. Chain `.when()` and `.otherwise()` to complete it.

    Wrapper for :func:`pyspark.sql.functions.when`.
    """
    return WhenStatement[T](F.when(condition.to_spark(), value.to_spark()))


def xxhash64(*cols: ts.Column) -> ts.Long:
    """Returns a 64-bit hash of the given columns using the xxHash algorithm.

    Wrapper for :func:`pyspark.sql.functions.xxhash64`.
    """
    return ts.Long(F.xxhash64(*[c.to_spark() for c in cols]))
