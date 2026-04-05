import pyspark.sql.functions as F
from pyspark.sql.types import DataType

import typespark
from typespark.columns.columns import TypedColumn

__all__ = []


def rank() -> typespark.Int:
    """Returns the rank of rows within a window partition, with gaps for ties.

    Wrapper for :func:`pyspark.sql.functions.rank`.
    """
    return typespark.Int(F.rank())


def dense_rank() -> typespark.Int:
    """Returns the rank of rows within a window partition, without gaps for ties.

    Wrapper for :func:`pyspark.sql.functions.dense_rank`.
    """
    return typespark.Int(F.dense_rank())


def row_number() -> typespark.Int:
    """Returns a sequential row number starting from 1 within a window partition.

    Wrapper for :func:`pyspark.sql.functions.row_number`.
    """
    return typespark.Int(F.row_number())


def lag[T: DataType](
    col: TypedColumn[T],
    offset: int = 1,
    default: TypedColumn[T] | None = None,
) -> TypedColumn[T]:
    """Returns the value from a row at a given offset prior to the current row within a window.

    Wrapper for :func:`pyspark.sql.functions.lag`.
    """
    if default is not None:
        return TypedColumn(F.lag(col.to_spark(), offset, default.to_spark()))
    return TypedColumn(F.lag(col.to_spark(), offset))


def lead[T: DataType](
    col: TypedColumn[T],
    offset: int = 1,
    default: TypedColumn[T] | None = None,
) -> TypedColumn[T]:
    """Returns the value from a row at a given offset after the current row within a window.

    Wrapper for :func:`pyspark.sql.functions.lead`.
    """
    if default is not None:
        return TypedColumn(F.lead(col.to_spark(), offset, default.to_spark()))
    return TypedColumn(F.lead(col.to_spark(), offset))
