import pyspark.sql.functions as F

import typespark
from typespark.columns.columns import TypedColumn

__all__ = []


def rank() -> typespark.Int:
    """Returns the rank of rows within a window partition, with gaps for ties.

    Wrapper for :func:`pyspark.sql.functions.rank`.
    """
    return typespark.Int.wrap(F.rank())


def dense_rank() -> typespark.Int:
    """Returns the rank of rows within a window partition, without gaps for ties.

    Wrapper for :func:`pyspark.sql.functions.dense_rank`.
    """
    return typespark.Int.wrap(F.dense_rank())


def row_number() -> typespark.Int:
    """Returns a sequential row number starting from 1 within a window partition.

    Wrapper for :func:`pyspark.sql.functions.row_number`.
    """
    return typespark.Int.wrap(F.row_number())


def lag[C: TypedColumn](
    col: C,
    offset: int = 1,
    default: C | None = None,
) -> C:
    """Returns the value from a row at a given offset prior to the current row within a window.

    Wrapper for :func:`pyspark.sql.functions.lag`.
    """
    if default is not None:
        return type(col).wrap(F.lag(col.to_spark(), offset, default.to_spark()))
    return type(col).wrap(F.lag(col.to_spark(), offset))


def lead[C: TypedColumn](
    col: C,
    offset: int = 1,
    default: C | None = None,
) -> C:
    """Returns the value from a row at a given offset after the current row within a window.

    Wrapper for :func:`pyspark.sql.functions.lead`.
    """
    if default is not None:
        return type(col).wrap(F.lead(col.to_spark(), offset, default.to_spark()))
    return type(col).wrap(F.lead(col.to_spark(), offset))


def ntile(n: int) -> typespark.Int:
    """Returns the bucket number (1 to `n`) of each row within an ordered window partition.

    Wrapper for :func:`pyspark.sql.functions.ntile`.
    """
    return typespark.Int.wrap(F.ntile(n))


def percent_rank() -> typespark.Double:
    """Returns the relative rank of rows within a window partition, from 0 to 1.

    Wrapper for :func:`pyspark.sql.functions.percent_rank`.
    """
    return typespark.Double.wrap(F.percent_rank())


def cume_dist() -> typespark.Double:
    """Returns the cumulative distribution of rows within a window partition.

    Wrapper for :func:`pyspark.sql.functions.cume_dist`.
    """
    return typespark.Double.wrap(F.cume_dist())


def nth_value[C: TypedColumn](col: C, offset: int, ignoreNulls: bool = False) -> C:
    """Returns the value of `col` at the `offset`-th row of the window frame (1-based).

    Wrapper for :func:`pyspark.sql.functions.nth_value`.
    """
    return type(col).wrap(F.nth_value(col.to_spark(), offset, ignoreNulls))
