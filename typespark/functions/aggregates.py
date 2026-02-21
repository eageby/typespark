from pyspark.sql import functions as F
from pyspark.sql.types import DataType

from typespark.columns import TypedColumn
from typespark.columns.groups import _AggregateColumn
from typespark.type_alias import Integer


def sum[T: DataType](col: TypedColumn[T]) -> TypedColumn[T]:
    return _AggregateColumn(TypedColumn(F.sum(col.to_spark())))  # type: ignore


def first[T: DataType](col: TypedColumn[T]) -> TypedColumn[T]:
    return _AggregateColumn(TypedColumn(F.first(col.to_spark())))  # type: ignore


def max[T: DataType](col: TypedColumn[T]) -> TypedColumn[T]:
    return _AggregateColumn(TypedColumn(F.max(col.to_spark())))  # type: ignore


def max_by[T: DataType](col: TypedColumn[T], ord: TypedColumn) -> TypedColumn[T]:
    return _AggregateColumn(TypedColumn(F.max_by(col.to_spark(), ord.to_spark())))  # type: ignore


def min[T: DataType](col: TypedColumn[T]) -> TypedColumn[T]:
    return _AggregateColumn(TypedColumn(F.min(col.to_spark())))  # type: ignore


def count(col: TypedColumn) -> Integer:
    return _AggregateColumn(TypedColumn(F.count(col.to_spark())))  # type: ignore
