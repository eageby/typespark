from pyspark.sql import functions as F
from pyspark.sql.types import DataType

from typespark.columns import TypedColumn
from typespark.groups import _AggregateColumn
from typespark.type_alias import Integer


def sum[T: DataType](col: TypedColumn[T]) -> TypedColumn[T]:
    return _AggregateColumn(TypedColumn(F.sum(col)))  # type: ignore


def count(col: TypedColumn) -> Integer:
    return _AggregateColumn(TypedColumn(F.count(col)))  # type: ignore
