from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql.types import DataType

if TYPE_CHECKING:
    from typespark.columns import TypedColumn


class _GroupBase:
    column: TypedColumn
    _alias: str

    def __init__(self, column: TypedColumn) -> None:
        self.column = column

    def alias(self, alias: str):
        self.column = self.column.alias(alias)
        return self

    def cast(self, type: DataType):
        self.column = self.column.cast(type)
        return self


class _GroupColumn(_GroupBase):
    pass


class _AggregateColumn(_GroupBase):
    pass
