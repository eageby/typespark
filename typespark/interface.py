from typing import Any, Optional, Protocol, Self, Union, runtime_checkable

from pyspark.sql import Column, types

from typespark.columns.columns import TypedColumn


@runtime_checkable
class SupportsETLFrame(Protocol):
    def select(self, *cols: Union[str, Column]) -> "SupportsETLFrame": ...

    def withColumn(self, colName: str, col: Column) -> "SupportsETLFrame": ...
    def drop(self, *cols: str) -> "SupportsETLFrame": ...
    def union(self, other: Self) -> Self: ...
    def join(
        self,
        other: Any,
        on: Optional[Union[str, list[str], Column]] = None,
        how: Optional[str] = None,
    ) -> "SupportsETLFrame": ...

    def filter(
        self, condition: Union[str, Column, TypedColumn[types.BooleanType]]
    ) -> "SupportsETLFrame": ...
    def distinct(self) -> "SupportsETLFrame": ...
    def show(
        self, n: int = 20, truncate: bool = True, vertical: bool = False
    ) -> None: ...
