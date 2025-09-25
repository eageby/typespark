from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Union

import attrs
from pyspark.sql import Column
from pyspark.sql.functions import col, explode, floor

if TYPE_CHECKING:
    from typespark.base import BaseDataFrame
    from typespark.columns import TypedColumn


LiteralType = Union[str, int, float, bool]


@attrs.define()
class DeferredColumn:
    parent: Generator
    col: Column


class Generator[T: TypedColumn](ABC):
    """When building a dataframe plan, top level projections are separated into two steps."""

    _parent: T
    _alias: str
    _elem_type: type[T]
    operations = []

    def __init__(self, parent: T, alias: str, elem_type: type[T]):
        self._parent = parent
        self._alias = alias
        self._col: Column | None = None
        self._elem_type = elem_type

    def __hash__(self) -> int:
        return hash(self._alias)

    def _materialize(self) -> Column:
        if self._col is not None:
            c = self._col
        c = col(self._alias)

        return self._elem_type.set_column(c, self._alias)

    # delegate all unknown attributes/methods to materialized Column
    def __getattr__(self, name: str):
        col = self._materialize()
        attr = getattr(col, name)

        # if callable(attr):

        #     def wrapper(*args, **kwargs):
        #         return attr(*args, **kwargs)

        #     return wrapper
        return DeferredColumn(self, attr)

    @abstractmethod
    def column_operation(self) -> Column: ...


class Exploded[T: TypedColumn](Generator):
    def column_operation(self):
        return explode(self._parent).alias(self._alias)
