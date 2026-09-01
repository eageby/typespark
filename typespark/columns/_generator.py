from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Union

import attrs
from pyspark.sql import Column
from pyspark.sql.functions import col, explode
from pyspark.sql.types import DataType

if TYPE_CHECKING:
    from typespark.columns import TypedColumn


LiteralType = Union[str, int, float, bool]


def _operand(other):
    """Unwrap a DeferredColumn operand so operators see the underlying column."""
    return other.col if isinstance(other, DeferredColumn) else other


@attrs.define(eq=False)
class DeferredColumn:
    parent: "Generator"
    col: "Column | TypedColumn"

    def alias(self, alias: str):
        self.col = self.col.alias(alias)
        return self

    def cast(self, data_type: DataType):
        self.col = self._c.cast(data_type)
        return self

    @property
    def _c(self) -> Any:
        """The wrapped column, untyped — Column and TypedColumn share the API."""
        return self.col

    def _defer(self, col) -> "DeferredColumn":
        """Keep an operation deferred so the generator still materializes first."""
        return DeferredColumn(self.parent, col)

    # delegate unknown attributes/methods to the wrapped column, staying deferred
    def __getattr__(self, name: str):
        from typespark.columns.columns import TypedColumn as _TypedColumn

        if name.startswith("_") or name in ("col", "parent"):
            raise AttributeError(name)

        attr = getattr(self.col, name)
        if isinstance(attr, (Column, _TypedColumn)):
            return self._defer(attr)
        if callable(attr):

            def wrapped(*args, **kwargs):
                result = attr(
                    *[_operand(a) for a in args],
                    **{k: _operand(v) for k, v in kwargs.items()},
                )
                if isinstance(result, (Column, _TypedColumn)):
                    return self._defer(result)
                return result

            return wrapped
        return attr

    # Operators — mirror TypedColumn's set, wrapping results as deferred.
    def __eq__(self, other) -> "DeferredColumn":  # type: ignore[override]
        return self._defer(self.col == _operand(other))

    def __ne__(self, other) -> "DeferredColumn":  # type: ignore[override]
        return self._defer(self.col != _operand(other))

    def __lt__(self, other) -> "DeferredColumn":
        return self._defer(self.col < _operand(other))

    def __le__(self, other) -> "DeferredColumn":
        return self._defer(self.col <= _operand(other))

    def __gt__(self, other) -> "DeferredColumn":
        return self._defer(self.col > _operand(other))

    def __ge__(self, other) -> "DeferredColumn":
        return self._defer(self.col >= _operand(other))

    def __add__(self, other) -> "DeferredColumn":
        return self._defer(self.col + _operand(other))

    def __radd__(self, other) -> "DeferredColumn":
        return self._defer(_operand(other) + self.col)

    def __sub__(self, other) -> "DeferredColumn":
        return self._defer(self.col - _operand(other))

    def __rsub__(self, other) -> "DeferredColumn":
        return self._defer(_operand(other) - self.col)

    def __mul__(self, other) -> "DeferredColumn":
        return self._defer(self.col * _operand(other))

    def __rmul__(self, other) -> "DeferredColumn":
        return self._defer(_operand(other) * self.col)

    def __truediv__(self, other) -> "DeferredColumn":
        return self._defer(self.col / _operand(other))

    def __rtruediv__(self, other) -> "DeferredColumn":
        return self._defer(_operand(other) / self.col)

    def __mod__(self, other) -> "DeferredColumn":
        return self._defer(self.col % _operand(other))

    def __pow__(self, other) -> "DeferredColumn":
        return self._defer(self.col ** _operand(other))

    def __neg__(self) -> "DeferredColumn":
        return self._defer(-self.col)

    def __and__(self, other) -> "DeferredColumn":
        return self._defer(self.col & _operand(other))

    def __or__(self, other) -> "DeferredColumn":
        return self._defer(self.col | _operand(other))

    def __invert__(self) -> "DeferredColumn":
        return self._defer(~self._c)


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

    def _materialize(self) -> T:
        if self._col is not None:
            c = self._col
        c = col(self._alias)

        return self._elem_type._set_column(c, self._alias)

    # delegate all unknown attributes/methods to materialized Column
    def __getattr__(self, name: str):
        from typespark.columns.columns import TypedColumn as _TypedColumn

        materialized = self._materialize()
        attr = getattr(materialized, name)
        if isinstance(attr, (Column, _TypedColumn)):
            return DeferredColumn(self, attr)
        return attr

    @abstractmethod
    def column_operation(self) -> Column: ...


class Exploded[T: TypedColumn](Generator[T]):
    def column_operation(self):
        return explode(self._parent._col).alias(self._alias)
