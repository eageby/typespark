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

# What an operator on a deferred column accepts on the right-hand side.
OperandType = Union["DeferredColumn", Column, "TypedColumn[Any]", LiteralType]


def _operand(other: OperandType) -> Column | LiteralType:
    """Reduce an operand to a Spark column (or literal) usable in an expression."""
    from typespark.columns.columns import TypedColumn as _TypedColumn

    if isinstance(other, (DeferredColumn, _TypedColumn)):
        return other.to_spark()
    return other


@attrs.define(eq=False)
class DeferredColumn:
    parent: "Generator[Any]"
    col: "Column | TypedColumn[Any]"

    def alias(self, alias: str) -> "DeferredColumn":
        self.col = self.to_spark().alias(alias)
        return self

    def cast(self, data_type: DataType) -> "DeferredColumn":
        self.col = self.to_spark().cast(data_type)
        return self

    def to_spark(self) -> Column:
        """The wrapped column as a plain Spark column."""
        from typespark.columns.columns import TypedColumn as _TypedColumn

        return self.col.to_spark() if isinstance(self.col, _TypedColumn) else self.col

    def _defer(self, col: "Column | TypedColumn[Any]") -> "DeferredColumn":
        """Keep an operation deferred so the generator still materializes first."""
        return DeferredColumn(self.parent, col)

    # delegate unknown attributes/methods to the wrapped column, staying deferred
    def __getattr__(self, name: str) -> Any:
        from typespark.columns.columns import TypedColumn as _TypedColumn

        if name.startswith("_") or name in ("col", "parent"):
            raise AttributeError(name)

        attr = getattr(self.col, name)
        if isinstance(attr, (Column, _TypedColumn)):
            return self._defer(attr)
        if callable(attr):

            def wrapped(*args: Any, **kwargs: Any) -> Any:
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
    def __eq__(self, other: OperandType) -> "DeferredColumn":  # type: ignore[override]
        return self._defer(self.to_spark() == _operand(other))

    def __ne__(self, other: OperandType) -> "DeferredColumn":  # type: ignore[override]
        return self._defer(self.to_spark() != _operand(other))

    def __lt__(self, other: OperandType) -> "DeferredColumn":
        return self._defer(self.to_spark() < _operand(other))

    def __le__(self, other: OperandType) -> "DeferredColumn":
        return self._defer(self.to_spark() <= _operand(other))

    def __gt__(self, other: OperandType) -> "DeferredColumn":
        return self._defer(self.to_spark() > _operand(other))

    def __ge__(self, other: OperandType) -> "DeferredColumn":
        return self._defer(self.to_spark() >= _operand(other))

    def __add__(self, other: OperandType) -> "DeferredColumn":
        return self._defer(self.to_spark() + _operand(other))

    def __radd__(self, other: OperandType) -> "DeferredColumn":
        return self._defer(self.to_spark().__radd__(_operand(other)))

    def __sub__(self, other: OperandType) -> "DeferredColumn":
        return self._defer(self.to_spark() - _operand(other))

    def __rsub__(self, other: OperandType) -> "DeferredColumn":
        return self._defer(self.to_spark().__rsub__(_operand(other)))

    def __mul__(self, other: OperandType) -> "DeferredColumn":
        return self._defer(self.to_spark() * _operand(other))

    def __rmul__(self, other: OperandType) -> "DeferredColumn":
        return self._defer(self.to_spark().__rmul__(_operand(other)))

    def __truediv__(self, other: OperandType) -> "DeferredColumn":
        return self._defer(self.to_spark() / _operand(other))

    def __rtruediv__(self, other: OperandType) -> "DeferredColumn":
        return self._defer(self.to_spark().__rtruediv__(_operand(other)))

    def __mod__(self, other: OperandType) -> "DeferredColumn":
        return self._defer(self.to_spark() % _operand(other))

    def __pow__(self, other: OperandType) -> "DeferredColumn":
        return self._defer(self.to_spark() ** _operand(other))

    def __neg__(self) -> "DeferredColumn":
        return self._defer(-self.to_spark())

    def __and__(self, other: OperandType) -> "DeferredColumn":
        return self._defer(self.to_spark() & _operand(other))

    def __or__(self, other: OperandType) -> "DeferredColumn":
        return self._defer(self.to_spark() | _operand(other))

    def __invert__(self) -> "DeferredColumn":
        return self._defer(~self.to_spark())


class Generator[T: TypedColumn[Any]](ABC):
    """When building a dataframe plan, top level projections are separated into two steps."""

    _parent: T
    _alias: str
    _elem_type: type[T]

    def __init__(self, parent: T, alias: str, elem_type: type[T]) -> None:
        self._parent = parent
        self._alias = alias
        self._elem_type = elem_type

    def __hash__(self) -> int:
        return hash(self._alias)

    def _materialize(self) -> T:
        """The generator's output, referenced by alias — it exists only after step 1."""
        return self._elem_type._set_column(col(self._alias), self._alias)

    # delegate all unknown attributes/methods to materialized Column
    def __getattr__(self, name: str) -> "DeferredColumn | Any":
        from typespark.columns.columns import TypedColumn as _TypedColumn

        materialized = self._materialize()
        attr = getattr(materialized, name)
        if isinstance(attr, (Column, _TypedColumn)):
            return DeferredColumn(self, attr)
        return attr

    @abstractmethod
    def column_operation(self) -> Column: ...


class Exploded[T: TypedColumn[Any]](Generator[T]):
    def column_operation(self) -> Column:
        return explode(self._parent._col).alias(self._alias)
