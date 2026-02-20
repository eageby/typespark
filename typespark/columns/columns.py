import functools
from typing import Optional, Self

import pyspark.sql
from pyspark.sql.types import BooleanType, DataType

from .groups import _GroupColumn


class TypedColumn[T: DataType]:
    _col: pyspark.sql.Column
    _name: str

    def __hash__(self) -> int:
        return hash(self._name)

    @functools.wraps(pyspark.sql.Column.alias)
    def name(self, alias: str, **kwargs): ...

    def __init__(self, c: pyspark.sql.Column | Self):  # pylint: disable=super-init-not-called
        self._col = c.to_spark() if isinstance(c, TypedColumn) else c

    def _set_name(self, name: str):
        self._name = name
        return self

    def to_spark(self):
        return self._col

    @classmethod
    def set_column(
        cls,
        col: pyspark.sql.Column,
        name: str,
        tp: Optional[type["TypedColumn"]] = None,
    ):
        self = object.__new__(cls)
        self._col = col
        self._name = name
        return self

    def __getattr__(self, item) -> pyspark.sql.Column:
        return getattr(self._col, item)

    def __repr__(self):
        return f"{self.__class__.__name__}<'{self._name}'>"

    def group(self) -> Self:
        return _GroupColumn(self)  # type: ignore

    def cast[U: DataType](self, dataType: U) -> "TypedColumn[U]":
        return TypedColumn(self._col.cast(dataType))._set_name(self._name)

    def alias(self, alias: str, **kwargs) -> "Self":
        return AliasedTypedColumn(self._col.alias(alias, **kwargs), alias, self._name)  # type: ignore

    def isNull(self):
        return TypedColumn[BooleanType](self._col.isNull())

    def isNotNull(self):
        return TypedColumn[BooleanType](self._col.isNotNull())

    def like(self, pattern: str):
        return TypedColumn[BooleanType](self._col.like(pattern))

    def _get_operand_column(self, other: "TypedColumn"):
        # Utility function to decide if 'other' is a ColumnWrapper or a raw value.
        if isinstance(other, TypedColumn):
            return other._col
        elif isinstance(other, pyspark.sql.Column):
            return other
        else:
            return other

    # Arithmetic operators
    def __add__(self, other: "TypedColumn"):
        return TypedColumn(self._col.__add__(self._get_operand_column(other)))

    def __sub__(self, other: "TypedColumn"):
        return TypedColumn(self._col.__sub__(self._get_operand_column(other)))

    def __mul__(self, other: "TypedColumn"):
        return TypedColumn(self._col.__mul__(self._get_operand_column(other)))

    def __truediv__(self, other: "TypedColumn"):
        return TypedColumn(self._col.__truediv__(self._get_operand_column(other)))

    def __mod__(self, other: "TypedColumn"):
        return TypedColumn(self._col.__mod__(self._get_operand_column(other)))

    def __pow__(self, other: "TypedColumn"):
        return TypedColumn(self._col.__pow__(self._get_operand_column(other)))

    # Comparison operators
    def __eq__(  # type: ignore[override]
        self, other: "TypedColumn"
    ) -> "TypedColumn[BooleanType]":
        return TypedColumn[BooleanType](
            self._col.__eq__(self._get_operand_column(other))
        )

    def __ne__(  # type: ignore[override]
        self, other: "TypedColumn"
    ) -> "TypedColumn":
        return TypedColumn[BooleanType](
            self._col.__ne__(self._get_operand_column(other))
        )

    def __lt__(self, other: "TypedColumn"):
        return TypedColumn[BooleanType](
            self._col.__lt__(self._get_operand_column(other))
        )

    def __le__(self, other: "TypedColumn"):
        return TypedColumn[BooleanType](
            self._col.__le__(self._get_operand_column(other))
        )

    def __gt__(self, other: "TypedColumn"):
        return TypedColumn[BooleanType](
            self._col.__gt__(self._get_operand_column(other))
        )

    def __ge__(self, other: "TypedColumn"):
        return TypedColumn[BooleanType](
            self._col.__ge__(self._get_operand_column(other))
        )

    # Logical operators
    def __and__(self, other: "TypedColumn"):
        return TypedColumn[BooleanType](
            self._col.__and__(self._get_operand_column(other))
        )

    def __or__(self, other: "TypedColumn"):
        return TypedColumn[BooleanType](
            self._col.__or__(self._get_operand_column(other))
        )

    def __invert__(self):
        return TypedColumn[BooleanType](self._col.__invert__())

    # Unary negation
    def __neg__(self):
        return TypedColumn[T](self._col.__neg__())

    def desc(self):
        return TypedColumn[T](self.column.desc())

    def asc(self):
        return TypedColumn[T](self.column.asc())


class AliasedTypedColumn(TypedColumn):
    """When creating projected query plans we need to keep track of what has been aliased."""

    original_name: str

    def __init__(self, c: pyspark.sql.Column, name: str, original_name: str):
        super().__init__(c)
        self._name = name
        self.original_name = original_name
