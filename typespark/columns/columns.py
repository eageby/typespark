import functools
from typing import TYPE_CHECKING, Any, ClassVar, Self, overload

__all__ = ["TypedColumn", "AliasedTypedColumn", "Bool"]

import pyspark.sql
import pyspark.sql.functions
from pyspark.sql.types import BooleanType, DataType

from ._groups import _GroupColumn

if TYPE_CHECKING:
    from pyspark.sql.types import (
        BinaryType,
        DateType,
        DecimalType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        ShortType,
        StringType,
        TimestampType,
    )
    from typespark.columns.primitives import (
        Binary,
        Date,
        Decimal,
        Double,
        Float,
        Integer,
        Long,
        Short,
        String,
        Timestamp,
    )

_concrete_registry: dict[type[DataType], "type[TypedColumn]"] = {}


class TypedColumn[T: DataType]:
    _col: pyspark.sql.Column
    _name: str | None = None
    _data_type: ClassVar[type[DataType] | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls._data_type is not None:
            _concrete_registry[cls._data_type] = cls

    @classmethod
    def _cast_schema(cls) -> DataType | None:
        return cls._data_type() if cls._data_type is not None else None

    def __hash__(self) -> int:
        return hash(self._name)

    def __init__(self, c: "pyspark.sql.Column | TypedColumn | Self"):  # pylint: disable=super-init-not-called
        col = c.to_spark() if isinstance(c, TypedColumn) else c
        schema = self.__class__._cast_schema()
        self._col = col.cast(schema) if schema is not None else col
        if isinstance(c, TypedColumn):
            self._name = c._name

    @classmethod
    def wrap(cls, c: pyspark.sql.Column) -> "Self":
        """Wrap a column without casting."""
        self = object.__new__(cls)
        object.__setattr__(self, "_col", c)  # Circumventing frozen
        object.__setattr__(self, "_name", None)  # Circumventing frozen
        return self

    def _set_name(self, name: str | None):
        self._name = name
        return self

    def to_spark(self) -> pyspark.sql.Column:
        return self._col

    @classmethod
    def _set_column(cls, col: pyspark.sql.Column, name: str) -> "Self":
        self = object.__new__(cls)
        self._col = col
        self._name = name
        return self

    def __repr__(self):
        return (
            f"{self.__class__.__name__}<'{self._name}'>"
            if self._name is not None
            else f"{self.__class__.__name__}<>"
        )

    def group(self) -> Self:
        """Mark column as a group key for use in an aggregate DataFrame constructor.

        Example::

            class Agg(ts.DataFrame):
                name: ts.String
                total: ts.Int

            Agg(df, name=df.name.group(), total=ts.sum(df.age))

        The returned value is a ``_GroupColumn`` sentinel. Arithmetic and comparison
        operations on it raise ``TypeError`` — operate on the field after construction
        instead.
        """
        return _GroupColumn(self)  # type: ignore

    if TYPE_CHECKING:

        @overload
        def cast(self, dataType: "BooleanType") -> "Bool": ...
        @overload
        def cast(self, dataType: "BinaryType") -> "Binary": ...
        @overload
        def cast(self, dataType: "DateType") -> "Date": ...
        @overload
        def cast(self, dataType: "DecimalType") -> "Decimal": ...
        @overload
        def cast(self, dataType: "DoubleType") -> "Double": ...
        @overload
        def cast(self, dataType: "FloatType") -> "Float": ...
        @overload
        def cast(self, dataType: "IntegerType") -> "Integer": ...
        @overload
        def cast(self, dataType: "LongType") -> "Long": ...
        @overload
        def cast(self, dataType: "ShortType") -> "Short": ...
        @overload
        def cast(self, dataType: "StringType") -> "String": ...
        @overload
        def cast(self, dataType: "TimestampType") -> "Timestamp": ...

    def cast[U: DataType](self, dataType: U) -> "TypedColumn[U]":
        target = _concrete_registry.get(type(dataType), TypedColumn)
        return target.wrap(self._col.cast(dataType))._set_name(self._name)  # type: ignore[return-value]

    def alias(self, alias: str, **kwargs) -> "Self":
        return AliasedTypedColumn(self._col.alias(alias, **kwargs), alias, self._name)  # type: ignore

    @functools.wraps(alias)
    def name(self, alias: str, **kwargs) -> "Self":
        return self.alias(alias)

    def isNull(self) -> "Bool":
        return Bool.wrap(self._col.isNull())

    def isNotNull(self) -> "Bool":
        return Bool.wrap(self._col.isNotNull())

    def like(self, pattern: str) -> "Bool":
        return Bool.wrap(self._col.like(pattern))

    def _get_operand_column(self, other: Any):
        if isinstance(other, TypedColumn):
            return other._col
        else:
            return other

    # Comparison operators — accept same-type column only.
    # Primitives override to additionally accept Python literals.
    def __eq__(self, other: Self) -> "Bool":  # type: ignore[override]
        return Bool.wrap(self._col.__eq__(self._get_operand_column(other)))

    def __ne__(self, other: Self) -> "Bool":  # type: ignore[override]
        return Bool.wrap(self._col.__ne__(self._get_operand_column(other)))

    def __lt__(self, other: Self) -> "Bool":
        return Bool.wrap(self._col.__lt__(self._get_operand_column(other)))

    def __le__(self, other: Self) -> "Bool":
        return Bool.wrap(self._col.__le__(self._get_operand_column(other)))

    def __gt__(self, other: Self) -> "Bool":
        return Bool.wrap(self._col.__gt__(self._get_operand_column(other)))

    def __ge__(self, other: Self) -> "Bool":
        return Bool.wrap(self._col.__ge__(self._get_operand_column(other)))

    def __add__(self, other: Any) -> "TypedColumn":
        return TypedColumn(self._col.__add__(_unwrap(other)))

    def __sub__(self, other: Any) -> "TypedColumn":
        return TypedColumn(self._col.__sub__(_unwrap(other)))

    def __mul__(self, other: Any) -> "TypedColumn":
        return TypedColumn(self._col.__mul__(_unwrap(other)))

    def __truediv__(self, other: Any) -> "TypedColumn":
        return TypedColumn(self._col.__truediv__(_unwrap(other)))

    def __mod__(self, other: Any) -> "TypedColumn":
        return TypedColumn(self._col.__mod__(_unwrap(other)))

    def __pow__(self, other: Any) -> "TypedColumn":
        return TypedColumn(self._col.__pow__(_unwrap(other)))

    def __neg__(self) -> "TypedColumn":
        return TypedColumn(self._col.__neg__())

    def desc(self) -> "Self":
        return type(self).wrap(self._col.desc())

    def asc(self) -> "Self":
        return type(self).wrap(self._col.asc())

    def isin(self, *cols: Any) -> "Bool":
        return Bool.wrap(self.to_spark().isin(*cols))

    def rlike(self, item: str) -> "Bool":
        return Bool.wrap(self.to_spark().rlike(item))

    def between(self, lowerBound: Self, upperBound: Self) -> "Bool":
        return Bool.wrap(
            self.to_spark().between(
                self._get_operand_column(lowerBound),
                self._get_operand_column(upperBound),
            )
        )


class AliasedTypedColumn(TypedColumn):
    """When creating projected query plans we need to keep track of what has been aliased."""

    original_name: str

    def __init__(self, c: pyspark.sql.Column, name: str, original_name: str):
        super().__init__(c)
        self._name = name
        self.original_name = original_name


class Bool(TypedColumn[BooleanType]):
    _data_type = BooleanType
    __hash__ = TypedColumn.__hash__

    def __init__(self, col: "pyspark.sql.Column | TypedColumn[Any] | bool | None"):
        if not isinstance(col, (TypedColumn, pyspark.sql.Column)):
            col = pyspark.sql.functions.lit(col)
        super().__init__(col)

    def __eq__(self, other: "Bool | bool") -> "Bool":  # type: ignore[override]
        return Bool.wrap(self._col.__eq__(_unwrap(other)))

    def __ne__(self, other: "Bool | bool") -> "Bool":  # type: ignore[override]
        return Bool.wrap(self._col.__ne__(_unwrap(other)))

    def __and__(self, other: "Bool | bool") -> "Bool":
        return Bool.wrap(self._col.__and__(_unwrap(other)))

    def __or__(self, other: "Bool | bool") -> "Bool":
        return Bool.wrap(self._col.__or__(_unwrap(other)))

    def __invert__(self) -> "Bool":
        return Bool.wrap(self._col.__invert__())


def _unwrap(other: object) -> Any:
    return other._col if isinstance(other, TypedColumn) else other  # type: ignore[union-attr]
