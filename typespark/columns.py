import functools
from typing import Optional, get_args, get_origin

from pyspark.sql import Column
from pyspark.sql.types import ArrayType, DataType

from typespark.generator import Exploded
from typespark.utils import unwrap_type


class TypedColumn[T: DataType](Column):
    _col: Column
    _name: str | None

    def __hash__(self) -> int:
        return hash(self._name)

    @functools.wraps(Column.alias)
    def name(self, alias: str, **kwargs): ...

    def __init__(self, c: Column):  # pylint: disable=super-init-not-called
        """Only initializes from existing Column."""
        self._col = c

    def _set_name(self, name: str):
        self._name = name
        return self

    def to_spark(self):
        return self._col

    @classmethod
    def set_column(
        cls, col: Column, name: str, tp: Optional[type["TypedColumn"]] = None
    ):
        return cls(col)._set_name(name)

    def __getattr__(self, item) -> Column:
        return getattr(self._col, item)

    def __repr__(self):
        return f"{self.__class__.__name__}<'{self._name}'>"

    def cast[U: DataType](self, dataType: U) -> "TypedColumn[U]":
        return TypedColumn(self._col.cast(dataType))

    def alias[U: DataType](self, alias: str, **kwargs) -> "TypedColumn[U]":
        return TypedColumn(self._col.alias(alias, **kwargs))._set_name(self._name)


def is_typed_column_type(tp) -> bool:
    tp = unwrap_type(tp)
    origin = get_origin(tp)
    if origin and (origin is TypedColumn or issubclass(origin, TypedColumn)):
        return True
    if isinstance(tp, type) and issubclass(tp, TypedColumn):
        return True
    return False


class TypedArrayType[T: TypedColumn](TypedColumn[ArrayType]):
    _elem_type: type[TypedColumn]

    @classmethod
    def set_column(cls, col: Column, name: str, tp: Optional[type[TypedColumn]] = None):
        new = super().set_column(col, name, tp)
        if tp:
            new._elem_type = get_args(tp)[0]

        return new

    def explode(self) -> T:
        return Exploded(
            parent=self, alias=self._name, elem_type=self._elem_type
        )  # type:ignore
