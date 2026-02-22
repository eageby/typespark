from typing import Optional, Self, get_args

from pyspark.sql import Column
from pyspark.sql.types import ArrayType

from typespark.columns.columns import TypedColumn
from typespark.columns.generator import Exploded


class TypedArrayType[T: TypedColumn](TypedColumn[ArrayType]):
    _elem_type: type[T] = None  # type:ignore

    def __init__(self, c: Column | Self, type: Optional[type[T]] = None):
        super().__init__(c)
        if type is not None:
            self._elem_type = type

    def getItem(self, key: int) -> T:
        item = self._col.getItem(key)
        if self._elem_type is not None:
            return self._elem_type(item)
        return TypedColumn(item)  # type:ignore

    @classmethod
    def set_column(cls, col: Column, name: str, tp: Optional[type[TypedColumn]] = None):
        new = super().set_column(col, name, tp)
        if tp:
            new._elem_type = get_args(tp)[0]

        return new

    def explode(self) -> T:
        return Exploded(parent=self, alias=self._name, elem_type=self._elem_type)  # type:ignore
