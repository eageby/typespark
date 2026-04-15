import attrs
from typing import ClassVar

from pyspark.sql import Column
from pyspark.sql.types import ArrayType

from typespark.columns.columns import TypedColumn
from typespark.columns.generator import Exploded
from typespark.exceptions import UnnamedColumnError

_array_cache: dict[type, type] = {}


def ArrayOf(elem_type: type[TypedColumn]) -> type["TypedArrayType"]:
    if elem_type in _array_cache:
        return _array_cache[elem_type]
    cls = type(
        f"Array[{elem_type.__name__}]",
        (TypedArrayType,),
        {"_elem_type": elem_type},
    )
    _array_cache[elem_type] = cls
    return cls


class TypedArrayType[T: TypedColumn](TypedColumn[ArrayType]):
    _elem_type: ClassVar[type[TypedColumn] | None] = None

    def __class_getitem__(cls, item):
        return ArrayOf(item)

    @classmethod
    def _cast_schema(cls) -> ArrayType | None:
        if cls._elem_type is None:
            return None
        elem = cls._elem_type
        elem_schema = elem.generate_schema() if attrs.has(elem) else elem._data_type()
        return ArrayType(elem_schema)

    def getItem(self, key: int) -> T:
        item = self._col.getItem(key)
        if self._elem_type is not None:
            return self._elem_type(item)
        return TypedColumn(item)  # type: ignore

    def explode(self) -> T:
        if self._name is None:
            raise UnnamedColumnError(column_type=self.__class__, operation="explode")
        return Exploded(parent=self, alias=self._name, elem_type=self._elem_type)  # type: ignore
