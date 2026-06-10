import attrs
from typing import ClassVar

from pyspark.sql import Column
from pyspark.sql.types import ArrayType

from typespark.columns.columns import TypedColumn
from typespark.columns._generator import Exploded
from typespark._exceptions import UnnamedColumnError

_array_cache: dict[type, type] = {}


def _elem_data_type(elem: "type[TypedColumn]") -> "DataType":
    """Recursively resolve DataType for an array element type."""
    from pyspark.sql.types import DataType
    if attrs.has(elem):
        return elem.generate_schema()  # type: ignore[attr-defined]
    if issubclass(elem, TypedArrayType):
        inner = elem._elem_type
        if inner is None:
            raise ValueError(f"Nested array type {elem.__name__} has no element type")
        return ArrayType(_elem_data_type(inner))
    if elem._data_type is not None:
        return elem._data_type()
    raise ValueError(f"Cannot determine DataType for element type {elem!r}")


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
        return ArrayType(_elem_data_type(cls._elem_type))

    @classmethod
    def generate_schema(cls) -> ArrayType:
        if cls._elem_type is None:
            raise ValueError(f"{cls.__name__} has no element type — use ArrayOf(ElemType)")
        return ArrayType(_elem_data_type(cls._elem_type))

    def getItem(self, key: int) -> T:
        item = self._col.getItem(key)
        if self._elem_type is not None:
            return self._elem_type(item)
        return TypedColumn(item)  # type: ignore

    def explode(self) -> T:
        if self._name is None:
            raise UnnamedColumnError(column_type=self.__class__, operation="explode")
        return Exploded(parent=self, alias=self._name, elem_type=self._elem_type)  # type: ignore
