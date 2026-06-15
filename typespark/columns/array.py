import attrs
from typing import TYPE_CHECKING, ClassVar, Self

from pyspark.sql import Column
import pyspark.sql.functions as F
from pyspark.sql.functions import from_json as _spark_from_json
from pyspark.sql.types import ArrayType, StringType

if TYPE_CHECKING:
    from pyspark.sql.types import DataType

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
        return ArrayType(_elem_data_type(elem._elem_type))
    if elem._data_type is not None:
        return elem._data_type()
    raise ValueError(f"Cannot determine DataType for element type {elem!r}")


def ArrayOf[T: TypedColumn](elem_type: type[T]) -> "type[TypedArrayType[T]]":
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
    _elem_type: ClassVar[type[TypedColumn]]

    if not TYPE_CHECKING:
        def __class_getitem__(cls, item):
            return ArrayOf(item)

    @classmethod
    def _cast_schema(cls) -> ArrayType:
        return ArrayType(_elem_data_type(cls._elem_type))

    @classmethod
    def empty(cls) -> "Self":
        return cls.wrap(F.array().cast(cls.generate_schema()))

    @classmethod
    def null(cls) -> "Self":
        return cls.wrap(F.lit(None).cast(cls.generate_schema()))

    @classmethod
    def from_json(cls, json: "TypedColumn[StringType]", options: dict[str, str] | None = None) -> "Self":
        if json._name is None:
            raise UnnamedColumnError(column_type=json.__class__, operation="from_json")
        return cls._set_column(_spark_from_json(json.to_spark(), cls.generate_schema(), options), json._name)

    @classmethod
    def generate_schema(cls) -> ArrayType:
        return ArrayType(_elem_data_type(cls._elem_type))

    def getItem(self, key: int) -> T:
        item = self._col.getItem(key)
        name = f"{self._name}[{key}]" if self._name else str(key)
        return self._elem_type._set_column(item, name)

    def explode(self) -> T:
        if self._name is None:
            raise UnnamedColumnError(column_type=self.__class__, operation="explode")
        return Exploded(parent=self, alias=self._name, elem_type=self._elem_type)  # type: ignore
