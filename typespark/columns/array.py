import warnings
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, overload

import attrs
import pyspark.sql.functions as F
from pyspark.sql import Column
from pyspark.sql.functions import from_json as _spark_from_json
from pyspark.sql.types import ArrayType, DecimalType, StringType
from typing_extensions import deprecated

if TYPE_CHECKING:
    from pyspark.sql.types import DataType

from typespark._exceptions import UnnamedColumnError
from typespark.columns._generator import Exploded
from typespark.columns.columns import TypedColumn, _concrete_registry
from typespark.columns.primitives import Decimal

_array_cache: dict[type, type] = {}


def _elem_data_type(
    elem: "type[TypedColumn]", decimal: DecimalType | None = None
) -> "DataType":
    """Recursively resolve DataType for an array element type.

    ``decimal`` overrides the DataType of a (possibly nested) Decimal element.
    """
    if attrs.has(elem):
        return elem.generate_schema()  # type: ignore[attr-defined]
    if issubclass(elem, TypedArrayType):
        return ArrayType(_elem_data_type(elem._require_elem_type(), decimal))
    if decimal is not None and issubclass(elem, Decimal):
        return decimal
    if elem._data_type is not None:
        return elem._data_type()
    raise ValueError(f"Cannot determine DataType for element type {elem!r}")


def _innermost_elem(elem: "type[TypedColumn]") -> "type[TypedColumn]":
    while issubclass(elem, TypedArrayType):
        elem = elem._require_elem_type()
    return elem


def _column_class_for(data_type: "DataType") -> "type[TypedColumn]":
    """Resolve the column class for a DataType, recursing into arrays."""
    if isinstance(data_type, ArrayType):
        return _array_of(_column_class_for(data_type.elementType))
    return _concrete_registry.get(type(data_type), TypedColumn)


def _array_of[T: TypedColumn](elem_type: type[T]) -> "type[TypedArrayType[T]]":
    if elem_type in _array_cache:
        return _array_cache[elem_type]
    cls = type(
        f"Array[{elem_type.__name__}]",
        (TypedArrayType,),
        {"_elem_type": elem_type},
    )
    _array_cache[elem_type] = cls
    return cls


@deprecated("ArrayOf(T) is removed in 2.0; use ts.Array[T], which is the same class.")
def ArrayOf[T: TypedColumn](elem_type: type[T]) -> "type[TypedArrayType[T]]":
    """Deprecated alias of ``Array[T]``."""
    return _array_of(elem_type)


class TypedArrayType[T: TypedColumn](TypedColumn[ArrayType]):
    # ClassVar can't hold a type variable; _require_elem_type() restores T.
    _elem_type: ClassVar[type[TypedColumn[Any]] | None] = None

    if not TYPE_CHECKING:

        def __class_getitem__(cls, item):
            return _array_of(item)

    @overload
    def __init__(self, c: "Column | TypedColumn[Any]") -> None:
        """Wrap ``c``, casting to the array schema."""

    @overload
    def __init__(
        self: "TypedArrayType[Decimal] | TypedArrayType[TypedArrayType[Decimal]]",
        c: "Column | TypedColumn[Any]",
        precision: int,
        scale: int = 0,
    ) -> None:
        """Cast to an array of DecimalType(precision, scale), e.g.
        ``Array[Decimal](col, precision=18, scale=4)``."""

    def __init__(self, c, precision=None, scale=None):  # type: ignore[misc]
        if precision is None and scale is None:
            super().__init__(c)
            return
        elem = self._require_elem_type()
        if not issubclass(_innermost_elem(elem), Decimal):
            raise TypeError(
                f"precision/scale only apply to Decimal elements, not {self.__class__.__name__}"
            )
        decimal = DecimalType(
            10 if precision is None else precision, 0 if scale is None else scale
        )
        col = c.to_spark() if isinstance(c, TypedColumn) else c
        self._col = col.cast(ArrayType(_elem_data_type(elem, decimal)))
        if isinstance(c, TypedColumn):
            self._name = c._name

    @classmethod
    def _require_elem_type(cls) -> type[T]:
        if cls._elem_type is None:
            raise TypeError(f"{cls.__name__} has no element type — use Array[ElemType]")
        return cast("type[T]", cls._elem_type)

    @classmethod
    def _cast_schema(cls) -> ArrayType | None:
        if cls._elem_type is None:
            warnings.warn(
                f"{cls.__name__} constructed without a type parameter — "
                "use TypedArrayType[ElemType](col) to enable schema casting. "
                "No cast will be applied.",
                UserWarning,
                stacklevel=3,
            )
            return None
        return ArrayType(_elem_data_type(cls._elem_type))

    @classmethod
    def empty(cls) -> "Self":
        return cls.wrap(F.array().cast(cls.generate_schema()))

    @classmethod
    def null(cls) -> "Self":
        return cls.wrap(F.lit(None).cast(cls.generate_schema()))

    @classmethod
    def from_json(
        cls, json: "TypedColumn[StringType]", options: dict[str, str] | None = None
    ) -> "Self":
        if json._name is None:
            raise UnnamedColumnError(column_type=json.__class__, operation="from_json")
        return cls._set_column(
            _spark_from_json(json.to_spark(), cls.generate_schema(), options),
            json._name,
        )

    @classmethod
    def generate_schema(cls) -> ArrayType:
        if cls._elem_type is None:
            raise TypeError(
                f"Cannot generate schema for unparameterized {cls.__name__}"
            )
        return ArrayType(_elem_data_type(cls._elem_type))

    def getItem(self, key: int) -> T:
        item = self._col.getItem(key)
        name = f"{self._name}[{key}]" if self._name else str(key)
        return self._require_elem_type()._set_column(item, name)

    def explode(self) -> T:
        if self._name is None:
            raise UnnamedColumnError(column_type=self.__class__, operation="explode")
        return Exploded(parent=self, alias=self._name, elem_type=self._elem_type)  # type: ignore
