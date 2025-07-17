from typing import Generic, TypeVar

import attrs
from pyspark.sql import Column
from pyspark.sql.types import ArrayType, StructType

T = TypeVar("T")


class TypedColumn(Generic[T], Column):
    pass
    # _col: Column
    # _name: str

    # def __init__(self, c: Column, name: str):  # pylint: disable=super-init-not-called
    #     """Only initializes from existing Column."""
    #     self._col = c
    #     self._name = name

    # def __getattr__(self, item) -> Column:
    #     return getattr(self._col, item)

    # def __repr__(self):
    #     return f"{self.__class__.__name__}<'{self._name}'>"


class Struct(TypedColumn[StructType]):
    @classmethod
    def __init_subclass__(cls):
        attrs.define(
            slots=False,
            init=False,
        )(cls)


class TypedArrayType(ArrayType, Generic[T]):
    pass


# class TypedList(TypedColumn[TypedArrayType[T]]):
#     def __getitem__(self, item) -> TypedColumn[T]:
#         return TypedColumn(self.getItem(item))
