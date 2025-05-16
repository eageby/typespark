import logging
from typing import Annotated, Generic, Protocol, Self, TypeVar, cast

import attrs
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StructType

from typespark.attrs_descriptor import use_descriptor

logging.basicConfig(level=logging.INFO)


T = TypeVar("T")


class HasName(Protocol):
    name: str


class TypedColumn(Generic[T], Column):
    _col: Column
    _name: str

    def __init__(self, c: Column, name: str):  # pylint: disable=super-init-not-called
        """Only initializes from existing Column."""
        self._col = c
        self._name = name

    def __getattr__(self, item) -> Column:
        return getattr(self._col, item)

    def __repr__(self):
        return f"{self.__class__.__name__}<'{self._name}'>"


class Descriptor:
    name: str

    def __get__(self, obj: "TypedDataFrame | None", objtype=None) -> HasName:
        if obj:
            if obj._alias:
                return TypedColumn(F.col(f"{obj._alias}.{self.name}"), self.name)

            return TypedColumn(obj._dataframe[self.name], self.name)

        return self

    def __set__(self, obj, value):
        setattr(obj, self.name, value)

    def __set_name__(self, owner, name):

        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}<'{self.name}'>"


class Struct(TypedColumn[StructType]):
    @classmethod
    def __init_subclass__(cls):
        attrs.define(
            slots=False,
            init=False,
        )(cls)


class TypedArrayType(ArrayType, Generic[T]):
    pass


class TypedList(TypedColumn[TypedArrayType[T]]):
    def __getitem__(self, item) -> TypedColumn[T]:
        return TypedColumn(self.getItem(item))


D = TypeVar("D", bound="TypedDataFrame")

type Aliased[D] = Annotated[D, "Aliased"]


class TypedDataFrame(DataFrame):
    _dataframe: DataFrame
    _alias: str | None = None

    def __init__(
        self, dataframe: DataFrame, alias: str | None = None
    ):  # pylint: disable=super-init-not-called
        """Only initializes from existing DataFrame."""
        self._dataframe = dataframe
        self._alias = alias

    @classmethod
    def __init_subclass__(cls):
        attrs.define(
            slots=False,
            init=False,
            field_transformer=use_descriptor(TypedColumn, Descriptor),
        )(cls)

    def alias(self, alias: str) -> Aliased[Self]:
        new = self._dataframe.alias(alias)
        result = self.__class__(new, alias)
        return result

    def __getattr__(self, item):
        return getattr(self._dataframe, item)
