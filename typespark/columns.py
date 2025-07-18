import functools
import attrs
import attr
from typing import Generic, TypeVar, dataclass_transform, get_origin

from pyspark.sql import Column, DataFrame
from pyspark.sql.types import ArrayType, StructType

from typespark.metadata import decimal, field, foreign_key, primary_key
from typespark.utils import get_field_name, unwrap_type

T = TypeVar("T")


class TypedColumn(Column, Generic[T]):
    _col: Column
    _name: str

    @functools.wraps(Column.alias)
    def name(self, alias: str, **kwargs): ...

    def __init__(self, c: Column, name: str):  # pylint: disable=super-init-not-called
        """Only initializes from existing Column."""
        self._col = c
        self._name = name

    @classmethod
    def from_df(cls, df: DataFrame, name: str):
        return cls(df[name], name)

    def __getattr__(self, item) -> Column:
        return getattr(self._col, item)

    def __repr__(self):
        return f"{self.__class__.__name__}<'{self._name}'>"


def is_typed_column_type(tp) -> bool:
    tp = unwrap_type(tp)
    try:
        if get_origin(tp) is TypedColumn:
            return True
        if isinstance(tp, type) and issubclass(tp, TypedColumn):
            return True
    except TypeError:
        pass
    return False


class TypedArrayType(ArrayType, Generic[T]):
    pass


@dataclass_transform(
    field_specifiers=(attrs.field, attr.ib, decimal, foreign_key, primary_key, field),
)
@attrs.define(init=False, slots=False)
class Struct(TypedColumn[StructType]):
    @classmethod
    def from_df(cls, df: DataFrame, name: str):
        new = super().from_df(df, name)

        for field_name, f in attrs.fields_dict(cls).items():
            if is_typed_column_type(f.type):
                object.__setattr__(
                    new,
                    field_name,
                    unwrap_type(f.type).from_df(df[name], get_field_name(f)),
                )
        return new

    @classmethod
    def __init_subclass__(cls):
        attrs.define(init=False, slots=False)(cls)
