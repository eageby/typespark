from typing import (
    Any,
    Generator,
    Optional,
    Self,
    Union,
    dataclass_transform,
    get_origin,
)

import attr
import attrs
from pyspark.sql import Column, DataFrame

from typespark.field_transforms import (
    FieldTransformer,
    add_converter,
    pipe_tranformers,
    set_alias,
)
from typespark.interface import SupportsETLFrame, SupportsGroupedData
from typespark.metadata import decimal, field, foreign_key, primary_key
from typespark.serialization_alias import Aliasable
from typespark.typed_dataframe import TypedColumn
from typespark.utils import get_field_name


def _dataframe_converter(df: "_Base | DataFrame"):
    if isinstance(df, DataFrame):
        return df
    return df.to_df()


@dataclass_transform(
    field_specifiers=(attrs.field, attr.ib, decimal, foreign_key, primary_key, field),
)
def _define(cls, field_transformers: list[FieldTransformer] | None = None):
    if field_transformers is None:
        field_transformers = [add_converter(set_alias)]

    ft = pipe_tranformers(*field_transformers)

    return attrs.define(field_transformer=ft, slots=False, frozen=True)(cls)


@dataclass_transform(
    frozen_default=True,
    field_specifiers=(attrs.field, attr.ib, decimal, foreign_key, primary_key, field),
)
@attrs.define(frozen=True)
class _Base:
    _dataframe: DataFrame = attrs.field(converter=_dataframe_converter, alias="df")
    _alias: Optional[str] = attrs.field(init=False, default=None)

    def __getattr__(self, name: str):
        field_annotation = attrs.fields_dict(self.__class__).get(name, None)
        if (
            field_annotation is not None
            and get_origin(field_annotation.type) is TypedColumn
        ):
            return getattr(self._dataframe, get_field_name(field_annotation))
        return getattr(self._dataframe, name)

    def to_df(self):
        return self._dataframe

    @property
    def columns(self) -> Generator[TypedColumn, None, None]:
        yield from [
            getattr(self, field.name)
            for field in attrs.fields(self.__class__)
            if get_origin(field.type) is TypedColumn
        ]

    @classmethod
    def _column_aliases(cls):
        yield from [
            get_field_name(field)
            for field in attrs.fields(cls)
            if get_origin(field.type) is TypedColumn
        ]

    @classmethod
    def __init_subclass__(cls):
        _define(cls)

    def __attrs_post_init__(self):
        object.__setattr__(
            self, "_dataframe", self._dataframe.select(*self.columns)
        )  # Circumventing frozen

    @classmethod
    def from_df(cls, df: DataFrame, alias: str | None = None):
        new = cls.__new__(cls)
        object.__setattr__(new, "_alias", alias)
        if alias is not None:
            selected_df = df.alias(alias)
        else:
            selected_df = df
        object.__setattr__(new, "_dataframe", selected_df)

        return new


class BaseDataFrame(_Base, SupportsETLFrame, Aliasable):
    def select(self, *cols: Union[str, Column] | TypedColumn) -> "BaseDataFrame":
        return BaseDataFrame.from_df(self._dataframe.select(*cols))

    def withColumn(self, colName: str, col: Column) -> "BaseDataFrame":
        return BaseDataFrame.from_df(self._dataframe.withColumn(colName, col))

    def drop(self, *cols) -> "BaseDataFrame":
        return BaseDataFrame.from_df(self._dataframe.drop(*cols))

    def show(self, n: int = 20, truncate: bool = True, vertical: bool = False):
        self._dataframe.show(n, truncate, vertical)

    def distinct(self) -> Self:
        return self.from_df(self._dataframe.distinct())

    def filter(self, condition) -> Self:
        return self.from_df(self._dataframe.filter(condition))

    def alias(self, alias: str) -> Self:
        return self.from_df(self._dataframe, alias)

    def groupBy(self, *cols: str | Column) -> SupportsGroupedData:
        return self._dataframe.groupBy(*cols)

    def join(
        self,
        other: Any,
        on: str | list[str] | Column | None = None,
        how: str | None = None,
    ) -> "BaseDataFrame":
        return BaseDataFrame.from_df(self._dataframe.join(other, on, how))
