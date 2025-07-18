from typing import Any, Generator, Self, Union, dataclass_transform, Optional

import attr
import attrs
from pyspark.sql import Column, DataFrame

from typespark.define import define
from typespark.interface import SupportsETLFrame, SupportsGroupedData
from typespark.metadata import decimal, field, foreign_key, primary_key
from typespark.serialization_alias import Aliasable
from typespark.columns import TypedColumn, is_typed_column_type
from typespark.utils import get_field_name, unwrap_type


def _dataframe_converter(df: "_Base | DataFrame"):
    if isinstance(df, DataFrame):
        return df
    return df.to_df()


@dataclass_transform(
    frozen_default=True,
    field_specifiers=(attrs.field, attr.ib, decimal, foreign_key, primary_key, field),
)
@attrs.define(frozen=True)
class _Base:
    _dataframe: DataFrame = attrs.field(converter=_dataframe_converter, alias="df")
    _alias: Optional[str] = attrs.field(init=False, default=None)

    def __getattr__(self, name: str):
        return getattr(self._dataframe, name)

    def to_df(self):
        return self._dataframe

    @property
    def columns(self) -> Generator[TypedColumn, None, None]:
        yield from [
            getattr(self, field.name)
            for field in attrs.fields(self.__class__)
            if is_typed_column_type(field.type)
        ]

    @classmethod
    def _column_aliases(cls):
        yield from [
            get_field_name(field)
            for field in attrs.fields(cls)
            if is_typed_column_type(field.type)
        ]

    @classmethod
    def __init_subclass__(cls):
        define(cls)

    def __attrs_post_init__(self):
        object.__setattr__(
            self, "_dataframe", self._dataframe.select(*self.columns)
        )  # Circumventing frozen

    @classmethod
    def from_df(
        cls, df: DataFrame, alias: str | None = None, disable_select: bool = False
    ):
        new = cls.__new__(cls)
        object.__setattr__(new, "_alias", alias)

        if not disable_select:
            df = df.select(*cls._column_aliases())

        if alias is not None:
            df = df.alias(alias)

        object.__setattr__(new, "_dataframe", df)

        for field_name, f in attrs.fields_dict(cls).items():

            if is_typed_column_type(f.type):
                object.__setattr__(
                    new, field_name, unwrap_type(f.type).from_df(df, get_field_name(f))
                )

        return new


class BaseDataFrame(_Base, SupportsETLFrame, Aliasable):
    def select(self, *cols: Union[str, Column] | TypedColumn) -> "BaseDataFrame":
        return BaseDataFrame.from_df(self._dataframe.select(*cols), disable_select=True)

    def withColumn(self, colName: str, col: Column) -> "BaseDataFrame":
        return BaseDataFrame.from_df(
            self._dataframe.withColumn(colName, col), disable_select=True
        )

    def drop(self, *cols) -> "BaseDataFrame":
        return BaseDataFrame.from_df(self._dataframe.drop(*cols), disable_select=True)

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
        return BaseDataFrame.from_df(
            self._dataframe.join(other, on, how), disable_select=True
        )
