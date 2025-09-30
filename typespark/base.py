from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    Optional,
    Self,
    Union,
    dataclass_transform,
)

import attr
import attrs
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DataType

from typespark.columns import AliasedTypedColumn, TypedColumn, is_typed_column_type
from typespark.define import define
from typespark.generator import DeferredColumn, Generator
from typespark.interface import SupportsETLFrame, SupportsGroupedData
from typespark.metadata import decimal, field, foreign_key, primary_key
from typespark.mixins import Aliasable, SchemaDefaults
from typespark.utils import get_field_name, unwrap_type

if TYPE_CHECKING:
    from typespark.field_transforms import FieldTransformer


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
    def columns(self) -> list[TypedColumn]:
        return [
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
    def __init_subclass__(
        cls, field_transformers: Optional[list[FieldTransformer]] = None
    ):
        define(cls, field_transformers=field_transformers)

    def columndict(self) -> dict[str, TypedColumn]:
        return attrs.asdict(self, filter=lambda f, _: is_typed_column_type(f.type))

    def __attrs_post_init__(self):
        object.__setattr__(
            self, "_dataframe", self.select(*self.columns).to_df()
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
            field_alias = get_field_name(f)
            if is_typed_column_type(f.type):
                object.__setattr__(
                    new,
                    field_name,
                    unwrap_type(f.type).set_column(
                        df[field_alias], field_alias, unwrap_type(f.type)
                    ),
                )

        return new


class BaseDataFrame(_Base, SupportsETLFrame, Aliasable, SchemaDefaults):
    def select(
        self, *cols: Union[str, Column] | TypedColumn[DataType]
    ) -> "BaseDataFrame":
        projections = set([c.parent for c in cols if isinstance(c, DeferredColumn)])
        normal_cols = [
            c
            for c in cols
            if not (isinstance(c, DeferredColumn) or isinstance(c, Generator))
        ]

        if len(projections) > 0:
            projected_cols = [
                c.column_operation() if isinstance(c, Generator) else c
                for c in projections
            ]
            # Prevent aliasing normal cols twice
            original_named_cols = [
                F.col(n.original_name) if isinstance(n, AliasedTypedColumn) else n
                for n in normal_cols
            ]
            # Step 1: materialize generators
            df = self._dataframe.select(*projected_cols, *original_named_cols)

            # Step 2: select final materialized expressions
            final_cols = [c.col if isinstance(c, DeferredColumn) else c for c in cols]

        else:
            df = self._dataframe
            final_cols = [
                c.column_operation() if isinstance(c, Generator) else c for c in cols
            ]
        return BaseDataFrame.from_df(df.select(*final_cols), disable_select=True)

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

    def union(self, other: Self) -> Self:
        return self.__class__.from_df(self._dataframe.unionByName(other.to_df()))

    def join(
        self,
        other: Any,
        on: str | list[str] | Column | None = None,
        how: str | None = None,
    ) -> "BaseDataFrame":
        return BaseDataFrame.from_df(
            self._dataframe.join(other, on, how), disable_select=True
        )
