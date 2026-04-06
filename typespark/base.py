"""
Attrs-based schema machinery for TypeSpark DataFrames.

_Base is a frozen attrs class that wraps a pyspark.sql.DataFrame and manages
typed column fields. It is inherited by user-defined schema classes (via
BaseDataFrame) to provide field introspection and schema generation.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Optional,
    Self,
    dataclass_transform,
    overload,
)

import attr
import attrs
import pyspark.sql
from pyspark.sql import functions as F

from typespark import schema
from typespark.columns import TypedColumn, is_typed_column_type
from typespark.define import define
from typespark.metadata import decimal, field, foreign_key, primary_key
from typespark.utils import get_field_name, unwrap_type

if TYPE_CHECKING:
    from typespark.field_transforms import FieldTransformer


def _dataframe_converter(df: "_Base | pyspark.sql.DataFrame"):
    if isinstance(df, pyspark.sql.DataFrame):
        return df
    return df.to_df()


@dataclass_transform(
    frozen_default=True,
    field_specifiers=(attrs.field, attr.ib, decimal, foreign_key, primary_key, field),
)
@attrs.define(frozen=True)
class _Base:
    _dataframe: pyspark.sql.DataFrame = attrs.field(
        converter=_dataframe_converter, alias="df"
    )
    _alias: Optional[str] = attrs.field(init=False, default=None)

    def to_spark(self):
        return self.to_df()

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
    def generate_schema(cls):
        return schema.generate_schema(cls)

    @classmethod
    def __init_subclass__(
        cls, field_transformers: Optional[list[FieldTransformer]] = None
    ):
        define(cls, field_transformers=field_transformers)

    def columndict(self) -> dict[str, TypedColumn]:
        return {
            k: getattr(self, k)
            for k, v in attrs.asdict(
                self, filter=lambda f, _: is_typed_column_type(f.type)
            ).items()
        }

    def __attrs_post_init__(self):
        object.__setattr__(
            self, "_dataframe", self.select(*self.columns).to_df()
        )  # Circumventing frozen

    @overload
    @classmethod
    def from_df(
        cls,
        df: pyspark.sql.DataFrame,
        alias: str | None = None,
        disable_select: bool = False,
    ) -> Self: ...

    @overload
    @classmethod
    def from_df(
        cls, df: _Base, alias: str | None = None, disable_select: bool = False
    ) -> Self: ...

    @classmethod
    def from_df(
        cls,
        df: "pyspark.sql.DataFrame | _Base",
        alias: str | None = None,
        disable_select: bool = False,
    ) -> Self:
        new = cls.__new__(cls)
        object.__setattr__(new, "_alias", alias)

        if isinstance(df, _Base):
            df = df.to_df()

        if not disable_select:
            df = df.select(*cls._column_aliases())

        if alias is not None:
            df = df.alias(alias)

        object.__setattr__(new, "_dataframe", df)

        for field_name, f in attrs.fields_dict(cls).items():
            field_alias = get_field_name(f)
            if is_typed_column_type(f.type):
                if alias is not None:
                    column_reference = F.col(f"{alias}.{field_alias}")
                else:
                    column_reference = df[field_alias]

                object.__setattr__(
                    new,
                    field_name,
                    unwrap_type(f.type).set_column(
                        column_reference, field_alias, unwrap_type(f.type)
                    ),
                )

        return new
