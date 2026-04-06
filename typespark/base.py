"""
Schema base for TypeSpark.

_Base is a plain class (not attrs-decorated) that provides typed column field
introspection, schema generation, and the central _build() factory method.
It knows about attrs fields and TypedColumns but NOT about DataFrames.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Self,
    dataclass_transform,
)

import attrs

from typespark import schema
from typespark.columns import TypedColumn, is_typed_column_type
from typespark.exceptions import InvalidDefaultColumnError, MissingColumnError
from typespark.utils import get_field_name, unwrap_type

from .define import define
from .field_transforms import FieldTransformer

if TYPE_CHECKING:
    pass


@dataclass_transform(
    frozen_default=True,
)
class _Base:
    @property
    def columns(self) -> list[TypedColumn]:
        return [
            getattr(self, field.name)
            for field in attrs.fields(self.__class__)
            if is_typed_column_type(field.type)
        ]

    @classmethod
    def __init_subclass__(
        cls, field_transformers: list[FieldTransformer] | None = None
    ):
        define(cls, field_transformers=field_transformers)

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

    def columndict(self) -> dict[str, TypedColumn]:
        return {
            f.name: getattr(self, f.name)
            for f in attrs.fields(self.__class__)
            if is_typed_column_type(f.type)
        }

    @classmethod
    def _build(cls, source, *, col_ref=None) -> Self:
        """Construct instance from a data source.

        source: data container supporting __getitem__ (DataFrame or Column).
                Columns are extracted via source[field_alias].
                Not stored by _Base — callers store it as _dataframe, _col, etc.
        col_ref: optional callable (field_alias: str) -> Column.
                 When provided, overrides source[field_alias] for column
                 reference creation. Used by from_df for aliased access.
        """
        new = cls.__new__(cls)
        # DataFrame.columns returns list[str]; Column does not
        raw_columns = getattr(source, "columns", None)
        available = set(raw_columns) if isinstance(raw_columns, list) else None

        for field_name, f in attrs.fields_dict(cls).items():
            if not is_typed_column_type(f.type):
                continue
            field_alias = get_field_name(f)
            missing = available is not None and field_alias not in available

            if not missing:
                column = col_ref(field_alias) if col_ref else source[field_alias]
                object.__setattr__(
                    new,
                    field_name,
                    unwrap_type(f.type).set_column(
                        column, field_alias, unwrap_type(f.type)
                    ),
                )
            elif f.default is not attrs.NOTHING:
                default = f.default
                if not isinstance(default, TypedColumn):
                    raise InvalidDefaultColumnError(
                        model=cls,
                        field_name=field_name,
                        field_alias=field_alias,
                        default_value=default,
                    )
                object.__setattr__(
                    new,
                    field_name,
                    unwrap_type(f.type).set_column(
                        default.to_spark().alias(field_alias),
                        field_alias,
                        unwrap_type(f.type),
                    ),
                )
            else:
                raise MissingColumnError(
                    model=cls,
                    field_name=field_name,
                    expected_column=field_alias,
                )
        return new
