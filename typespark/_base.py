"""
Schema base for TypeSpark.

_Base is a plain class (not attrs-decorated) that provides typed column field
introspection, schema generation, and the central _build() factory method.
It knows about attrs fields and TypedColumns but NOT about DataFrames.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Self,
    dataclass_transform,
)

import attrs
from pyspark.sql.types import StructType

from typespark import schema
from typespark._exceptions import InvalidDefaultColumnError, MissingColumnError
from typespark.columns import TypedColumn, is_typed_column_type
from typespark.utils import get_field_name, unwrap_type

from ._define import define
from .field_transforms import FieldTransformer


def _convert(f: attrs.Attribute, value: Any, instance: Any) -> Any:
    """Apply a field's converter like attrs' generated ``__init__`` does."""
    conv = f.converter
    if conv is None:
        return value
    if isinstance(conv, attrs.Converter):
        # attrs' stubs don't expose these attributes.
        args: list[Any] = [value]
        if getattr(conv, "takes_self"):  # noqa: B009
            args.append(instance)
        if getattr(conv, "takes_field"):  # noqa: B009
            args.append(f)
        return getattr(conv, "converter")(*args)  # noqa: B009
    return conv(value)


@dataclass_transform(
    frozen_default=True,
)
class _Base:
    # Set by attrs at runtime; declared so mypy treats subclasses as
    # AttrsInstance. ClassVar[Any] matches attrs' own protocol, so it
    # doesn't clash with attrs classes like _DataFrameFields.
    if TYPE_CHECKING:
        __attrs_attrs__: ClassVar[Any]

    @property
    def columns(self) -> list[TypedColumn]:
        return [
            getattr(self, field.name)
            for field in attrs.fields(self.__class__)
            if is_typed_column_type(field.type)
        ]

    @classmethod
    def __init_subclass__(
        cls, field_transformers: list[FieldTransformer] | None = None, **kwargs: Any
    ):
        super().__init_subclass__(**kwargs)
        define(cls, field_transformers=field_transformers)

    @classmethod
    def _column_aliases(cls) -> Iterator[str]:
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
                struct_cls = unwrap_type(f.type)
                if (
                    available is not None
                    and struct_cls is not None
                    and isinstance(struct_cls, type)
                    and issubclass(struct_cls, _Base)
                    and hasattr(source, "schema")
                ):
                    try:
                        nested_type = source.schema[field_alias].dataType
                    except KeyError:
                        pass
                    else:
                        if isinstance(nested_type, StructType):
                            actual = set(nested_type.fieldNames())
                            for sub_alias in struct_cls._column_aliases():
                                if sub_alias not in actual:
                                    raise MissingColumnError(
                                        model=struct_cls,
                                        field_name=sub_alias,
                                        expected_column=sub_alias,
                                        available_columns=sorted(actual),
                                        parent_field=field_alias,
                                    )
                column = col_ref(field_alias) if col_ref else source[field_alias]
                object.__setattr__(
                    new,
                    field_name,
                    unwrap_type(f.type)._set_column(column, field_alias),
                )
            elif f.default is not attrs.NOTHING:
                default = f.default
                if isinstance(default, attrs.Factory):  # type: ignore[arg-type]
                    factory: Any = default
                    default = (
                        factory.factory(new)
                        if factory.takes_self
                        else factory.factory()
                    )
                # As in attrs' __init__, the default goes through the converter.
                # Every column field has one (set_alias), so a non-column default
                # usually fails inside it; report that as an invalid default.
                raw_default = default
                invalid = InvalidDefaultColumnError(
                    model=cls,
                    field_name=field_name,
                    field_alias=field_alias,
                    default_value=raw_default,
                )
                try:
                    default = _convert(f, default, new)
                except Exception as e:
                    if isinstance(raw_default, TypedColumn):
                        raise
                    raise invalid from e
                if not isinstance(default, TypedColumn):
                    raise invalid
                object.__setattr__(
                    new,
                    field_name,
                    unwrap_type(f.type)._set_column(
                        default.to_spark().alias(field_alias),
                        field_alias,
                    ),
                )
            else:
                raise MissingColumnError(
                    model=cls,
                    field_name=field_name,
                    expected_column=field_alias,
                    available_columns=sorted(available)
                    if available is not None
                    else None,
                )
        return new
