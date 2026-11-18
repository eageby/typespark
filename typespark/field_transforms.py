from typing import Any, Callable, Optional

import attrs

from typespark import metadata
from typespark._exceptions import InvalidLiteralDefaultError
from typespark.columns import TypedArrayType, TypedColumn, is_typed_column_type
from typespark.utils import get_field_name, unwrap_type

FieldTransformer = Callable[[type, list[attrs.Attribute]], list[attrs.Attribute]]


def pipe_tranformers(*transformers: FieldTransformer) -> FieldTransformer:
    def inner(cls: type, fields: list[attrs.Attribute]):
        val = fields
        for t in transformers:
            val = t(cls, val)
        return val

    return inner


def set_alias(field: attrs.Attribute):
    alias = get_field_name(field)

    def alias_converter(x: TypedColumn):
        return x.alias(alias)

    return alias_converter


def df_alias(func: Callable[[str], str]) -> FieldTransformer:
    def alias_converter(field: attrs.Attribute):
        m = field.metadata.copy()
        if m.get(metadata.DF_ALIAS) is not None:
            return field

        f = field.evolve(metadata={**m, metadata.DF_ALIAS: func(field.name)})
        return f

    def inner(_: type, fields: list[attrs.Attribute]) -> list[attrs.Attribute]:
        return [alias_converter(f) for f in fields]

    return inner


def _literal_factory(col_cls: type[TypedColumn], value: Any) -> Callable[[], Any]:
    """Build the default via the column's own constructor.

    Primitive constructors take a literal (or ``None``) and handle any
    type-specific literal quirks (e.g. ``TimestampNTZ`` avoiding a session-tz
    shift). Structs and arrays only take ``None``, built with ``null()``.
    """
    if _is_container(col_cls):
        return col_cls.null  # type: ignore[attr-defined]
    return lambda: col_cls(value)


def _is_container(col_cls: type) -> bool:
    from typespark._base import _Base

    return issubclass(col_cls, (_Base, TypedArrayType))


def resolve_literal_defaults(
    cls: type, fields: list[attrs.Attribute]
) -> list[attrs.Attribute]:
    """Turn ``ts.field(default=<literal>)`` into a factory typed by the annotation."""
    new_fields = []
    for field in fields:
        if not isinstance(field.default, metadata.LiteralDefault):
            new_fields.append(field)
            continue

        value = field.default.value
        if not is_typed_column_type(field.type):
            new_fields.append(field.evolve(default=value))
            continue

        col_cls = unwrap_type(field.type)
        if value is None and field.metadata.get(metadata.NULLABLE_COLUMN) is False:
            raise InvalidLiteralDefaultError(
                model=cls,
                field_name=field.name,
                value=value,
                reason="The field is declared nullable=False.",
            )
        if value is not None and _is_container(col_cls):
            raise InvalidLiteralDefaultError(
                model=cls,
                field_name=field.name,
                value=value,
                reason=(
                    f"{col_cls.__name__} is a struct or array; only None is "
                    "supported as a literal default. Use factory= for anything else."
                ),
            )
        new_fields.append(
            field.evolve(default=attrs.Factory(_literal_factory(col_cls, value)))
        )
    return new_fields


def converter[T](
    func: Callable[[T], T],
) -> Callable[[attrs.Attribute], Callable[[T], T]]:
    def inner(_: attrs.Attribute):
        return func

    return inner


def to_transformer(
    converter: Callable[[attrs.Attribute], Callable[[Any], Any]],
    predicate: Optional[Callable[[attrs.Attribute], bool]] = None,
) -> FieldTransformer:
    def field_transformer(
        _: type, fields: list[attrs.Attribute]
    ) -> list[attrs.Attribute]:
        new_fields = []
        for field in fields:
            skip = predicate and not predicate(field)

            if not skip and is_typed_column_type(field.type):
                new_converter = converter(field)
                existing = field.converter

                if existing is not None:
                    composed = attrs.converters.pipe(existing, new_converter)
                    new_field = field.evolve(converter=composed)
                else:
                    new_field = field.evolve(converter=new_converter)

                new_fields.append(new_field)
            else:
                new_fields.append(field)

        return new_fields

    return field_transformer
