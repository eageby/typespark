from typing import Any, Callable, Optional, get_origin

import attrs

from typespark import metadata
from typespark.columns import TypedColumn, is_typed_column_type
from typespark.utils import get_field_name

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
        if metadata.DF_ALIAS in m:
            return field

        f = field.evolve(metadata={**m, metadata.DF_ALIAS: func(field.name)})
        return f

    def inner(_: type, fields: list[attrs.Attribute]) -> list[attrs.Attribute]:
        return [alias_converter(f) for f in fields]

    return inner


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
