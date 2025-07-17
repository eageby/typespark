from typing import Any, Callable, get_origin

import attrs

from typespark.typed_dataframe import TypedColumn
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

    def alias_converter(x):
        return x.alias(alias)

    return alias_converter


def add_converter(converter: Callable[[attrs.Attribute], Callable[[Any], Any]]):
    def field_transformer(
        _: type, fields: list[attrs.Attribute]
    ) -> list[attrs.Attribute]:
        new_fields = []
        for field in fields:
            if get_origin(field.type) == TypedColumn:
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
