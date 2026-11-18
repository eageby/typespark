import inspect
import typing
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


def _converter_input_type(converter: Any) -> Optional[type[TypedColumn]]:
    """The column type a converter takes, from its first parameter's annotation.

    ``None`` when it can't be resolved (a lambda, no hints, a non-column hint).
    """
    fn = (
        getattr(converter, "converter")  # not exposed in attrs' stubs
        if isinstance(converter, attrs.Converter)
        else converter
    )
    try:
        params = list(inspect.signature(fn).parameters)
        hints = typing.get_type_hints(fn)
    except (TypeError, ValueError, NameError):
        return None
    if not params or params[0] not in hints:
        return None
    tp = unwrap_type(hints[params[0]])
    return tp if is_typed_column_type(tp) else None


def resolve_literal_defaults(
    cls: type, fields: list[attrs.Attribute]
) -> list[attrs.Attribute]:
    """Turn ``ts.field(default=<literal>)`` into a typed column factory.

    Like attrs, a default is the value *before* the converter runs. With a
    converter, the literal is built as the converter's input type (falling back
    to the annotation when that can't be read, e.g. a lambda); without one, as
    the annotated type.
    """
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
        # With a converter, a null default may be filled in, so the
        # nullable=False check only applies without one.
        if field.converter is not None:
            col_cls = _converter_input_type(field.converter) or col_cls
        elif value is None and field.metadata.get(metadata.NULLABLE_COLUMN) is False:
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
