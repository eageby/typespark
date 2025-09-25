import inspect
from typing import Callable, Iterable, get_args

import attrs
from attr import AttrsInstance
from pyspark.sql.types import DataType, StructField, StructType
from simple_parsing import docstring

from typespark.base import BaseDataFrame
from typespark.columns import TypedColumn, is_typed_column_type
from typespark.metadata import MetaData


def _extract_items(attrs_instance: AttrsInstance, *extract_keys: Iterable[str]):
    """Returns a dictionary only containing the items with keys specified in
    :arg:`extract_keys`."""

    return {
        k: v
        for k, v in attrs.asdict(
            attrs_instance,
            filter=lambda _attr, value: value is not None,
        ).items()
        if k in extract_keys
    }


def _extract_arg_names(func: Callable):
    """Inspect function signature and extract names of arguments."""
    return inspect.signature(func).parameters.keys()


def _extract_kwargs(func: Callable, attrs_instance: AttrsInstance):
    """Given a callable, return the items that match the signature."""
    return _extract_items(attrs_instance, *_extract_arg_names(func))


def _kwarg_safe_call(func: Callable, attrs_instance: AttrsInstance):
    """Call callable with only relevant kwargs."""
    return func(**_extract_kwargs(func, attrs_instance))


def get_type_instance(field: attrs.Attribute, m: MetaData):
    if field.type is None:
        raise ValueError("Field type missing")

    if not get_args(field.type) and is_typed_column_type(field.type):
        return generate_schema(field.type)
    else:
        field_type: type[DataType] = get_args(field.type)[0]
        return _kwarg_safe_call(field_type, m)


def _construct_struct_field(
    cls: type[BaseDataFrame] | type[TypedColumn], field: attrs.Attribute
):
    m = MetaData(
        **{k: v for k, v in field.metadata.items() if k in _extract_arg_names(MetaData)}
    )
    name = m.df_alias or field.name
    doc = docstring.get_attribute_docstring(cls, field.alias or field.name).help_string
    type_instance = get_type_instance(field, m)

    field_kwargs = _extract_kwargs(StructField, m)

    return StructField(
        name,
        type_instance,
        **field_kwargs,
        metadata={"comment": doc},
    )


def _get_type(field: attrs.Attribute):
    if field.type is None:
        raise ValueError("Field type is None.")

    if hasattr(field.type, "__origin__"):
        return field.type.__origin__

    return field.type


def generate_schema(cls: type[BaseDataFrame] | type[TypedColumn]):
    return StructType(
        [
            _construct_struct_field(cls, f)
            for f in attrs.fields(cls)
            if _get_type(f) == TypedColumn  # or issubclass(_get_type(f), TypedColumn)
        ]
    )
