from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Callable, Iterable

import attrs
from attr import AttrsInstance
from pyspark.sql.types import ArrayType, DataType, StructField, StructType
from simple_parsing import docstring

from typespark.columns import TypedColumn, is_typed_column_type
from typespark.columns.array import TypedArrayType
from typespark.metadata import MetaData

if TYPE_CHECKING:
    from typespark.base import _Base
    from typespark.dataframe import BaseDataFrame


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


def get_type_instance(field: attrs.Attribute, m: MetaData) -> DataType:
    tp = field.type
    if tp is None:
        raise ValueError("Field type missing")

    # Struct — has attrs fields, generates a StructType
    if attrs.has(tp):
        return generate_schema(tp)  # type: ignore

    # Array — _elem_type carries the element class
    if issubclass(tp, TypedArrayType):
        elem = tp._elem_type
        if elem is None:
            raise ValueError(f"Array field '{field.name}' has no element type")
        elem_schema = generate_schema(elem) if attrs.has(elem) else _kwarg_safe_call(elem._data_type, m)  # type: ignore
        return ArrayType(elem_schema)

    # Primitive — _data_type is the Spark DataType class
    return _kwarg_safe_call(tp._data_type, m)  # type: ignore


def _construct_struct_field(
    cls: type[BaseDataFrame] | type[TypedColumn] | type[_Base], field: attrs.Attribute
):
    m = MetaData(**field.metadata)
    name = m.df_alias or field.name
    doc = docstring.get_attribute_docstring(cls, field.alias or field.name).help_string
    type_instance = get_type_instance(field, m)

    field_kwargs = _extract_kwargs(StructField, m)

    return StructField(
        name,
        type_instance,
        **field_kwargs,
        metadata={"comment": doc} if doc or len(doc) > 0 else {},
    )


def _get_type(field: attrs.Attribute):
    if field.type is None:
        raise ValueError("Field type is None.")

    if hasattr(field.type, "__origin__"):
        return field.type.__origin__

    return field.type


def generate_schema(cls: type[BaseDataFrame] | type[TypedColumn] | type[_Base]):
    return StructType(
        [
            _construct_struct_field(cls, f)
            for f in attrs.fields(cls)
            if is_typed_column_type(_get_type(f))
        ]
    )
