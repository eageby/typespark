from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Union, get_args, get_origin

if TYPE_CHECKING:
    from typespark.base import BaseDataFrame
import attrs

from typespark.metadata import DF_ALIAS, MetaData


def get_field_name(field: attrs.Attribute):
    return field.metadata.get(DF_ALIAS, None) or field.name


def unwrap_type(tp):
    origin = get_origin(tp)
    if origin in (Annotated, Union):
        for arg in get_args(tp):
            base = unwrap_type(arg)
            if base is not None:
                return base
    return tp


def unwrap_origin(tp: type) -> type:
    origin = get_origin(tp)
    if origin is not None:
        return unwrap_origin(origin)
    return tp


def get_primary_keys(cls: type[BaseDataFrame]):
    return {
        fn: f
        for fn, f in attrs.fields_dict(cls).items()
        if MetaData(**f.metadata).primary_key
    }


def get_foreign_keys(cls: type[BaseDataFrame]):
    return {
        fn: f
        for fn, f in attrs.fields_dict(cls).items()
        if MetaData(**f.metadata).foreign_key
    }
