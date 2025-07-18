from typing import Annotated, Union, get_args, get_origin

import attrs

from typespark.metadata import MetaData


def get_field_name(field: attrs.Attribute):
    return MetaData(**field.metadata).df_alias or field.name


def unwrap_type(tp):
    origin = get_origin(tp)
    if origin in (Annotated, Union):
        for arg in get_args(tp):
            base = unwrap_type(arg)
            if base is not None:
                return base
    return tp
