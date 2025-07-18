from typing import dataclass_transform

import attr
import attrs

from typespark.field_transforms import (
    FieldTransformer,
    add_converter,
    pipe_tranformers,
    set_alias,
)
from typespark.metadata import decimal, field, foreign_key, primary_key


@dataclass_transform(
    field_specifiers=(attrs.field, attr.ib, decimal, foreign_key, primary_key, field),
)
def define(cls, field_transformers: list[FieldTransformer] | None = None):
    if field_transformers is None:
        field_transformers = [add_converter(set_alias)]

    ft = pipe_tranformers(*field_transformers)

    return attrs.define(field_transformer=ft, slots=False, frozen=True)(cls)
