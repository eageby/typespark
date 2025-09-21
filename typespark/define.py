from typing import dataclass_transform

import attr
import attrs

from typespark.field_transforms import (
    FieldTransformer,
    pipe_tranformers,
    set_alias,
    to_transformer,
)
from typespark.metadata import decimal, field, foreign_key, primary_key


@dataclass_transform(
    field_specifiers=(attrs.field, attr.ib, decimal, foreign_key, primary_key, field),
)
def define(cls, field_transformers: list[FieldTransformer] | None = None):

    ft = pipe_tranformers(*(field_transformers or []), to_transformer(set_alias))

    return attrs.define(field_transformer=ft, slots=False, frozen=True)(cls)
