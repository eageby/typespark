from typing import Optional, dataclass_transform

import attr
import attrs
from pyspark.sql import Column
from pyspark.sql.types import StructType

from typespark.columns import TypedColumn, is_typed_column_type
from typespark.field_transforms import FieldTransformer, pipe_tranformers
from typespark.metadata import decimal, field, foreign_key, primary_key
from typespark.utils import get_field_name, unwrap_type


@dataclass_transform(
    field_specifiers=(attrs.field, attr.ib, decimal, foreign_key, primary_key, field),
)
@attrs.define(init=False, slots=False)
class Struct(TypedColumn[StructType]):
    @classmethod
    def set_column(cls, col: Column, name: str, tp: Optional[type[TypedColumn]] = None):
        new = super().set_column(col, name, tp)

        for field_name, f in attrs.fields_dict(cls).items():
            field_alias = get_field_name(f)
            if is_typed_column_type(f.type):
                object.__setattr__(
                    new,
                    field_name,
                    unwrap_type(f.type).set_column(
                        col[field_alias], field_alias, unwrap_type(f.type)
                    ),
                )
        return new

    def fields(self) -> dict[str, TypedColumn]:
        return attrs.asdict(self, filter=lambda f, _: is_typed_column_type(f.type))

    @classmethod
    def __init_subclass__(
        cls, field_transformers: Optional[list[FieldTransformer]] = None
    ):
        ft = pipe_tranformers(*(field_transformers or []))
        attrs.define(init=False, slots=False, field_transformer=ft)(cls)
