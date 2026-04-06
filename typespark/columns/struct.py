from typing import Self, dataclass_transform

import attr
import attrs
from pyspark.sql import Column
from pyspark.sql.functions import from_json, struct
from pyspark.sql.types import StringType, StructType

from typespark.base import _Base
from typespark.columns import TypedColumn
from typespark.columns.utils import is_typed_column_type
from typespark.metadata import decimal, field, foreign_key, primary_key
from typespark.utils import get_field_name, unwrap_origin


@dataclass_transform(
    field_specifiers=(attrs.field, attr.ib, decimal, foreign_key, primary_key, field),
    frozen_default=True,
)
class Struct(TypedColumn[StructType], _Base):
    @classmethod
    def set_column(cls, col: Column, name: str, tp: type[TypedColumn] | None = None):
        new = cls._build(col)
        object.__setattr__(new, "_col", col)  # Circumventing frozen
        object.__setattr__(new, "_name", name)  # Circumventing frozen
        return new

    def __attrs_post_init__(self):
        metadata = attrs.fields_dict(self.__class__)
        object.__setattr__(
            self,
            "_col",
            struct(
                *[
                    self.__getattribute__(k)._col.alias(k)
                    if issubclass(unwrap_origin(metadata[k].type or object), Struct)
                    else v._col.alias(get_field_name(metadata[k]))
                    for k, v in self.fields().items()
                ]
            ),
        )

    def fields(self) -> dict[str, TypedColumn]:
        return attrs.asdict(self, filter=lambda f, _: is_typed_column_type(f.type))

    @classmethod
    def from_json(
        cls, json: TypedColumn[StringType], options: dict[str, str] | None = None
    ) -> Self:
        return cls.set_column(
            from_json(json.to_spark(), cls.generate_schema(), options), json._name, None
        )
