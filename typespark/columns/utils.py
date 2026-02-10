from typing import get_origin

from typespark.columns.columns import TypedColumn
from typespark.utils import unwrap_type


def is_typed_column_type(tp) -> bool:
    tp = unwrap_type(tp)
    origin = get_origin(tp)
    if origin and (origin is TypedColumn or issubclass(origin, TypedColumn)):
        return True
    if isinstance(tp, type) and issubclass(tp, TypedColumn):
        return True
    return False
