import inspect
from typing import Any, Callable, Dict, Iterable, Optional, TypedDict, Unpack

import attrs

_NULLABLE_COLUMN = f"_{__package__ or __name__}_nullable_column"
_DECIMAL_TYPE_METADATA_PRECISION = (
    f"_{__package__ or __name__}_decimal_type_metadata_precision"
)
_DECIMAL_TYPE_METADATA_SCALE = f"_{__package__ or __name__}_decimal_type_metadata_scale"


def base(
    alias: Optional[str] = None,
    nullable: Optional[bool] = None,
):

    metadata = {
        _NULLABLE_COLUMN: nullable,
    }

    return attrs.field(metadata=metadata, alias=alias)


def decimal(
    precision: int,
    scale: int,
    alias: Optional[str] = None,
    nullable: Optional[bool] = None,
):

    metadata = {
        _DECIMAL_TYPE_METADATA_PRECISION: precision,
        _DECIMAL_TYPE_METADATA_SCALE: scale,
        _NULLABLE_COLUMN: nullable,
    }

    return attrs.field(metadata=metadata, alias=alias)


@attrs.define
class MetaData:
    precision: Optional[int] = attrs.field(
        alias=_DECIMAL_TYPE_METADATA_PRECISION, default=None
    )
    scale: Optional[int] = attrs.field(alias=_DECIMAL_TYPE_METADATA_SCALE, default=None)
    nullable: bool = attrs.field(alias=_NULLABLE_COLUMN, default=True)
