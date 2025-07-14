import inspect
from typing import Any, Callable, Dict, Iterable, Optional, TypedDict, Unpack

import attrs

from typespark.typed_dataframe import TypedDataFrame

_NULLABLE_COLUMN = f"_{__package__ or __name__}_nullable_column"
_DECIMAL_TYPE_METADATA_PRECISION = (
    f"_{__package__ or __name__}_decimal_type_metadata_precision"
)
_DECIMAL_TYPE_METADATA_SCALE = f"_{__package__ or __name__}_decimal_type_metadata_scale"

_PRIMARY_KEY = f"_{__package__ or __name__}_primary_key"
_FOREIGN_KEY = f"_{__package__ or __name__}_foreign_key"


def base(
    alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    primary_key: Optional[bool] = None,
    foreign_key: Optional[type[TypedDataFrame]] = None,
    **kwargs,
):

    metadata = {
        _NULLABLE_COLUMN: nullable,
        _PRIMARY_KEY: primary_key,
        _FOREIGN_KEY: foreign_key,
    }

    return attrs.field(metadata=metadata, alias=alias)


def primary_key(
    alias: Optional[str] = None,
):
    metadata = {
        _PRIMARY_KEY: True,
    }

    return attrs.field(metadata=metadata, alias=alias)


def foreign_key(
    reference: type[TypedDataFrame],
    alias: Optional[str] = None,
):
    metadata = {
        _FOREIGN_KEY: reference,
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
    primary_key: Optional[bool] = attrs.field(alias=_PRIMARY_KEY, default=None)
    foreign_key: Optional[type[TypedDataFrame]] = attrs.field(
        alias=_FOREIGN_KEY, default=None
    )
