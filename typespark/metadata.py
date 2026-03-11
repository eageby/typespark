from typing import Optional

import attrs

NULLABLE_COLUMN = f"_{__package__ or __name__}_nullable_column"
DECIMAL_TYPE_METADATA_PRECISION = (
    f"_{__package__ or __name__}_decimal_type_metadata_precision"
)
DECIMAL_TYPE_METADATA_SCALE = f"_{__package__ or __name__}_decimal_type_metadata_scale"

PRIMARY_KEY = f"_{__package__ or __name__}_primary_key"
FOREIGN_KEY = f"_{__package__ or __name__}_foreign_key"
DF_ALIAS = f"_{__package__ or __name__}_dataframe_alias"


def field(
    df_alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    is_primary_key: Optional[bool] = None,
    is_foreign_key: Optional[type] = None,
    metadata: dict[str, str] = {},
    **kwargs,
):

    md = {
        DF_ALIAS: df_alias,
        NULLABLE_COLUMN: nullable,
        PRIMARY_KEY: is_primary_key,
        FOREIGN_KEY: is_foreign_key,
    } | metadata

    return attrs.field(metadata=md, **kwargs)


def primary_key(
    df_alias: Optional[str] = None,
    metadata: dict[str, str] = {},
):
    md = {
        DF_ALIAS: df_alias,
        PRIMARY_KEY: True,
    } | metadata

    return attrs.field(metadata=md)


def foreign_key(
    reference: type,
    df_alias: Optional[str] = None,
    metadata: dict[str, str] = {},
):
    md = {
        DF_ALIAS: df_alias,
        FOREIGN_KEY: reference,
    } | metadata

    return attrs.field(metadata=md)


def decimal(
    precision: int,
    scale: int,
    df_alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    metadata: dict[str, str] = {},
    **kwargs,
):

    md = {
        DF_ALIAS: df_alias,
        DECIMAL_TYPE_METADATA_PRECISION: precision,
        DECIMAL_TYPE_METADATA_SCALE: scale,
        NULLABLE_COLUMN: nullable,
    } | metadata
    return attrs.field(metadata=md, **kwargs)


@attrs.define
class MetaData:
    precision: Optional[int] = attrs.field(
        alias=DECIMAL_TYPE_METADATA_PRECISION, default=None
    )
    scale: Optional[int] = attrs.field(alias=DECIMAL_TYPE_METADATA_SCALE, default=None)
    nullable: bool = attrs.field(alias=NULLABLE_COLUMN, default=True)
    primary_key: Optional[bool] = attrs.field(alias=PRIMARY_KEY, default=None)
    foreign_key: Optional[type] = attrs.field(alias=FOREIGN_KEY, default=None)

    df_alias: Optional[str] = attrs.field(alias=DF_ALIAS, default=None)
