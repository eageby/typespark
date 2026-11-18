import datetime
import decimal as _decimal
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Optional, TypedDict, Unpack, overload

import attrs

if TYPE_CHECKING:
    from typespark import _default_types as _dt
    from typespark.columns.columns import TypedColumn
    from typespark.columns.primitives import Date
    from typespark.columns.primitives import Decimal as DecimalColumn

_LITERAL_TYPES = (str, int, float, _decimal.Decimal, datetime.date)


class AttrsFieldKwargs(TypedDict, total=False):
    """``attrs.field`` options passed through by ``field``/``decimal``.

    Excludes ``default``/``factory`` (explicit parameters) and ``metadata``, so
    those are type checked rather than swallowed by ``**kwargs``.
    """

    validator: Any
    repr: Any
    hash: Optional[bool]
    init: bool
    type: Optional[type]
    converter: Any
    kw_only: bool
    eq: Any
    order: Any
    on_setattr: Any
    alias: Optional[str]


NULLABLE_COLUMN = f"_{__package__ or __name__}_nullable_column"
DECIMAL_TYPE_METADATA_PRECISION = (
    f"_{__package__ or __name__}_decimal_type_metadata_precision"
)
DECIMAL_TYPE_METADATA_SCALE = f"_{__package__ or __name__}_decimal_type_metadata_scale"

PRIMARY_KEY = f"_{__package__ or __name__}_primary_key"
FOREIGN_KEY = f"_{__package__ or __name__}_foreign_key"
DF_ALIAS = f"_{__package__ or __name__}_dataframe_alias"


@attrs.frozen
class LiteralDefault:
    """A literal (or ``None``) default, typed from the field annotation later.

    ``ts.field`` runs before attrs has read the annotation, so it cannot build
    the typed column itself. It stores the raw value here, and the
    ``resolve_literal_defaults`` field transformer swaps it for a factory that
    builds the column from the field's type.
    """

    value: Any


def _default_kwargs(default: Any, factory: Any) -> dict[str, Any]:
    """Map ``ts.field``'s ``default``/``factory`` onto ``attrs.field`` kwargs.

    - ``default``: a literal or ``None`` is wrapped in ``LiteralDefault``;
      anything else (a ``TypedColumn``, an ``attrs.Factory``) passes through.
    - ``factory``: a callable is passed as attrs ``factory``; an
      ``attrs.Factory`` is passed as attrs ``default`` so ``takes_self`` works.
    """
    if factory is not None:
        if default is not attrs.NOTHING:
            raise ValueError("The `default` and `factory` arguments are exclusive.")
        if isinstance(factory, attrs.Factory):  # type: ignore[arg-type]
            return {"default": factory}
        return {"factory": factory}
    if default is attrs.NOTHING:
        return {}
    if default is None or isinstance(default, _LITERAL_TYPES):
        return {"default": LiteralDefault(default)}
    return {"default": default}


@overload
def field[C: "TypedColumn[Any]"](
    df_alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    is_primary_key: Optional[bool] = None,
    is_foreign_key: Optional[type] = None,
    metadata: Mapping[Any, Any] = {},  # noqa: B006
    *,
    factory: "Callable[[], C]",
    **kwargs: Unpack[AttrsFieldKwargs],
) -> "C": ...


@overload
def field[C: "TypedColumn[Any]"](
    df_alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    is_primary_key: Optional[bool] = None,
    is_foreign_key: Optional[type] = None,
    metadata: Mapping[Any, Any] = {},  # noqa: B006
    *,
    factory: "C",
    **kwargs: Unpack[AttrsFieldKwargs],
) -> "C": ...


@overload
def field[C: "TypedColumn[Any]"](
    df_alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    is_primary_key: Optional[bool] = None,
    is_foreign_key: Optional[type] = None,
    metadata: Mapping[Any, Any] = {},  # noqa: B006
    *,
    default: "C",
    **kwargs: Unpack[AttrsFieldKwargs],
) -> "C": ...


@overload
def field(
    df_alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    is_primary_key: Optional[bool] = None,
    is_foreign_key: Optional[type] = None,
    metadata: Mapping[Any, Any] = {},  # noqa: B006
    *,
    default: None,
    **kwargs: Unpack[AttrsFieldKwargs],
) -> Any: ...


@overload
def field(  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
    df_alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    is_primary_key: Optional[bool] = None,
    is_foreign_key: Optional[type] = None,
    metadata: Mapping[Any, Any] = {},  # noqa: B006
    *,
    default: bool,
    **kwargs: Unpack[AttrsFieldKwargs],
) -> "_dt.BoolDefault": ...


@overload
def field(
    df_alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    is_primary_key: Optional[bool] = None,
    is_foreign_key: Optional[type] = None,
    metadata: Mapping[Any, Any] = {},  # noqa: B006
    *,
    default: int,
    **kwargs: Unpack[AttrsFieldKwargs],
) -> "_dt.IntDefault": ...


@overload
def field(
    df_alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    is_primary_key: Optional[bool] = None,
    is_foreign_key: Optional[type] = None,
    metadata: Mapping[Any, Any] = {},  # noqa: B006
    *,
    default: float,
    **kwargs: Unpack[AttrsFieldKwargs],
) -> "_dt.FloatDefault": ...


@overload
def field(
    df_alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    is_primary_key: Optional[bool] = None,
    is_foreign_key: Optional[type] = None,
    metadata: Mapping[Any, Any] = {},  # noqa: B006
    *,
    default: _decimal.Decimal,
    **kwargs: Unpack[AttrsFieldKwargs],
) -> "DecimalColumn": ...


@overload
def field(
    df_alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    is_primary_key: Optional[bool] = None,
    is_foreign_key: Optional[type] = None,
    metadata: Mapping[Any, Any] = {},  # noqa: B006
    *,
    default: str,
    **kwargs: Unpack[AttrsFieldKwargs],
) -> "_dt.StrDefault": ...


@overload
def field(
    df_alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    is_primary_key: Optional[bool] = None,
    is_foreign_key: Optional[type] = None,
    metadata: Mapping[Any, Any] = {},  # noqa: B006
    *,
    default: datetime.datetime,
    **kwargs: Unpack[AttrsFieldKwargs],
) -> "_dt.DatetimeDefault": ...


@overload
def field(
    df_alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    is_primary_key: Optional[bool] = None,
    is_foreign_key: Optional[type] = None,
    metadata: Mapping[Any, Any] = {},  # noqa: B006
    *,
    default: datetime.date,
    **kwargs: Unpack[AttrsFieldKwargs],
) -> "Date": ...


@overload
def field(
    df_alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    is_primary_key: Optional[bool] = None,
    is_foreign_key: Optional[type] = None,
    metadata: Mapping[Any, Any] = {},  # noqa: B006
    **kwargs: Unpack[AttrsFieldKwargs],
) -> Any: ...


def field(
    df_alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    is_primary_key: Optional[bool] = None,
    is_foreign_key: Optional[type] = None,
    metadata: Mapping[Any, Any] = {},  # noqa: B006
    *,
    default: Any = attrs.NOTHING,
    factory: Any = None,
    **kwargs: Unpack[AttrsFieldKwargs],
) -> Any:
    """Declare a column field.

    ``default`` takes a literal or ``None``; it is turned into a column of the
    field's annotated type (e.g. ``default=None`` on a ``ts.String`` is a null
    string column). ``factory`` takes a callable returning a ``TypedColumn``,
    or an ``attrs.Factory`` (for ``takes_self=True``).
    """
    md = {
        DF_ALIAS: df_alias,
        NULLABLE_COLUMN: nullable,
        PRIMARY_KEY: is_primary_key,
        FOREIGN_KEY: is_foreign_key,
    } | dict(metadata)

    return attrs.field(metadata=md, **_default_kwargs(default, factory), **kwargs)


def primary_key(
    df_alias: Optional[str] = None,
    metadata: Mapping[Any, Any] = {},  # noqa: B006
):
    md = {
        DF_ALIAS: df_alias,
        PRIMARY_KEY: True,
    } | dict(metadata)

    return attrs.field(metadata=md)


def foreign_key(
    reference: type,
    df_alias: Optional[str] = None,
    metadata: Mapping[Any, Any] = {},  # noqa: B006
):
    md = {
        DF_ALIAS: df_alias,
        FOREIGN_KEY: reference,
    } | dict(metadata)

    return attrs.field(metadata=md)


@overload
def decimal[C: "TypedColumn[Any]"](
    precision: int,
    scale: int,
    df_alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    metadata: Mapping[Any, Any] = {},  # noqa: B006
    *,
    factory: "Callable[[], C]",
    **kwargs: Unpack[AttrsFieldKwargs],
) -> "C": ...


@overload
def decimal[C: "TypedColumn[Any]"](
    precision: int,
    scale: int,
    df_alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    metadata: Mapping[Any, Any] = {},  # noqa: B006
    *,
    factory: "C",
    **kwargs: Unpack[AttrsFieldKwargs],
) -> "C": ...


@overload
def decimal[C: "TypedColumn[Any]"](
    precision: int,
    scale: int,
    df_alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    metadata: Mapping[Any, Any] = {},  # noqa: B006
    *,
    default: "C",
    **kwargs: Unpack[AttrsFieldKwargs],
) -> "C": ...


@overload
def decimal(
    precision: int,
    scale: int,
    df_alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    metadata: Mapping[Any, Any] = {},  # noqa: B006
    *,
    default: None,
    **kwargs: Unpack[AttrsFieldKwargs],
) -> Any: ...


@overload
def decimal(
    precision: int,
    scale: int,
    df_alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    metadata: Mapping[Any, Any] = {},  # noqa: B006
    *,
    default: int | float | _decimal.Decimal,
    **kwargs: Unpack[AttrsFieldKwargs],
) -> "DecimalColumn": ...


@overload
def decimal(
    precision: int,
    scale: int,
    df_alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    metadata: Mapping[Any, Any] = {},  # noqa: B006
    **kwargs: Unpack[AttrsFieldKwargs],
) -> Any: ...


def decimal(
    precision: int,
    scale: int,
    df_alias: Optional[str] = None,
    nullable: Optional[bool] = None,
    metadata: Mapping[Any, Any] = {},  # noqa: B006
    *,
    default: Any = attrs.NOTHING,
    factory: Any = None,
    **kwargs: Unpack[AttrsFieldKwargs],
) -> Any:
    md = {
        DF_ALIAS: df_alias,
        DECIMAL_TYPE_METADATA_PRECISION: precision,
        DECIMAL_TYPE_METADATA_SCALE: scale,
        NULLABLE_COLUMN: nullable,
    } | dict(metadata)
    return attrs.field(metadata=md, **_default_kwargs(default, factory), **kwargs)


@attrs.define(init=False)
class MetaData:
    precision: Optional[int] = attrs.field(
        alias=DECIMAL_TYPE_METADATA_PRECISION, default=None
    )
    scale: Optional[int] = attrs.field(alias=DECIMAL_TYPE_METADATA_SCALE, default=None)
    nullable: bool = attrs.field(alias=NULLABLE_COLUMN, default=True)
    primary_key: Optional[bool] = attrs.field(alias=PRIMARY_KEY, default=None)
    foreign_key: Optional[type] = attrs.field(alias=FOREIGN_KEY, default=None)

    df_alias: Optional[str] = attrs.field(alias=DF_ALIAS, default=None)

    def __init__(self, **kwargs):
        filtered = {
            attribute.alias or attribute.name: kwargs[attribute.alias or attribute.name]
            for attribute in attrs.fields(self.__class__)
            if (attribute.alias or attribute.name) in kwargs
        }
        self.__attrs_init__(**filtered)  # type: ignore[attr-defined]
