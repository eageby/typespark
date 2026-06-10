from __future__ import annotations

from typing import TYPE_CHECKING

import pyspark.sql
from pyspark.sql.types import DataType

if TYPE_CHECKING:
    from typespark.columns import TypedColumn


class _GroupBase:
    column: TypedColumn
    _alias: str

    def __init__(self, column: TypedColumn) -> None:
        self.column = column

    def to_spark(self) -> pyspark.sql.Column:
        return self.column.to_spark()

    def alias(self, alias: str):
        self.column = self.column.alias(alias)
        return self

    def cast(self, type: DataType):
        self.column = self.column.cast(type)
        return self


def _make_forbidden(op: str):
    def _forbidden(self, *_, **__):
        raise TypeError(
            f"`{op}` not allowed on _GroupColumn; "
            "value materializes as the field type after DataFrame init."
        )
    return _forbidden


_FORBIDDEN_OPS = [
    "__add__", "__radd__", "__sub__", "__rsub__",
    "__mul__", "__rmul__", "__truediv__", "__rtruediv__",
    "__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__",
]


class _GroupColumn(_GroupBase):
    pass


for _op in _FORBIDDEN_OPS:
    setattr(_GroupColumn, _op, _make_forbidden(_op))  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Aggregate sentinel mixin — dynamic subclass of the primitive type
# ---------------------------------------------------------------------------


class _AggregateMixin:
    """Marker mixin for aggregate columns produced by agg functions.

    Instances are dynamic subclasses of the concrete primitive type
    (e.g. ``_AggDouble(Double, _AggregateMixin)``), so all primitive
    arithmetic and comparison ops work normally. ``_resolve`` detects
    them via ``isinstance(c, _AggregateMixin)``.

    ``alias`` is overridden to preserve the dynamic subclass type — the default
    ``TypedColumn.alias`` returns ``AliasedTypedColumn`` which loses the mixin
    marker before ``_resolve`` can inspect it.
    """

    def alias(self, alias: str, **kwargs):
        inst = object.__new__(type(self))
        inst._col = self._col.alias(alias, **kwargs)  # type: ignore[attr-defined]
        inst._name = alias
        return inst

    def cast(self, dataType):
        inst = object.__new__(type(self))
        inst._col = self._col.cast(dataType)  # type: ignore[attr-defined]
        inst._name = self._name  # type: ignore[attr-defined]
        return inst

    def to_spark(self) -> pyspark.sql.Column:
        return super().to_spark()  # type: ignore[misc]


_agg_cls_cache: dict[type, type] = {}


def _make_aggregate(target_cls: type, spark_col: pyspark.sql.Column):
    """Return an instance of a dynamic subclass of *target_cls* that also
    inherits from ``_AggregateMixin``.

    Uses ``TypedColumn.wrap`` to skip the per-type cast — the Spark agg
    expression already carries the right type.
    """
    if target_cls not in _agg_cls_cache:
        _agg_cls_cache[target_cls] = type(
            f"_Agg{target_cls.__name__}", (_AggregateMixin, target_cls), {}
        )
    agg_cls = _agg_cls_cache[target_cls]
    return agg_cls.wrap(spark_col)
