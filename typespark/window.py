"""Typed wrapper around :class:`pyspark.sql.Window`.

Mirrors the PySpark builder API, but ``partitionBy`` / ``orderBy`` also accept
typed columns::

    w = ts.Window.partitionBy(orders.customer_id).orderBy(orders.amount.desc())
    orders.qualify(tsf.row_number().over(w) == 1)
"""

from typing import TypeAlias

import pyspark.sql
from pyspark.sql import Window as _SparkWindow
from pyspark.sql.window import WindowSpec as _SparkWindowSpec

from typespark.columns.columns import TypedColumn

__all__ = ["Window", "WindowSpec"]

WindowColumn: TypeAlias = TypedColumn | pyspark.sql.Column | str


def _unwrap(cols: tuple[WindowColumn, ...]) -> list[pyspark.sql.Column | str]:
    return [c.to_spark() if isinstance(c, TypedColumn) else c for c in cols]


class WindowSpec:
    """A window specification. Build one with :class:`Window`."""

    def __init__(self, spec: _SparkWindowSpec) -> None:
        self._spec = spec

    def to_spark(self) -> _SparkWindowSpec:
        return self._spec

    def partitionBy(self, *cols: WindowColumn) -> "WindowSpec":
        return WindowSpec(self._spec.partitionBy(*_unwrap(cols)))

    def orderBy(self, *cols: WindowColumn) -> "WindowSpec":
        return WindowSpec(self._spec.orderBy(*_unwrap(cols)))

    def rowsBetween(self, start: int, end: int) -> "WindowSpec":
        return WindowSpec(self._spec.rowsBetween(start, end))

    def rangeBetween(self, start: int, end: int) -> "WindowSpec":
        return WindowSpec(self._spec.rangeBetween(start, end))


class Window:
    """Entry point for building a :class:`WindowSpec`, like :class:`pyspark.sql.Window`."""

    unboundedPreceding: int = _SparkWindow.unboundedPreceding
    unboundedFollowing: int = _SparkWindow.unboundedFollowing
    currentRow: int = _SparkWindow.currentRow

    @staticmethod
    def partitionBy(*cols: WindowColumn) -> WindowSpec:
        return WindowSpec(_SparkWindow.partitionBy(*_unwrap(cols)))

    @staticmethod
    def orderBy(*cols: WindowColumn) -> WindowSpec:
        return WindowSpec(_SparkWindow.orderBy(*_unwrap(cols)))

    @staticmethod
    def rowsBetween(start: int, end: int) -> WindowSpec:
        return WindowSpec(_SparkWindow.rowsBetween(start, end))

    @staticmethod
    def rangeBetween(start: int, end: int) -> WindowSpec:
        return WindowSpec(_SparkWindow.rangeBetween(start, end))
