from __future__ import annotations

from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    from typespark.columns import TypedColumn


@attrs.define()
class _GroupColumn:
    column: TypedColumn
    _alias: str

    def __init__(self, column: TypedColumn) -> None:
        self.column = column

    def alias(self, alias: str):
        self.column = self.column.alias(alias)
        return self


class _AggregateColumn:
    column: TypedColumn
    _alias: str

    def __init__(self, column: TypedColumn) -> None:
        self.column = column

    def alias(self, alias: str):
        self.column = self.column.alias(alias)
        return self
