"""Static-only return types for ``ts.field(default=<literal>)``.

A literal default is typed from the field annotation at runtime, so one literal
fits several column types (an ``int`` can default a ``Long``, ``Double``,
``Decimal``, ...). The overloads on ``ts.field`` return one of these classes,
each a subclass of every column type the literal may default, so a type checker
accepts ``x: ts.Long = ts.field(default=5)`` and rejects
``x: ts.String = ts.field(default=5)``.

Only imported under ``TYPE_CHECKING``; none of these classes exist at runtime.
"""

from typing import Any

from typespark.columns.columns import Bool
from typespark.columns.primitives import (
    Date,
    Decimal,
    Double,
    Float,
    IntOf,
    Long,
    Short,
    StringOf,
    Timestamp,
    TimestampNTZ,
)

# The bases disagree on method signatures (e.g. comparison operand types);
# irrelevant here, since these only serve as assignment targets.


class BoolDefault(Bool): ...


class StrDefault(StringOf[Any]): ...


class IntDefault(IntOf[Any], Short, Long, Float, Double, Decimal): ...  # type: ignore[misc]  # pyright: ignore


class FloatDefault(Float, Double, Decimal): ...  # type: ignore[misc]  # pyright: ignore


class DatetimeDefault(Timestamp, TimestampNTZ, Date): ...  # type: ignore[misc]  # pyright: ignore
