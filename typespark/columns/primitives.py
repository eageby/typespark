import datetime
import decimal as _decimal
from typing import Any, Self

from pyspark.sql.types import (
    BinaryType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    TimestampType,
)

from typespark.columns.columns import Bool, TypedColumn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unwrap(other: object) -> Any:
    """Unwrap a TypedColumn to its PySpark Column, or pass through a literal."""
    return other._col if isinstance(other, TypedColumn) else other  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Comparison-only mixins (eq, ne, lt, le, gt, ge)
# ---------------------------------------------------------------------------


class _StringOpsMixin(TypedColumn):
    __hash__ = TypedColumn.__hash__

    def __eq__(self, other: "Self | str") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__eq__(_unwrap(other)))

    def __ne__(self, other: "Self | str") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__ne__(_unwrap(other)))

    def __lt__(self, other: "Self | str") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__lt__(_unwrap(other)))

    def __le__(self, other: "Self | str") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__le__(_unwrap(other)))

    def __gt__(self, other: "Self | str") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__gt__(_unwrap(other)))

    def __ge__(self, other: "Self | str") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__ge__(_unwrap(other)))


class _DateOpsMixin(TypedColumn):
    __hash__ = TypedColumn.__hash__

    def __eq__(self, other: "Self | datetime.date") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__eq__(_unwrap(other)))

    def __ne__(self, other: "Self | datetime.date") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__ne__(_unwrap(other)))

    def __lt__(self, other: "Self | datetime.date") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__lt__(_unwrap(other)))

    def __le__(self, other: "Self | datetime.date") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__le__(_unwrap(other)))

    def __gt__(self, other: "Self | datetime.date") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__gt__(_unwrap(other)))

    def __ge__(self, other: "Self | datetime.date") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__ge__(_unwrap(other)))


class _TimestampOpsMixin(TypedColumn):
    __hash__ = TypedColumn.__hash__

    def __eq__(self, other: "Self | datetime.datetime") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__eq__(_unwrap(other)))

    def __ne__(self, other: "Self | datetime.datetime") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__ne__(_unwrap(other)))

    def __lt__(self, other: "Self | datetime.datetime") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__lt__(_unwrap(other)))

    def __le__(self, other: "Self | datetime.datetime") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__le__(_unwrap(other)))

    def __gt__(self, other: "Self | datetime.datetime") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__gt__(_unwrap(other)))

    def __ge__(self, other: "Self | datetime.datetime") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__ge__(_unwrap(other)))


class _BinaryOpsMixin(TypedColumn):
    __hash__ = TypedColumn.__hash__

    def __eq__(self, other: "Self | bytes") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__eq__(_unwrap(other)))

    def __ne__(self, other: "Self | bytes") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__ne__(_unwrap(other)))


# ---------------------------------------------------------------------------
# Numeric mixins (comparison + arithmetic + unary neg)
# ---------------------------------------------------------------------------


class _IntOpsMixin(TypedColumn):
    __hash__ = TypedColumn.__hash__

    # -- comparison --
    def __eq__(self, other: "Self | int") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__eq__(_unwrap(other)))

    def __ne__(self, other: "Self | int") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__ne__(_unwrap(other)))

    def __lt__(self, other: "Self | int") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__lt__(_unwrap(other)))

    def __le__(self, other: "Self | int") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__le__(_unwrap(other)))

    def __gt__(self, other: "Self | int") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__gt__(_unwrap(other)))

    def __ge__(self, other: "Self | int") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__ge__(_unwrap(other)))

    # -- arithmetic --
    def __add__(self, other: "Self | int") -> "Self":
        return self.__class__.wrap(self._col.__add__(_unwrap(other)))

    def __sub__(self, other: "Self | int") -> "Self":
        return self.__class__.wrap(self._col.__sub__(_unwrap(other)))

    def __mul__(self, other: "Self | int") -> "Self":
        return self.__class__.wrap(self._col.__mul__(_unwrap(other)))

    def __truediv__(self, other: "Self | int") -> "Double":
        return Double.wrap(self._col.__truediv__(_unwrap(other)))

    def __mod__(self, other: "Self | int") -> "Self":
        return self.__class__.wrap(self._col.__mod__(_unwrap(other)))

    def __pow__(self, other: "Self | int") -> "Double":
        return Double.wrap(self._col.__pow__(_unwrap(other)))

    def __neg__(self) -> "Self":
        return self.__class__.wrap(self._col.__neg__())


class _FloatOpsMixin(TypedColumn):
    __hash__ = TypedColumn.__hash__

    # -- comparison --
    def __eq__(self, other: "Self | int | float") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__eq__(_unwrap(other)))

    def __ne__(self, other: "Self | int | float") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__ne__(_unwrap(other)))

    def __lt__(self, other: "Self | int | float") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__lt__(_unwrap(other)))

    def __le__(self, other: "Self | int | float") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__le__(_unwrap(other)))

    def __gt__(self, other: "Self | int | float") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__gt__(_unwrap(other)))

    def __ge__(self, other: "Self | int | float") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__ge__(_unwrap(other)))

    # -- arithmetic --
    def __add__(self, other: "Self | int | float") -> "Self":
        return self.__class__.wrap(self._col.__add__(_unwrap(other)))

    def __sub__(self, other: "Self | int | float") -> "Self":
        return self.__class__.wrap(self._col.__sub__(_unwrap(other)))

    def __mul__(self, other: "Self | int | float") -> "Self":
        return self.__class__.wrap(self._col.__mul__(_unwrap(other)))

    def __truediv__(self, other: "Self | int | float") -> "Double":
        return Double.wrap(self._col.__truediv__(_unwrap(other)))

    def __mod__(self, other: "Self | int | float") -> "Self":
        return self.__class__.wrap(self._col.__mod__(_unwrap(other)))

    def __pow__(self, other: "Self | int | float") -> "Double":
        return Double.wrap(self._col.__pow__(_unwrap(other)))

    def __neg__(self) -> "Self":
        return self.__class__.wrap(self._col.__neg__())


class _DecimalOpsMixin(TypedColumn):
    __hash__ = TypedColumn.__hash__

    # -- comparison --
    def __eq__(self, other: "Self | int | float | _decimal.Decimal") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__eq__(_unwrap(other)))

    def __ne__(self, other: "Self | int | float | _decimal.Decimal") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__ne__(_unwrap(other)))

    def __lt__(self, other: "Self | int | float | _decimal.Decimal") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__lt__(_unwrap(other)))

    def __le__(self, other: "Self | int | float | _decimal.Decimal") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__le__(_unwrap(other)))

    def __gt__(self, other: "Self | int | float | _decimal.Decimal") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__gt__(_unwrap(other)))

    def __ge__(self, other: "Self | int | float | _decimal.Decimal") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__ge__(_unwrap(other)))

    # -- arithmetic --
    def __add__(self, other: "Self | int | float | _decimal.Decimal") -> "Self":
        return self.__class__.wrap(self._col.__add__(_unwrap(other)))

    def __sub__(self, other: "Self | int | float | _decimal.Decimal") -> "Self":
        return self.__class__.wrap(self._col.__sub__(_unwrap(other)))

    def __mul__(self, other: "Self | int | float | _decimal.Decimal") -> "Self":
        return self.__class__.wrap(self._col.__mul__(_unwrap(other)))

    def __truediv__(self, other: "Self | int | float | _decimal.Decimal") -> "Self":
        return self.__class__.wrap(self._col.__truediv__(_unwrap(other)))

    def __mod__(self, other: "Self | int | float | _decimal.Decimal") -> "Self":
        return self.__class__.wrap(self._col.__mod__(_unwrap(other)))

    def __pow__(self, other: "Self | int | float | _decimal.Decimal") -> "Double":
        return Double.wrap(self._col.__pow__(_unwrap(other)))

    def __neg__(self) -> "Self":
        return self.__class__.wrap(self._col.__neg__())


# ---------------------------------------------------------------------------
# Concrete primitive types
# ---------------------------------------------------------------------------


class String(_StringOpsMixin, TypedColumn[StringType]):
    _data_type = StringType


class Decimal(_DecimalOpsMixin, TypedColumn[DecimalType]):
    _data_type = DecimalType


class Binary(_BinaryOpsMixin, TypedColumn[BinaryType]):
    _data_type = BinaryType


class Integer(_IntOpsMixin, TypedColumn[IntegerType]):
    _data_type = IntegerType


class Long(_IntOpsMixin, TypedColumn[LongType]):
    _data_type = LongType


class Float(_FloatOpsMixin, TypedColumn[FloatType]):
    _data_type = FloatType


class Double(_FloatOpsMixin, TypedColumn[DoubleType]):
    _data_type = DoubleType


class Timestamp(_TimestampOpsMixin, TypedColumn[TimestampType]):
    _data_type = TimestampType


class Date(_DateOpsMixin, TypedColumn[DateType]):
    _data_type = DateType


Int = Integer
