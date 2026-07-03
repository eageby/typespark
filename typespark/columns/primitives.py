import datetime
import decimal as _decimal
from typing import Any, Self, overload

import pyspark.sql
from pyspark.sql.functions import lit
from pyspark.sql.types import (
    BinaryType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
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

    def __init__(self, col: "pyspark.sql.Column | TypedColumn[Any] | str | None"):
        if not isinstance(col, (TypedColumn, pyspark.sql.Column)):
            col = lit(col)
        super().__init__(col)

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

    def between(self, lowerBound: "Self | str", upperBound: "Self | str") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.between(_unwrap(lowerBound), _unwrap(upperBound)))


class _DateOpsMixin(TypedColumn):
    __hash__ = TypedColumn.__hash__

    def __init__(
        self, col: "pyspark.sql.Column | TypedColumn[Any] | datetime.date | None"
    ):
        if not isinstance(col, (TypedColumn, pyspark.sql.Column)):
            col = lit(col)
        super().__init__(col)

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

    def between(  # type: ignore[override]
        self, lowerBound: "Self | datetime.date", upperBound: "Self | datetime.date"
    ) -> Bool:
        return Bool.wrap(self._col.between(_unwrap(lowerBound), _unwrap(upperBound)))


class _TimestampOpsMixin(TypedColumn):
    __hash__ = TypedColumn.__hash__

    def __init__(
        self, col: "pyspark.sql.Column | TypedColumn[Any] | datetime.datetime | None"
    ):
        if not isinstance(col, (TypedColumn, pyspark.sql.Column)):
            col = lit(col)
        super().__init__(col)

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

    def between(  # type: ignore[override]
        self,
        lowerBound: "Self | datetime.datetime",
        upperBound: "Self | datetime.datetime",
    ) -> Bool:
        return Bool.wrap(self._col.between(_unwrap(lowerBound), _unwrap(upperBound)))


class _BinaryOpsMixin(TypedColumn):
    __hash__ = TypedColumn.__hash__

    def __init__(self, col: "pyspark.sql.Column | TypedColumn[Any] | bytes | None"):
        if not isinstance(col, (TypedColumn, pyspark.sql.Column)):
            col = lit(col)
        super().__init__(col)

    def __eq__(self, other: "Self | bytes") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__eq__(_unwrap(other)))

    def __ne__(self, other: "Self | bytes") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__ne__(_unwrap(other)))


# ---------------------------------------------------------------------------
# Numeric mixins (comparison + arithmetic + unary neg)
# ---------------------------------------------------------------------------


class _IntOpsMixin(TypedColumn):
    __hash__ = TypedColumn.__hash__

    def __init__(self, col: "pyspark.sql.Column | TypedColumn[Any] | int | None"):
        if not isinstance(col, (TypedColumn, pyspark.sql.Column)):
            col = lit(col)
        super().__init__(col)

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

    def between(self, lowerBound: "Self | int", upperBound: "Self | int") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.between(_unwrap(lowerBound), _unwrap(upperBound)))

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

    def __init__(
        self, col: "pyspark.sql.Column | TypedColumn[Any] | int | float | None"
    ):
        if not isinstance(col, (TypedColumn, pyspark.sql.Column)):
            col = lit(col)
        super().__init__(col)

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

    def between(  # type: ignore[override]
        self, lowerBound: "Self | int | float", upperBound: "Self | int | float"
    ) -> Bool:
        return Bool.wrap(self._col.between(_unwrap(lowerBound), _unwrap(upperBound)))

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

    def __init__(
        self,
        col: "pyspark.sql.Column | TypedColumn[Any] | int | float | _decimal.Decimal | None",
    ):
        if not isinstance(col, (TypedColumn, pyspark.sql.Column)):
            col = lit(col)
        super().__init__(col)

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

    def between(  # type: ignore[override]
        self,
        lowerBound: "Self | int | float | _decimal.Decimal",
        upperBound: "Self | int | float | _decimal.Decimal",
    ) -> Bool:
        return Bool.wrap(self._col.between(_unwrap(lowerBound), _unwrap(upperBound)))

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

    @overload
    def __init__(self, c: "Decimal") -> None:
        """Wrap an existing Decimal column without casting."""
        ...

    @overload
    def __init__(
        self,
        c: "pyspark.sql.Column | TypedColumn[Any] | int | float | _decimal.Decimal | None",
        precision: int = 10,
        scale: int = 0,
    ) -> None:
        """Cast to DecimalType(precision, scale). Accepts columns or Python literals."""
        ...

    def __init__(self, c, precision=None, scale=None):  # type: ignore[misc]
        if isinstance(c, (int, float, _decimal.Decimal)) or c is None:
            p, s = 10 if precision is None else precision, 0 if scale is None else scale
            self._col = lit(c).cast(DecimalType(p, s))
        elif precision is not None or scale is not None:
            p, s = 10 if precision is None else precision, 0 if scale is None else scale
            raw = c.to_spark() if isinstance(c, TypedColumn) else c
            self._col = raw.cast(DecimalType(p, s))
        else:
            self._col = c.to_spark() if isinstance(c, TypedColumn) else c


class Binary(_BinaryOpsMixin, TypedColumn[BinaryType]):
    _data_type = BinaryType


class Short(_IntOpsMixin, TypedColumn[ShortType]):
    _data_type = ShortType


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
