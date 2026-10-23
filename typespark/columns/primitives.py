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
    TimestampNTZType,
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

    # Comparing with a StringOf gives a plain `bool`, which filter()/when()
    # reject: a plain String may hold values outside the literal set.
    @overload  # type: ignore[override]
    def __eq__(self, other: "StringOf[Any]") -> bool: ...  # type: ignore[overload-overlap]
    @overload
    def __eq__(self, other: "Self | str") -> Bool: ...
    def __eq__(self, other: "Self | str") -> Bool:  # type: ignore[misc]
        return Bool.wrap(self._col.__eq__(_unwrap(other)))

    @overload  # type: ignore[override]
    def __ne__(self, other: "StringOf[Any]") -> bool: ...  # type: ignore[overload-overlap]
    @overload
    def __ne__(self, other: "Self | str") -> Bool: ...
    def __ne__(self, other: "Self | str") -> Bool:  # type: ignore[misc]
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
            col = self._lit(col)
        super().__init__(col)

    @staticmethod
    def _lit(value: "datetime.datetime | None") -> pyspark.sql.Column:
        return lit(value)

    def _coerce(self, other: "Self | datetime.datetime") -> Any:
        if isinstance(other, datetime.datetime):
            return self._lit(other)
        return _unwrap(other)

    def __eq__(self, other: "Self | datetime.datetime") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__eq__(self._coerce(other)))

    def __ne__(self, other: "Self | datetime.datetime") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__ne__(self._coerce(other)))

    def __lt__(self, other: "Self | datetime.datetime") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__lt__(self._coerce(other)))

    def __le__(self, other: "Self | datetime.datetime") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__le__(self._coerce(other)))

    def __gt__(self, other: "Self | datetime.datetime") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__gt__(self._coerce(other)))

    def __ge__(self, other: "Self | datetime.datetime") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.__ge__(self._coerce(other)))

    def between(  # type: ignore[override]
        self,
        lowerBound: "Self | datetime.datetime",
        upperBound: "Self | datetime.datetime",
    ) -> Bool:
        return Bool.wrap(
            self._col.between(self._coerce(lowerBound), self._coerce(upperBound))
        )


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
    # Comparing with an IntOf gives a plain `bool`, which filter()/when()
    # reject: a plain integer column may hold values outside the literal set.
    @overload  # type: ignore[override]
    def __eq__(self, other: "IntOf[Any]") -> bool: ...  # type: ignore[overload-overlap]
    @overload
    def __eq__(self, other: "Self | int") -> Bool: ...
    def __eq__(self, other: "Self | int") -> Bool:  # type: ignore[misc]
        return Bool.wrap(self._col.__eq__(_unwrap(other)))

    @overload  # type: ignore[override]
    def __ne__(self, other: "IntOf[Any]") -> bool: ...  # type: ignore[overload-overlap]
    @overload
    def __ne__(self, other: "Self | int") -> Bool: ...
    def __ne__(self, other: "Self | int") -> Bool:  # type: ignore[misc]
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


class StringOf[T: str](String):
    """String column restricted to the literal values in ``T``.

    Use with ``Literal``, e.g. ``StringOf[Literal["open", "closed"]]``.
    The restriction is static only; values are not validated at runtime.
    """

    __hash__ = String.__hash__

    def __eq__(self, other: "T | StringOf[T]") -> Bool:  # type: ignore[override]
        return super().__eq__(other)

    def __ne__(self, other: "T | StringOf[T]") -> Bool:  # type: ignore[override]
        return super().__ne__(other)

    def isin(self, *cols: "T | StringOf[T]") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.isin(*(_unwrap(c) for c in cols)))


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


class IntOf[T: int](Integer):
    """Integer column restricted to the literal values in ``T``.

    Use with ``Literal``, e.g. ``IntOf[Literal[1, 2, 3]]``.
    The restriction is static only; values are not validated at runtime.
    Arithmetic results fall back to plain ``Integer``.
    """

    __hash__ = Integer.__hash__

    def __eq__(self, other: "T | IntOf[T]") -> Bool:  # type: ignore[override]
        return super().__eq__(other)

    def __ne__(self, other: "T | IntOf[T]") -> Bool:  # type: ignore[override]
        return super().__ne__(other)

    def isin(self, *cols: "T | IntOf[T]") -> Bool:  # type: ignore[override]
        return Bool.wrap(self._col.isin(*(_unwrap(c) for c in cols)))

    def __add__(self, other: "Integer | int") -> Integer:  # type: ignore[override]
        return Integer.wrap(self._col.__add__(_unwrap(other)))

    def __sub__(self, other: "Integer | int") -> Integer:  # type: ignore[override]
        return Integer.wrap(self._col.__sub__(_unwrap(other)))

    def __mul__(self, other: "Integer | int") -> Integer:  # type: ignore[override]
        return Integer.wrap(self._col.__mul__(_unwrap(other)))

    def __mod__(self, other: "Integer | int") -> Integer:  # type: ignore[override]
        return Integer.wrap(self._col.__mod__(_unwrap(other)))

    def __neg__(self) -> Integer:  # type: ignore[override]
        return Integer.wrap(self._col.__neg__())


class Long(_IntOpsMixin, TypedColumn[LongType]):
    _data_type = LongType


class Float(_FloatOpsMixin, TypedColumn[FloatType]):
    _data_type = FloatType


class Double(_FloatOpsMixin, TypedColumn[DoubleType]):
    _data_type = DoubleType


class Timestamp(_TimestampOpsMixin, TypedColumn[TimestampType]):
    _data_type = TimestampType


class TimestampNTZ(_TimestampOpsMixin, TypedColumn[TimestampNTZType]):
    _data_type = TimestampNTZType

    @staticmethod
    def _lit(value: "datetime.datetime | None") -> pyspark.sql.Column:
        # lit(datetime) yields a session-tz TimestampType, and casting that to
        # NTZ shifts the wall-clock value. Casting from a string does not.
        if value is None:
            return lit(None).cast(TimestampNTZType())
        return lit(value.isoformat(sep=" ")).cast(TimestampNTZType())


class Date(_DateOpsMixin, TypedColumn[DateType]):
    _data_type = DateType


Int = Integer
