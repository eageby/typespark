import pyspark.sql.functions as F

import typespark
from typespark.columns.columns import TypedColumn

from ._type_aliases import Numeric


def abs[T: Numeric](col: T) -> T:
    """Returns the absolute value of `col`.

    Wrapper for :func:`pyspark.sql.functions.abs`.
    """
    return TypedColumn(F.abs(col.to_spark()))  # type: ignore


def acos(col: Numeric) -> typespark.Double:
    """Returns the inverse cosine of `col` in radians.

    Wrapper for :func:`pyspark.sql.functions.acos`.
    """
    return typespark.Double(F.acos(col.to_spark()))


def acosh(col: Numeric) -> typespark.Double:
    """Returns the inverse hyperbolic cosine of `col`.

    Wrapper for :func:`pyspark.sql.functions.acosh`.
    """
    return typespark.Double(F.acosh(col.to_spark()))


def add_months(start: typespark.Date, months: typespark.Integer | int) -> typespark.Date:
    """Returns the date that is `months` months after `start`.

    Wrapper for :func:`pyspark.sql.functions.add_months`.
    """
    return typespark.Date(
        F.add_months(
            start.to_spark(),
            months.to_spark() if isinstance(months, typespark.Column) else months,
        )
    )


def asin(col: Numeric) -> typespark.Double:
    """Returns the inverse sine of `col` in radians.

    Wrapper for :func:`pyspark.sql.functions.asin`.
    """
    return typespark.Double(F.asin(col.to_spark()))


def asinh(col: Numeric) -> typespark.Double:
    """Returns the inverse hyperbolic sine of `col`.

    Wrapper for :func:`pyspark.sql.functions.asinh`.
    """
    return typespark.Double(F.asinh(col.to_spark()))


def atan(col: Numeric) -> typespark.Double:
    """Returns the inverse tangent of `col` in radians.

    Wrapper for :func:`pyspark.sql.functions.atan`.
    """
    return typespark.Double(F.atan(col.to_spark()))


def atan2(col1: Numeric, col2: Numeric) -> typespark.Double:
    """Returns the angle in radians between the positive x-axis and the point (col2, col1).

    Wrapper for :func:`pyspark.sql.functions.atan2`.
    """
    return typespark.Double(F.atan2(col1.to_spark(), col2.to_spark()))


def atanh(col: Numeric) -> typespark.Double:
    """Returns the inverse hyperbolic tangent of `col`.

    Wrapper for :func:`pyspark.sql.functions.atanh`.
    """
    return typespark.Double(F.atanh(col.to_spark()))


def bround(col: Numeric, scale: int = 0) -> typespark.Double:
    """Returns `col` rounded to `scale` decimal places using banker's rounding (round half to even).

    Wrapper for :func:`pyspark.sql.functions.bround`.
    """
    return typespark.Double(F.bround(col.to_spark(), scale))


def cbrt(col: Numeric) -> typespark.Double:
    """Returns the cube root of `col`.

    Wrapper for :func:`pyspark.sql.functions.cbrt`.
    """
    return typespark.Double(F.cbrt(col.to_spark()))


def ceil(col: Numeric, scale: typespark.Int | int | None = None) -> typespark.Column:
    """Returns the smallest integer not less than `col`. With `scale`, rounds up to that many decimal places.

    Wrapper for :func:`pyspark.sql.functions.ceil`.
    """
    return typespark.Column(
        F.ceil(
            col.to_spark(),
            scale.to_spark() if isinstance(scale, typespark.Column) else scale,
        )
    )


def cos(col: Numeric) -> typespark.Double:
    """Returns the cosine of `col` (angle in radians).

    Wrapper for :func:`pyspark.sql.functions.cos`.
    """
    return typespark.Double(F.cos(col.to_spark()))


def cosh(col: Numeric) -> typespark.Double:
    """Returns the hyperbolic cosine of `col`.

    Wrapper for :func:`pyspark.sql.functions.cosh`.
    """
    return typespark.Double(F.cosh(col.to_spark()))


def cot(col: Numeric) -> typespark.Double:
    """Returns the cotangent of `col` (angle in radians).

    Wrapper for :func:`pyspark.sql.functions.cot`.
    """
    return typespark.Double(F.cot(col.to_spark()))


def csc(col: Numeric) -> typespark.Double:
    """Returns the cosecant of `col` (angle in radians).

    Wrapper for :func:`pyspark.sql.functions.csc`.
    """
    return typespark.Double(F.csc(col.to_spark()))


def degrees(col: Numeric) -> typespark.Double:
    """Converts `col` from radians to degrees.

    Wrapper for :func:`pyspark.sql.functions.degrees`.
    """
    return typespark.Double(F.degrees(col.to_spark()))


def e() -> typespark.Double:
    """Returns Euler's number (e ≈ 2.718).

    Wrapper for :func:`pyspark.sql.functions.e`.
    """
    return typespark.Double(F.e())


def exp(col: Numeric) -> typespark.Double:
    """Returns e raised to the power of `col`.

    Wrapper for :func:`pyspark.sql.functions.exp`.
    """
    return typespark.Double(F.exp(col.to_spark()))


def expm1(col: Numeric) -> typespark.Double:
    """Returns e raised to the power of `col` minus 1 (more accurate than exp(col) - 1 near zero).

    Wrapper for :func:`pyspark.sql.functions.expm1`.
    """
    return typespark.Double(F.expm1(col.to_spark()))


def factorial(col: typespark.Integer) -> typespark.Long:
    """Returns the factorial of `col`.

    Wrapper for :func:`pyspark.sql.functions.factorial`.
    """
    return typespark.Long(F.factorial(col.to_spark()))


def floor(col: Numeric, scale: typespark.Int | int | None = None) -> typespark.Column:
    """Returns the largest integer not greater than `col`. With `scale`, rounds down to that many decimal places.

    Wrapper for :func:`pyspark.sql.functions.floor`.
    """
    return typespark.Column(
        F.floor(
            col.to_spark(),
            scale.to_spark() if isinstance(scale, typespark.Column) else scale,
        )
    )


def hypot(col1: Numeric, col2: Numeric) -> typespark.Double:
    """Returns the hypotenuse of a right triangle with legs `col1` and `col2`.

    Wrapper for :func:`pyspark.sql.functions.hypot`.
    """
    return typespark.Double(F.hypot(col1.to_spark(), col2.to_spark()))


def ln(col: Numeric) -> typespark.Double:
    """Returns the natural logarithm of `col`.

    Wrapper for :func:`pyspark.sql.functions.ln`.
    """
    return typespark.Double(F.ln(col.to_spark()))


def log(col: Numeric, base: float | None = None) -> typespark.Double:
    """Returns the logarithm of `col`. Uses natural log when `base` is omitted.

    Wrapper for :func:`pyspark.sql.functions.log`.
    """
    if base is not None:
        return typespark.Double(F.log(base, col.to_spark()))  # type: ignore[call-arg, arg-type]
    return typespark.Double(F.log(col.to_spark()))


def log10(col: Numeric) -> typespark.Double:
    """Returns the base-10 logarithm of `col`.

    Wrapper for :func:`pyspark.sql.functions.log10`.
    """
    return typespark.Double(F.log10(col.to_spark()))


def log1p(col: Numeric) -> typespark.Double:
    """Returns the natural logarithm of `col + 1` (more accurate than log(col + 1) near zero).

    Wrapper for :func:`pyspark.sql.functions.log1p`.
    """
    return typespark.Double(F.log1p(col.to_spark()))


def log2(col: Numeric) -> typespark.Double:
    """Returns the base-2 logarithm of `col`.

    Wrapper for :func:`pyspark.sql.functions.log2`.
    """
    return typespark.Double(F.log2(col.to_spark()))


def negate[T: Numeric](col: T) -> T:
    """Returns the negation of `col`.

    Wrapper for :func:`pyspark.sql.functions.negate`.
    """
    return typespark.Column(F.negate(col.to_spark()))  # type: ignore


def pi() -> typespark.Double:
    """Returns the value of π (pi ≈ 3.14159).

    Wrapper for :func:`pyspark.sql.functions.pi`.
    """
    return typespark.Double(F.pi())


def pmod[T: Numeric](dividend: T, divisor: T) -> T:
    """Returns the positive remainder of `dividend` divided by `divisor`.

    Wrapper for :func:`pyspark.sql.functions.pmod`.
    """
    return typespark.Column(F.pmod(dividend.to_spark(), divisor.to_spark()))  # type: ignore


def positive[T: Numeric](col: T) -> T:
    """Returns `col` unchanged (unary positive).

    Wrapper for :func:`pyspark.sql.functions.positive`.
    """
    return typespark.Column(F.positive(col.to_spark()))  # type: ignore


def pow(col: Numeric, other: Numeric | float) -> typespark.Double:
    """Returns `col` raised to the power of `other`.

    Wrapper for :func:`pyspark.sql.functions.pow`.
    """
    return typespark.Double(
        F.pow(
            col.to_spark(),
            other.to_spark() if isinstance(other, TypedColumn) else other,
        )
    )


def radians(col: Numeric) -> typespark.Double:
    """Converts `col` from degrees to radians.

    Wrapper for :func:`pyspark.sql.functions.radians`.
    """
    return typespark.Double(F.radians(col.to_spark()))


def rand(seed: int | None = None) -> typespark.Double:
    """Returns a uniformly distributed random value in [0, 1).

    Wrapper for :func:`pyspark.sql.functions.rand`.
    """
    return typespark.Double(F.rand(seed))


def randn(seed: int | None = None) -> typespark.Double:
    """Returns a random value drawn from the standard normal distribution.

    Wrapper for :func:`pyspark.sql.functions.randn`.
    """
    return typespark.Double(F.randn(seed))


def rint(col: Numeric) -> typespark.Double:
    """Returns the double value closest to `col` that is equal to a mathematical integer.

    Wrapper for :func:`pyspark.sql.functions.rint`.
    """
    return typespark.Double(F.rint(col.to_spark()))


def round(col: Numeric, scale: int = 0) -> typespark.Column:
    """Returns `col` rounded to `scale` decimal places using HALF_UP rounding.

    Wrapper for :func:`pyspark.sql.functions.round`.
    """
    return typespark.Column(F.round(col.to_spark(), scale))


def sec(col: Numeric) -> typespark.Double:
    """Returns the secant of `col` (angle in radians).

    Wrapper for :func:`pyspark.sql.functions.sec`.
    """
    return typespark.Double(F.sec(col.to_spark()))


def sign[T: Numeric](col: T) -> T:
    """Returns -1, 0, or 1 depending on the sign of `col`.

    Wrapper for :func:`pyspark.sql.functions.sign`.
    """
    return typespark.Column(F.sign(col.to_spark()))  # type: ignore


def sin(col: Numeric) -> typespark.Double:
    """Returns the sine of `col` (angle in radians).

    Wrapper for :func:`pyspark.sql.functions.sin`.
    """
    return typespark.Double(F.sin(col.to_spark()))


def sinh(col: Numeric) -> typespark.Double:
    """Returns the hyperbolic sine of `col`.

    Wrapper for :func:`pyspark.sql.functions.sinh`.
    """
    return typespark.Double(F.sinh(col.to_spark()))


def sqrt(col: Numeric) -> typespark.Double:
    """Returns the square root of `col`.

    Wrapper for :func:`pyspark.sql.functions.sqrt`.
    """
    return typespark.Double(F.sqrt(col.to_spark()))


def tan(col: Numeric) -> typespark.Double:
    """Returns the tangent of `col` (angle in radians).

    Wrapper for :func:`pyspark.sql.functions.tan`.
    """
    return typespark.Double(F.tan(col.to_spark()))


def tanh(col: Numeric) -> typespark.Double:
    """Returns the hyperbolic tangent of `col`.

    Wrapper for :func:`pyspark.sql.functions.tanh`.
    """
    return typespark.Double(F.tanh(col.to_spark()))


def width_bucket(v: Numeric, min: Numeric, max: Numeric, numBucket: typespark.Integer | int) -> typespark.Long:
    """Returns the bucket number for `v` partitioning [`min`, `max`] into `numBucket` equal-width buckets.

    Wrapper for :func:`pyspark.sql.functions.width_bucket`.
    """
    return typespark.Long(
        F.width_bucket(
            v.to_spark(),
            min.to_spark(),
            max.to_spark(),
            numBucket.to_spark() if isinstance(numBucket, TypedColumn) else numBucket,
        )
    )
