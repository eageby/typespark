from typing import cast

import pyspark
import pyspark.sql

from .._exceptions import UnsupportedSparkFeatureError
from ..columns import TypedColumn


def _rewrap[T: TypedColumn](col: T, c: pyspark.sql.Column) -> T:
    """Wrap ``c`` as the same column class as ``col``.

    ``type(col).wrap`` loses ``T`` for mypy when its bound is a union (``Numeric``).
    """
    return cast(T, type(col).wrap(c))


def _spark_version() -> tuple[int, int]:
    major, minor = pyspark.__version__.split(".")[:2]
    return int(major), int(minor)


def _require_spark(min_version: str, feature: str) -> None:
    """Raise UnsupportedSparkFeatureError if pyspark is older than ``min_version``.

    For features PySpark 3.5 lacks; typespark doesn't emulate them there.
    """
    major, minor = (int(p) for p in min_version.split("."))
    if _spark_version() < (major, minor):
        raise UnsupportedSparkFeatureError(feature, min_spark=min_version)
