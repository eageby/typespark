from typing import cast

import pyspark.sql

from ..columns import TypedColumn


def _rewrap[T: TypedColumn](col: T, c: pyspark.sql.Column) -> T:
    """Wrap ``c`` as the same column class as ``col``.

    ``type(col).wrap`` loses ``T`` for mypy when its bound is a union (``Numeric``).
    """
    return cast(T, type(col).wrap(c))
