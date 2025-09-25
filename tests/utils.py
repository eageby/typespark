from typing import Any

from pyspark.sql import Column

from typespark.columns import TypedColumn


def same_column(first: Column | TypedColumn[Any], second: Column | TypedColumn[Any]):
    return first._jc.equals(second._jc)  # type: ignore # pylint: disable=protected-access
