from typing import Any

import pyspark.sql
from pyspark.sql import Row

from typespark.base import BaseDataFrame


def collect_values(df: pyspark.sql.DataFrame | BaseDataFrame):
    if isinstance(df, BaseDataFrame):
        df = df.to_df()
    return [row.asDict() for row in df.collect()]


def collect_column(df: pyspark.sql.DataFrame | BaseDataFrame, column: str) -> list[Any]:
    return [v[column] for v in collect_values(df)]


def single_row(spark, **kwargs):
    """Create a single-row DataFrame from keyword column name/value pairs."""
    row = Row(**kwargs)
    return spark.createDataFrame([row])
