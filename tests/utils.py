from typing import Any

import pyspark.sql

from typespark.base import BaseDataFrame


def collect_values(df: pyspark.sql.DataFrame | BaseDataFrame):
    if isinstance(df, BaseDataFrame):
        df = df.to_df()
    return [row.asDict() for row in df.collect()]


def collect_column(df: pyspark.sql.DataFrame | BaseDataFrame, column: str) -> list[Any]:
    return [v[column] for v in collect_values(df)]
