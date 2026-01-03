import pyspark.sql

from typespark.base import BaseDataFrame


def collect_values(df: pyspark.sql.DataFrame | BaseDataFrame):
    if isinstance(df, BaseDataFrame):
        df = df.to_df()
    return [row.asDict() for row in df.collect()]
