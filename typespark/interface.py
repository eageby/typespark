from typing import Protocol, Self, Union

from pyspark.sql import Column


class DataFrameInterface(Protocol):
    def select(self, *cols: Union[Column, str]) -> Self: ...
