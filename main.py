from typing import Annotated, TypeVar

import attrs
from attr import dataclass
from pyspark.sql import Row, SparkSession, types

from typespark import schema
from typespark.metadata import base, decimal
from typespark.typed_dataframe import Struct, TypedColumn, TypedDataFrame

data = [
    Row(name="Alice", age=30),
    Row(name="Alice", age=25),
]

spark = SparkSession.Builder().master("local").appName("pytest-pyspark").getOrCreate()
data = spark.createDataFrame(data)


class Nested(TypedDataFrame):
    name_: TypedColumn[types.StringType] = base(alias="name")
    age: TypedColumn[types.DecimalType] = decimal(10, 1)


Nested(data).show()
