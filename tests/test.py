from pyspark.sql import DataFrame

from tests.utils import same_column
from typespark import Int, String
from typespark.base import BaseDataFrame
from typespark.metadata import field
from typespark.typed_dataframe import Struct

# from pyspark.testing import assertDataFrameEqual


class Person(BaseDataFrame):
    name: String
    age: Int


def test_wrap_dataframe(dataframe: DataFrame):

    wrapped = Person.from_df(dataframe)

    assert same_column(wrapped.name, dataframe["name"])
    assert same_column(wrapped.age, dataframe["age"])
    assert not same_column(wrapped.name, dataframe["age"])


def test_aliasing(dataframe):
    df = Person.from_df(dataframe)

    class A(BaseDataFrame):
        n: String
        age: Int

    a1 = A(df, n=df.name, age=df.age).alias("a1")
    a2 = A(df, n=df.name, age=df.age).alias("a2")

    result = a1.join(a2, a1.n == a2.n).select(a1.n, a2.age)
    # assertDataFrameEqual(df._dataframe, result._dataframe)


def test_aliasing_with_column_alias(dataframe):
    df = Person.from_df(dataframe)

    class A(BaseDataFrame):
        n: String = field(df_alias="n2")
        age: Int

    a1 = A(df, n=df.name, age=df.age).alias("a1")
    a2 = A(df, n=df.name, age=df.age).alias("a2")

    result = a1.join(a2, a1.n == a2.n).select(a1.n, a2.age)

    # assertDataFrameEqual(df._dataframe, result._dataframe)


def test_struct_access(struct_dataframe):
    # need to wrap columns with typedcolumn do avoid accessing alias function through name
    class Nested(Struct):
        name: String
        age: String

    class DataClass(BaseDataFrame):
        struct: Nested

    df = DataClass.from_df(struct_dataframe)

    assert same_column(df.struct.name, df._dataframe["struct"]["name"])
    assert same_column(df.struct.age, df._dataframe["struct"]["age"])


def test_struct_access_with_alias(struct_dataframe):
    class Nested(Struct):
        name_: String = field(df_alias="name")
        age: String

    class DataClass(BaseDataFrame):
        struct: Nested

    df = DataClass.from_df(struct_dataframe)

    assert same_column(df.struct.name_, df._dataframe["struct"]["name"])
    assert same_column(df.struct.age, df._dataframe["struct"]["age"])
