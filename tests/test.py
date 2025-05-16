import attrs
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types

from typespark.metadata import base, decimal
from typespark.typed_dataframe import Descriptor, Struct, TypedColumn, TypedDataFrame

# from pyspark.testing import assertDataFrameEqual


def same_column(first: Column, second: Column):
    return first._jc.equals(second._jc)  # type: ignore # pylint: disable=protected-access


class People(TypedDataFrame):
    name: TypedColumn[types.StringType]
    age: TypedColumn[types.IntegerType]


def test_wrap_dataframe(dataframe: DataFrame):

    wrapped = People(dataframe)

    assert same_column(wrapped.name, dataframe["name"])
    assert same_column(wrapped.age, dataframe["age"])
    assert not same_column(wrapped.name, dataframe["age"])


def test_access(dataframe):
    people = People(dataframe)

    assert isinstance(people.name, TypedColumn)
    assert isinstance(People.name, Descriptor)


# def test_alias_type(dataframe):
#     p1 = People(dataframe).alias("p1")
#     assert not isinstance(p1, People)
#     assert isinstance(p1, Aliased)


""" def test_alias_join(dataframe):
    p1 = People(dataframe).alias("p1")
    p2 = People(dataframe).alias("p22")

    df = p1.join(p2, p1.name == p2.name).select(p1.name, p2.age)

    assertDataFrameEqual(p2, df)
 """


def test_metadata():

    class DataClass(TypedDataFrame):
        name: TypedColumn[types.StringType] = attrs.field(
            metadata={"test": 1},
        )
        age: TypedColumn[types.IntegerType]

    assert attrs.fields(DataClass)[0].metadata["test"] == 1


def test_alias():
    class DataClass(TypedDataFrame):
        name: TypedColumn[types.StringType] = base(alias="other")
        age: TypedColumn[types.IntegerType]

    assert attrs.fields(DataClass)[0].alias == "other"


def test_alias_access(dataframe: DataFrame):

    class DataClass(TypedDataFrame):
        name: TypedColumn[types.StringType] = base(alias="age")
        age: TypedColumn[types.IntegerType]

    wrapped = DataClass(dataframe)

    assert same_column(wrapped.name, dataframe["age"])
    assert not same_column(wrapped.name, dataframe["name"])


def test_struct_access(struct_dataframe):
    class Nested(Struct):
        name_: TypedColumn[types.StringType] = base(alias="name")
        age: TypedColumn[types.IntegerType]

    class DataClass(TypedDataFrame):
        struct: Nested

    df = DataClass(struct_dataframe)
    assert same_column(df.struct.name_, df["struct"]["name"])
    assert same_column(df.struct.age, df["struct"]["age"])
