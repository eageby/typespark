from pyspark.sql.types import StringType

from tests.utils import collect_values
from typespark import String
from typespark.dataframe import BaseDataFrame
from typespark.columns.array import TypedArrayType
from typespark.columns.struct import Struct
from typespark.metadata import field
from typespark.type_alias import Integer


def test_array(array_dataframe):
    class DataClass(BaseDataFrame):
        elements: TypedArrayType[Integer]

    df = DataClass.from_df(array_dataframe)

    array_values = collect_values(df.select(df.elements))

    assert len(array_values) == 2
    assert array_values[0]["elements"] == [1, 2]
    assert array_values[1]["elements"] == [3, 4]


def test_array_explode(array_dataframe):
    class DataClass(BaseDataFrame):
        elements: TypedArrayType[Integer]

    df = DataClass.from_df(array_dataframe)

    result = df.select(df.elements.explode())
    result_values = collect_values(result)

    assert len(result_values) == 4
    assert result_values[0]["elements"] == 1
    assert result_values[1]["elements"] == 2
    assert result_values[2]["elements"] == 3
    assert result_values[3]["elements"] == 4


def test_array_struct(array_struct_dataframe):
    class Nested(Struct):
        name_: String = field(df_alias="name")
        age: Integer

    class DataClass(BaseDataFrame):
        elements: TypedArrayType[Nested]

    df = DataClass.from_df(array_struct_dataframe)

    structs = df.elements.explode()

    struct_values = collect_values(df.select(structs.age))

    assert struct_values[0]["elements.age"] == 30
    assert struct_values[1]["elements.age"] == 25


def test_array_struct_multiple(array_struct_dataframe):
    class Nested(Struct):
        name_: String = field(df_alias="name")
        age: Integer

    class DataClass(BaseDataFrame):
        elements: TypedArrayType[Nested]

    df = DataClass.from_df(array_struct_dataframe)

    nested = df.elements.explode()
    struct_values = collect_values(df.select(nested.age, nested.name_))

    assert struct_values[0]["elements.age"] == 30
    assert struct_values[0]["elements.name"] == "Alice"
    assert struct_values[1]["elements.age"] == 25
    assert struct_values[1]["elements.name"] == "Bob"


def test_array_struct_to_new(array_struct_dataframe):
    class Nested(Struct):
        name_: String = field(df_alias="name")
        age: Integer

    class New(BaseDataFrame):
        n: String = field(df_alias="name")
        age: Integer

    class DataClass(BaseDataFrame):
        elements: TypedArrayType[Nested]

    df = DataClass.from_df(array_struct_dataframe)

    nested = df.elements.explode()

    result = New(df, n=nested.name_, age=nested.age)
    result_values = collect_values(result)

    assert result_values[0]["name"] == "Alice"
    assert result_values[0]["age"] == 30
    assert result_values[1]["name"] == "Bob"
    assert result_values[1]["age"] == 25


def test_array_struct_to_new_with_cast(array_struct_dataframe):
    class Nested(Struct):
        name_: String = field(df_alias="name")
        age: Integer

    class New(BaseDataFrame):
        n: String = field(df_alias="name")
        age: String

    class DataClass(BaseDataFrame):
        elements: TypedArrayType[Nested]

    df = DataClass.from_df(array_struct_dataframe)

    nested = df.elements.explode()

    result = New(df, n=nested.name_, age=nested.age.cast(StringType()))

    result_values = collect_values(result)

    assert result_values[0]["name"] == "Alice"
    assert result_values[0]["age"] == "30"  # After casting
    assert result_values[1]["name"] == "Bob"
    assert result_values[1]["age"] == "25"  # After casting
