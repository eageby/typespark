from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import lit

from tests.utils import same_column
from typespark import Int, String
from typespark.base import BaseDataFrame
from typespark.columns import TypedArrayType
from typespark.metadata import field
from typespark.struct import Struct
from typespark.type_alias import Integer


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


def test_struct_access(struct_dataframe):
    # need to wrap columns with typedcolumn do avoid accessing alias function through name
    class Nested(Struct):
        age: String
        name_: String = field(df_alias="name")

    class DataClass(BaseDataFrame):
        struct: Nested

    df = DataClass.from_df(struct_dataframe)

    assert same_column(df.struct.name_, df._dataframe["struct"]["name"])
    assert same_column(df.struct.age, df._dataframe["struct"]["age"])


def test_struct_access_with_alias(struct_dataframe):
    class Nested(Struct):
        name_: String = field(df_alias="name")
        age: String

    class DataClass(BaseDataFrame):
        struct: Nested

    df = DataClass.from_df(struct_dataframe).alias("a")

    assert same_column(df.struct.name_, df._dataframe["struct"]["name"])
    assert same_column(df.struct.age, df._dataframe["struct"]["age"])


def test_struct_select(struct_dataframe):
    class Nested(Struct):
        name_: String = field(df_alias="name")
        age: String

    class DataClass(BaseDataFrame):
        struct: Nested

    df = DataClass.from_df(struct_dataframe).alias("a")

    df.select(df.struct.age, df.struct.name_).show()


def test_struct_select_with_alias(struct_dataframe):
    class Nested(Struct):
        name_: String = field(df_alias="name")
        age: String

    class DataClass(BaseDataFrame):
        struct: Nested

    df = DataClass.from_df(struct_dataframe)

    df.select(df.struct.age, df.struct.name_).show()

    # assertDataFrameEqual(df._dataframe, result._dataframe)


def test_array(array_dataframe):

    class DataClass(BaseDataFrame):
        elements: TypedArrayType[Integer]

    df = DataClass.from_df(array_dataframe)
    df.show()


def test_array_explode(array_dataframe):

    class DataClass(BaseDataFrame):
        elements: TypedArrayType[Integer]

    df = DataClass.from_df(array_dataframe)
    df.select(df.elements.explode()).show()


def test_array_struct(array_struct_dataframe):
    class Nested(Struct):
        name_: String = field(df_alias="name")
        age: String

    class DataClass(BaseDataFrame):
        elements: TypedArrayType[Nested]

    df = DataClass.from_df(array_struct_dataframe)

    structs = df.elements.explode()

    df.select(structs.age).show()


def test_array_struct_multiple(array_struct_dataframe):
    class Nested(Struct):
        name_: String = field(df_alias="name")
        age: String

    class DataClass(BaseDataFrame):
        elements: TypedArrayType[Nested]

    df = DataClass.from_df(array_struct_dataframe)

    nested = df.elements.explode()
    df.select(nested.age, nested.name_).show()


def test_array_struct_to_new(array_struct_dataframe):
    class Nested(Struct):
        name_: String = field(df_alias="name")
        age: String

    class New(BaseDataFrame):
        n: String = field(df_alias="name")
        age: String

    class DataClass(BaseDataFrame):
        elements: TypedArrayType[Nested]

    df = DataClass.from_df(array_struct_dataframe)

    nested = df.elements.explode()

    New(df, n=nested.name_, age=nested.age).show()


def test_array_struct_to_new_with_cast(array_struct_dataframe):
    class Nested(Struct):
        name_: String = field(df_alias="name")
        age: String

    class New(BaseDataFrame):
        n: String = field(df_alias="name")
        age: Integer

    class DataClass(BaseDataFrame):
        elements: TypedArrayType[Nested]

    df = DataClass.from_df(array_struct_dataframe)

    nested = df.elements.explode()

    New(df, n=nested.name_, age=nested.age.cast(IntegerType())).show()


def test_array_struct_multiple_with_normal_column(array_struct_dataframe):
    class Nested(Struct):
        name_: String = field(df_alias="name")
        age: String

    class DataClass(BaseDataFrame):
        elements: TypedArrayType[Nested]

    class DataClass2(DataClass):
        test: Integer

    df = DataClass2.from_df(array_struct_dataframe.withColumn("test", lit(1)))

    nested = df.elements.explode()
    df.select(nested.age, nested.name_, df.test).show()


def test_array_struct_to_new_with_aliased_normal_column(array_struct_dataframe):
    class Nested(Struct):
        name_: String = field(df_alias="name")
        age: String

    class DataClass(BaseDataFrame):
        elements: TypedArrayType[Nested]
        test: Integer = field(df_alias="test2")

    df = DataClass.from_df(array_struct_dataframe.withColumn("test2", lit(1)))

    class New(BaseDataFrame):
        n: String
        a: String
        t: Integer

    nested = df.elements.explode()
    New(df, a=nested.age, n=nested.name_, t=df.test).show()

