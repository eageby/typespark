import json

from pyspark.sql import SparkSession

from tests.conftest import Person
from tests.utils import collect_values
from typespark import Int, String
from typespark.columns.struct import Struct
from typespark.dataframe import BaseDataFrame
from typespark.metadata import field
from typespark.type_alias import Integer


def test_struct_access(struct_dataframe):
    class Nested(Struct):
        age: String
        name_: String = field(df_alias="name")

    class DataClass(BaseDataFrame):
        struct: Nested

    df = DataClass.from_df(struct_dataframe)

    struct_values = [i["struct"] for i in collect_values(df)]

    assert struct_values[0]["name"] == "Alice"
    assert struct_values[0]["age"] == "30"
    assert struct_values[1]["name"] == "Bob"
    assert struct_values[1]["age"] == "25"


def test_struct_access_with_alias(struct_dataframe):
    class Nested(Struct):
        name_: String = field(df_alias="name")
        age: String

    class DataClass(BaseDataFrame):
        struct: Nested

    df = DataClass.from_df(struct_dataframe).alias("a")

    struct_values = [i["struct"] for i in collect_values(df)]

    assert struct_values[0]["name"] == "Alice"
    assert struct_values[0]["age"] == "30"
    assert struct_values[1]["name"] == "Bob"
    assert struct_values[1]["age"] == "25"


def test_struct_select(struct_dataframe):
    class Nested(Struct):
        name_: String = field(df_alias="name")
        age: String

    class DataClass(BaseDataFrame):
        struct: Nested

    df = DataClass.from_df(struct_dataframe).alias("a")

    struct_values = [i["struct"] for i in collect_values(df)]

    assert struct_values[0]["age"] == "30"
    assert struct_values[0]["name"] == "Alice"
    assert struct_values[1]["age"] == "25"
    assert struct_values[1]["name"] == "Bob"


def test_init_struct_in_struct(person: Person):
    class Nested(Struct):
        n: String
        age: Integer

    class Main(Struct):
        info: Nested

    class Data(BaseDataFrame):
        person: Main

    result = Data(
        person, person=Main(info=Nested(n=person.name, age=person.age))
    ).alias("r")

    result_values = collect_values(result)

    assert result_values[0]["person"]["info"]["n"] == "Alice"
    assert result_values[0]["person"]["info"]["age"] == 30
    assert result_values[1]["person"]["info"]["n"] == "Bob"
    assert result_values[1]["person"]["info"]["age"] == 25


def test_struct_init(person: Person):
    class PersonStruct(Struct):
        name_: String = field(df_alias="name")
        age: Integer

    class Container(BaseDataFrame):
        person: PersonStruct

    result = Container(
        person, person=PersonStruct(name_=person.name, age=person.age)
    ).alias("r")

    result_values = collect_values(result)

    assert result_values[0]["person"]["name"] == "Alice"
    assert result_values[0]["person"]["age"] == 30
    assert result_values[1]["person"]["name"] == "Bob"
    assert result_values[1]["person"]["age"] == 25


def test_struct_init_alias(person: Person):
    class PersonStruct(Struct):
        n: String = field(df_alias="name")
        a: Integer

    class Container(BaseDataFrame):
        person: PersonStruct

    result = Container(person, person=PersonStruct(n=person.name, a=person.age)).alias(
        "r"
    )

    result_values = collect_values(result)

    assert result_values[0]["person"]["name"] == "Alice"
    assert result_values[0]["person"]["a"] == 30
    assert result_values[1]["person"]["name"] == "Bob"
    assert result_values[1]["person"]["a"] == 25


def test_struct_from_json(spark: SparkSession):
    class Raw(BaseDataFrame):
        data: String

    class Data(Struct):
        id: Int
        name_: String = field(df_alias="name")

    class Normalized(BaseDataFrame):
        id: Int
        n: String

    data = [{"data": json.dumps({"id": 1, "name": 2})}]

    df = Raw.from_df(spark.createDataFrame(data))

    parsed = Data.from_json(df.data)
    tf = Normalized(df, id=parsed.id, n=parsed.name_)

    result_values = collect_values(tf)
    assert result_values[0]["id"] == 1
    assert result_values[0]["n"] == "2"
