import json

import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F

from tests.conftest import Person
from tests.utils import collect_values
from typespark import Int, Integer, MissingColumnError, String
from typespark.columns.struct import Struct
from typespark.dataframe import BaseDataFrame
from typespark.metadata import field


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


def test_missing_struct_subfield_raises(spark: SparkSession):
    from pyspark.sql import Row
    from pyspark.sql.types import StringType
    from pyspark.sql.types import StructField as SF
    from pyspark.sql.types import StructType as ST

    schema = ST(
        [SF("address", ST([SF("street", StringType()), SF("zip", StringType())]))]
    )
    data = [Row(address=Row(street="Main St", zip="12345"))]
    df = spark.createDataFrame(data, schema=schema)

    class Address(Struct):
        street: String
        zip: String
        city: String  # not present in the DataFrame

    class DataClass(BaseDataFrame):
        address: Address

    with pytest.raises(MissingColumnError) as exc_info:
        DataClass.from_df(df)

    err = exc_info.value
    assert err.parent_field == "address"
    assert err.expected_column == "city"
    assert err.model is Address
    assert "city" in str(err)
    assert "address" in str(err)


def test_array_of_struct_from_json(spark: SparkSession):
    class Raw(BaseDataFrame):
        data: String

    class Item(Struct):
        id: Int
        name_: String = field(df_alias="name")

    class Normalized(BaseDataFrame):
        first_id: Int
        first_name: String

    rows = [
        {"data": json.dumps([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])}
    ]
    df = Raw.from_df(spark.createDataFrame(rows))

    import typespark as ts

    parsed = ts.ArrayOf(Item).from_json(df.data)
    first = parsed.getItem(0)

    result = Normalized(df, first_id=first.id, first_name=first.name_)
    values = collect_values(result)

    assert values[0]["first_id"] == 1
    assert values[0]["first_name"] == "Alice"


def test_struct_wrap(spark: SparkSession):
    class Address(Struct):
        street: String
        city: String

    class Container(BaseDataFrame):
        address: Address

    raw_df = spark.createDataFrame([Row(street="Main St", city="Springfield")])
    result_df = raw_df.select(F.struct(F.col("street"), F.col("city")).alias("address"))
    container = Container.from_df(result_df)

    assert isinstance(container.address, Address)
    assert isinstance(container.address.street, String)
    assert isinstance(container.address.city, String)

    values = collect_values(container)
    assert values[0]["address"]["street"] == "Main St"
    assert values[0]["address"]["city"] == "Springfield"


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
