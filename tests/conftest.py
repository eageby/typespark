import pytest
from pyspark.sql import DataFrame, Row, SparkSession, types

from typespark import Int, String
from typespark.dataframe import BaseDataFrame


@pytest.fixture(scope="module")
def spark():
    """Creates a single Spark session for all tests."""
    session = (
        SparkSession.Builder().master("local").appName("pytest-pyspark").getOrCreate()
    )

    yield session  # Provide the session to tests

    session.stop()  # Cleanup after tests


@pytest.fixture(scope="function")
def dataframe(spark):
    """Creates a default dataframe."""
    data = [Row(name="Alice", age=30), Row(name="Bob", age=25)]

    yield spark.createDataFrame(data)


@pytest.fixture(scope="function")
def struct_dataframe(spark):
    """Creates a default dataframe."""
    data = [
        Row(struct={"name": "Alice", "age": 30}),
        Row(struct={"name": "Bob", "age": 25}),
    ]

    yield spark.createDataFrame(data)


@pytest.fixture(scope="function")
def array_dataframe(spark):
    """Creates a default dataframe."""
    data = [
        Row(elements=[1, 2]),
        Row(elements=[3, 4]),
    ]

    yield spark.createDataFrame(data)


@pytest.fixture(scope="function")
def array_struct_dataframe(spark):
    """Creates a default dataframe."""
    data = [
        Row(
            elements=[
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ]
        )
    ]

    yield spark.createDataFrame(
        data,
        schema=types.StructType(
            [
                types.StructField(
                    "elements",
                    types.ArrayType(
                        types.StructType(
                            [
                                types.StructField("name", types.StringType()),
                                types.StructField("age", types.IntegerType()),
                            ]
                        )
                    ),
                )
            ],
        ),
    )


class Person(BaseDataFrame):
    name: String
    age: Int


@pytest.fixture(scope="function")
def person(dataframe: DataFrame):
    yield Person.from_df(dataframe)


class Id(BaseDataFrame):
    value: Int


class Range(BaseDataFrame):
    id: Int


@pytest.fixture(scope="function")
def id(spark: SparkSession):
    yield Id.from_df(spark.createDataFrame([1, 2, 3, 3, 4], "int"))


@pytest.fixture(scope="function")
def small_range(spark: SparkSession):
    rf = spark.range(3)
    yield Range.from_df(rf.withColumn("id", rf["id"].cast(types.IntegerType())))
