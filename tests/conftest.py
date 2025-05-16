import pytest
from pyspark.sql import Row, SparkSession


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
        Row(struct={"name": "Alice", "age": 25}),
    ]

    yield spark.createDataFrame(data)
