import warnings

import pytest
from pyspark.sql import Row, functions as F
from pyspark.sql import types
from pyspark.sql.types import StringType

from tests.utils import collect_values
from typespark import Integer, String
from typespark.columns.array import TypedArrayType
from typespark.columns.struct import Struct
from typespark.dataframe import BaseDataFrame
from typespark.metadata import field


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


@pytest.fixture(scope="function")
def id_array_struct_dataframe(spark):
    """A scalar column alongside an array of structs, for mixed selects."""
    schema = types.StructType(
        [
            types.StructField("id", types.IntegerType()),
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
            ),
        ]
    )

    data = [
        Row(
            id=1,
            elements=[
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ],
        )
    ]

    yield spark.createDataFrame(data, schema=schema)


class _Nested(Struct):
    name: String
    age: Integer


class _IdArray(BaseDataFrame):
    id: Integer
    elements: TypedArrayType[_Nested]


def test_generator_with_plain_column(id_array_struct_dataframe):
    """Generator column selected alongside an ordinary typed column."""
    df = _IdArray.from_df(id_array_struct_dataframe)
    nested = df.elements.explode()

    values = collect_values(df.select(nested.age, df.id))

    assert [v["elements.age"] for v in values] == [30, 25]
    assert [v["id"] for v in values] == [1, 1]


def test_generator_with_calculated_column(id_array_struct_dataframe):
    """Generator column alongside an aliased calculated column."""
    df = _IdArray.from_df(id_array_struct_dataframe)
    nested = df.elements.explode()

    values = collect_values(df.select(nested.age, (df.id + 1).alias("id_plus")))

    assert [v["elements.age"] for v in values] == [30, 25]
    assert [v["id_plus"] for v in values] == [2, 2]


def test_generator_with_unaliased_calculated_column(id_array_struct_dataframe):
    """Generator column alongside a calculated column with no explicit alias."""
    df = _IdArray.from_df(id_array_struct_dataframe)
    nested = df.elements.explode()

    result = df.select(nested.age, df.id + 1)
    values = collect_values(result)

    assert [v["elements.age"] for v in values] == [30, 25]
    calc_name = [c for c in result.to_spark().columns if c != "elements.age"][0]
    assert [v[calc_name] for v in values] == [2, 2]


def test_calculation_on_generator_column(id_array_struct_dataframe):
    """Arithmetic applied to the generator-derived column itself."""
    df = _IdArray.from_df(id_array_struct_dataframe)
    nested = df.elements.explode()

    values = collect_values(df.select((nested.age + 1).alias("age_plus")))

    assert [v["age_plus"] for v in values] == [31, 26]


def test_bare_generator_with_plain_column(id_array_struct_dataframe):
    """Bare generator (no DeferredColumn access) alongside another column."""
    df = _IdArray.from_df(id_array_struct_dataframe)

    values = collect_values(df.select(df.elements.explode(), df.id))

    assert [v["elements"] for v in values] == [
        Row(name="Alice", age=30),
        Row(name="Bob", age=25),
    ]
    assert [v["id"] for v in values] == [1, 1]


def test_generator_column_comparison(id_array_struct_dataframe):
    """Comparison on a generator column yields a usable boolean projection."""
    df = _IdArray.from_df(id_array_struct_dataframe)
    nested = df.elements.explode()

    values = collect_values(df.select((nested.age > 26).alias("is_older")))

    assert [v["is_older"] for v in values] == [True, False]


def test_bare_typed_array_type_no_error(spark):
    """Regression: bare TypedArrayType(col) must not raise AttributeError."""
    col = F.array(F.lit(1), F.lit(2))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        instance = TypedArrayType(col)
    assert instance is not None
    assert any(issubclass(warning.category, UserWarning) for warning in w), \
        "Expected a UserWarning when constructing bare TypedArrayType"
