import datetime

import pytest
from pyspark.sql import SparkSession

from tests.conftest import Id, Person, Range
from tests.utils import collect_column, collect_values
from typespark import DataFrame, Int, String, Timestamp
from typespark.dataframe import BaseDataFrame
from typespark.exceptions import MissingColumnError
from typespark.literals import string_literal
from typespark.metadata import field


def test_wrap_dataframe(dataframe):
    wrapped = Person.from_df(dataframe)

    wrapped_values = collect_values(wrapped)

    assert wrapped_values[0]["name"] == "Alice"
    assert wrapped_values[0]["age"] == 30
    assert wrapped_values[1]["name"] == "Bob"
    assert wrapped_values[1]["age"] == 25

    # Ensure no extra columns are present
    assert len(wrapped_values[0]) == 2


def test_select_dataframe(person: Person):
    df = person.select(person.name)

    wrapped_values = collect_values(df)

    assert wrapped_values[0]["name"] == "Alice"
    assert wrapped_values[1]["name"] == "Bob"

    # Ensure no extra columns are present
    assert len(wrapped_values[0]) == 1
    assert list(wrapped_values[0].keys()) == ["name"]


def test_aliasing(person: Person):
    class A(BaseDataFrame):
        n: String
        age: Int

    a1 = A(person, n=person.name, age=person.age).alias("a1")
    a2 = A(person, n=person.name, age=person.age).alias("a2")

    result = a1.join(a2, a1.n == a2.n).select(a1.n, a2.age)

    result_values = collect_values(result)

    assert result_values[0]["n"] == "Alice"
    assert result_values[0]["age"] == 30
    assert result_values[1]["n"] == "Bob"
    assert result_values[1]["age"] == 25


def test_aliasing_with_column_alias(person: Person):
    class A(BaseDataFrame):
        n: String = field(df_alias="n2")
        age: Int

    a1 = A(person, n=person.name, age=person.age).alias("a1")
    a2 = A(person, n=person.name, age=person.age).alias("a2")

    result = a1.join(a2, a1.n == a2.n).select(a1.n, a2.age)

    result_values = collect_values(result)

    assert result_values[0]["n2"] == "Alice"
    assert result_values[0]["age"] == 30
    assert result_values[1]["n2"] == "Bob"
    assert result_values[1]["age"] == 25


def test_broadcast(id: Id, small_range: Range):
    sm = small_range.broadcast()

    df = id.join(sm, id.value == sm.id)

    result = collect_values(df)

    assert result[0]["value"] == 1
    assert result[1]["value"] == 2
    assert result[0]["id"] == 1
    assert result[1]["id"] == 2


def test_leftsemi_filters_rows(id: Id, small_range: Range):
    # id has [1, 2, 3, 3, 4], small_range has [0, 1, 2]
    result = id.leftsemi(small_range, id.value == small_range.id)

    assert isinstance(result, Id)
    assert sorted(collect_column(result, "value")) == [1, 2]


def test_leftsemi_preserves_schema(person: Person):
    result = person.leftsemi(person, person.name == person.name)

    assert isinstance(result, Person)
    assert result.to_df().columns == ["name", "age"]


def test_leftsemi_no_match_returns_empty(id: Id, small_range: Range):
    empty_range = Range.from_df(small_range.to_df().filter("id > 100"))
    result = id.leftsemi(empty_range, id.value == empty_range.id)

    assert isinstance(result, Id)
    assert result.count() == 0


def test_leftanti_filters_rows(id: Id, small_range: Range):
    # id has [1, 2, 3, 3, 4], small_range has [0, 1, 2]
    result = id.leftanti(small_range, id.value == small_range.id)

    assert isinstance(result, Id)
    assert sorted(collect_column(result, "value")) == [3, 3, 4]


def test_leftanti_preserves_schema(person: Person):
    result = person.leftanti(person, person.name == person.name)

    assert isinstance(result, Person)
    assert result.to_df().columns == ["name", "age"]


def test_leftanti_no_match_returns_all(id: Id, small_range: Range):
    empty_range = Range.from_df(small_range.to_df().filter("id > 100"))
    result = id.leftanti(empty_range, id.value == empty_range.id)

    assert isinstance(result, Id)
    assert result.count() == 5


def test_with_watermark(spark: SparkSession):
    class Dates(DataFrame):
        date: Timestamp

    data = [
        (datetime.datetime(2025, 1, 1),),
        (datetime.datetime(2026, 4, 3),),
        (datetime.datetime(2028, 4, 3),),
    ]

    df = Dates.from_df(spark.createDataFrame(data, schema=Dates.generate_schema()))

    values = collect_column(df.withWatermark(df.date, "10 minutes"), "date")

    assert values == [
        datetime.datetime(2025, 1, 1),
        datetime.datetime(2026, 4, 3),
        datetime.datetime(2028, 4, 3),
    ]


def test_with_watermark_alias(spark: SparkSession):
    class Dates(DataFrame):
        date: Timestamp = field(df_alias="dt")

    data = [
        (datetime.datetime(2025, 1, 1),),
        (datetime.datetime(2026, 4, 3),),
        (datetime.datetime(2028, 4, 3),),
    ]

    df = Dates.from_df(spark.createDataFrame(data, schema=Dates.generate_schema()))

    values = collect_column(df.withWatermark(df.date, "10 minutes"), "dt")

    assert values == [
        datetime.datetime(2025, 1, 1),
        datetime.datetime(2026, 4, 3),
        datetime.datetime(2028, 4, 3),
    ]


# ── from_df: column validation and defaults ──────────────────────


def test_disable_select_keeps_extra_columns(dataframe):
    class MinimalPerson(BaseDataFrame):
        name: String

    df = MinimalPerson.from_df(dataframe, disable_select=True)

    assert df.to_df().columns == ["name", "age"]


def test_select_typed_subset(dataframe):
    class MinimalPerson(BaseDataFrame):
        name: String

    df = MinimalPerson.from_df(dataframe)

    assert df.to_df().columns == ["name"]


def test_from_df_missing_col(dataframe):
    class ExtendedPerson(BaseDataFrame):
        name: String
        age: Int
        surname: String

    with pytest.raises(MissingColumnError) as exc_info:
        ExtendedPerson.from_df(dataframe)

    assert exc_info.value.expected_column == "surname"
    assert exc_info.value.available_columns == ["name", "age"]


def test_from_df_missing_col_with_default(dataframe):
    class ExtendedPerson(BaseDataFrame):
        name: String
        age: Int
        surname: String = field(default=string_literal("Doe"))

    df = ExtendedPerson.from_df(dataframe)

    assert df.to_df().columns == ["name", "age", "surname"]
    assert collect_column(df, "surname") == ["Doe", "Doe"]


def test_from_df_default_any_order(dataframe):
    """Default field before non-default field — ordering is free with kw_only."""

    class Reordered(BaseDataFrame):
        name: String
        surname: String = field(default=string_literal("Doe"))
        age: Int

    df = Reordered.from_df(dataframe)

    assert set(df.to_df().columns) == {"name", "surname", "age"}
    assert collect_column(df, "surname") == ["Doe", "Doe"]
    assert collect_column(df, "age") == [30, 25]
