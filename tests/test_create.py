import attrs
import pandas
import pytest
from pyspark.sql import Row

from tests.conftest import Person
from tests.utils import collect_values
from typespark import Int, Integer, String
from typespark.columns.array import TypedArrayType
from typespark.columns.struct import Struct
from typespark.dataframe import BaseDataFrame
from typespark.metadata import field


class WithDefault(BaseDataFrame):
    name: String
    surname: String = field(factory=lambda: String("Doe"))
    age: Int


class Derived(BaseDataFrame):
    name: String
    age: Int
    next_age: Int = field(
        default=attrs.Factory(lambda self: self.age + 1, takes_self=True)
    )


class Aliased(BaseDataFrame):
    name_: String = field(df_alias="name")
    age: Int


class Member(Struct):
    handle: String
    age: Integer


class Team(BaseDataFrame):
    id: Integer
    lead: Member
    members: TypedArrayType[Member]


def test_create_from_tuples(spark):
    df = Person.create([("Alice", 30), ("Bob", 25)], spark)

    assert isinstance(df, Person)
    assert collect_values(df) == [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
    ]


def test_create_from_dicts(spark):
    df = Person.create([{"age": 30, "name": "Alice"}], spark)

    assert collect_values(df) == [{"name": "Alice", "age": 30}]


def test_create_from_rows(spark):
    df = Person.create([Row(age=30, name="Alice")], spark)

    assert collect_values(df) == [{"name": "Alice", "age": 30}]


def test_create_uses_active_session(spark):
    df = Person.create([("Alice", 30)])

    assert collect_values(df) == [{"name": "Alice", "age": 30}]


def test_create_schema_types(spark):
    df = Person.create([("Alice", 30)], spark)

    assert df.to_df().schema == Person.generate_schema()


def test_create_df_alias_field(spark):
    df = Aliased.create([{"name": "Alice", "age": 30}], spark)

    assert collect_values(df.select(df.name_)) == [{"name": "Alice"}]


def test_create_nested(spark):
    df = Team.create(
        [(1, {"handle": "Alice", "age": 30}, [{"handle": "Bob", "age": 25}])], spark
    )

    assert df.to_df().collect()[0]["members"][0]["handle"] == "Bob"
    assert df.to_df().collect()[0]["lead"]["age"] == 30


def test_create_from_pandas(spark):
    pdf = pandas.DataFrame({"age": [30, 25], "name": ["Alice", "Bob"]})

    df = Person.create(pdf, spark)

    assert collect_values(df) == [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
    ]


def test_create_from_pandas_omit_defaults(spark):
    pdf = pandas.DataFrame({"name": ["Alice"], "age": [30]})

    df = WithDefault.create(pdf, spark)

    assert collect_values(df) == [{"name": "Alice", "surname": "Doe", "age": 30}]


def test_create_empty(spark):
    df = Person.create([], spark)

    assert df.to_df().count() == 0
    assert df.to_df().schema == Person.generate_schema()


# ── Defaults ────────────────────────────────────────────────────────


def test_create_tuples_omit_defaults(spark):
    df = WithDefault.create([("Alice", 30)], spark)

    assert collect_values(df) == [{"name": "Alice", "surname": "Doe", "age": 30}]


def test_create_tuples_override_defaults(spark):
    df = WithDefault.create([("Alice", "Smith", 30)], spark)

    assert collect_values(df) == [{"name": "Alice", "surname": "Smith", "age": 30}]


def test_create_tuples_wrong_length(spark):
    with pytest.raises(ValueError, match="must have 3 values"):
        WithDefault.create([("Alice",)], spark)


def test_create_tuples_mixed_lengths(spark):
    with pytest.raises(ValueError, match="Got lengths"):
        WithDefault.create([("Alice", 30), ("Bob", "Smith", 25)], spark)


def test_create_dicts_omit_defaults(spark):
    df = WithDefault.create([{"name": "Alice", "age": 30}], spark)

    assert collect_values(df) == [{"name": "Alice", "surname": "Doe", "age": 30}]


def test_create_dicts_default_is_column_level(spark):
    """A key missing in one row of a provided column is null, not the default."""
    df = WithDefault.create(
        [
            {"name": "Alice", "surname": "Smith", "age": 30},
            {"name": "Bob", "age": 25},
        ],
        spark,
    )

    assert [r["surname"] for r in collect_values(df)] == ["Smith", None]


def test_create_takes_self_factory(spark):
    df = Derived.create([("Alice", 30), ("Bob", 25)], spark)

    assert [r["next_age"] for r in collect_values(df)] == [31, 26]


def test_create_takes_self_factory_overridden(spark):
    df = Derived.create([{"name": "Alice", "age": 30, "next_age": 99}], spark)

    assert collect_values(df) == [{"name": "Alice", "age": 30, "next_age": 99}]


def test_create_unknown_key(spark):
    with pytest.raises(ValueError, match="Unknown columns"):
        Person.create([{"name": "Alice", "age": 30, "bogus": 1}], spark)


def test_create_mixed_dict_and_tuple(spark):
    with pytest.raises(ValueError, match="Cannot mix"):
        Person.create([{"name": "Alice", "age": 30}, ("Bob", 25)], spark)
