import pytest
from pyspark.sql.types import DoubleType, StringType

import typespark as ts
from tests.conftest import Person
from typespark.columns.groups import _FORBIDDEN_OPS, _GroupColumn
from typespark.functions.aggregates import count, sum


def test_total(person: Person):
    class Aggregate(ts.DataFrame):
        name: ts.String
        total: ts.Int

    df = person.union(person)

    tf = Aggregate(df, name=df.name.group(), total=sum(df.age))

    data = tf.to_df().collect()
    assert data[0]["name"] == "Bob" and data[0]["total"] == 50
    assert data[1]["name"] == "Alice" and data[1]["total"] == 60


def test_count(person: Person):
    class Aggregate(ts.DataFrame):
        name: ts.String
        n_rows: ts.Int

    df = person.union(person)
    tf = Aggregate(df, name=df.name.group(), n_rows=count(df.age))
    data = tf.to_df().collect()

    assert data[0]["n_rows"] == 2
    assert data[1]["n_rows"] == 2


def test_no_group(person: Person):
    class Aggregate(ts.DataFrame):
        n_rows: ts.Int

    df = person.union(person)
    tf = Aggregate(df, n_rows=count(df.age))

    assert tf.to_df().collect()[0]["n_rows"] == 4


def test_global_composite_aggregate(person: Person):
    class Aggregate(ts.DataFrame):
        total: ts.Int
        double_total: ts.Int

    df = person.union(person)
    tf = Aggregate(df, total=sum(df.age), double_total=sum(df.age) + sum(df.age))
    row = tf.to_df().collect()[0]

    assert row["total"] == 110  # (30+25)*2
    assert row["double_total"] == 220


def test_no_agg(person: Person):
    class Aggregate(ts.DataFrame):
        name: ts.String

    df = person.union(person)

    with pytest.raises(ValueError):
        Aggregate(df, name=df.name.group())


def test_rematerialization_type(person: Person):
    class Aggregate(ts.DataFrame):
        name: ts.String
        total: ts.Int

    df = person.union(person)
    tf = Aggregate(df, name=df.name.group(), total=sum(df.age))

    assert isinstance(tf.name, ts.String)
    assert not isinstance(tf.name, _GroupColumn)


def test_rematerialization_arithmetic(person: Person):
    class Aggregate(ts.DataFrame):
        name: ts.String
        total: ts.Int

    df = person.union(person)
    tf = Aggregate(df, name=df.name.group(), total=sum(df.age))

    result = tf.name + tf.name  # would raise TypeError without rematerialization
    assert result is not None


def test_composite_aggregate(person: Person):
    class Aggregate(ts.DataFrame):
        name: ts.String
        double_total: ts.Int

    df = person.union(person)
    tf = Aggregate(df, name=df.name.group(), double_total=sum(df.age) + sum(df.age))
    data = tf.to_df().collect()

    assert data[0]["double_total"] == 100  # Bob: 25+25+25+25
    assert data[1]["double_total"] == 120  # Alice: 30+30+30+30


@pytest.mark.parametrize("op", _FORBIDDEN_OPS)
def test_group_column_forbidden_ops(person: Person, op: str):
    col = person.name.group()
    assert isinstance(col, _GroupColumn)
    with pytest.raises(TypeError, match=f"`{op}`"):
        getattr(col, op)(col)


def test_cast(person: Person):
    class Aggregate(ts.DataFrame):
        name: ts.String
        n_rows: ts.Double

    df = person.union(person)

    tf = Aggregate(df, name=df.name.group(), n_rows=count(df.age).cast(DoubleType()))

    fields = tf.to_df().schema.fields

    assert isinstance(fields[0].dataType, StringType)
    assert isinstance(fields[1].dataType, DoubleType)
