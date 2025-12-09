import pytest
from pyspark.sql.types import DoubleType, StringType

import typespark as ts
from tests.conftest import Person
from typespark.aggregates import count, sum


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


def test_no_agg(person: Person):
    class Aggregate(ts.DataFrame):
        name: ts.String

    df = person.union(person)

    with pytest.raises(ValueError):
        Aggregate(df, name=df.name.group())


def test_cast(person: Person):
    class Aggregate(ts.DataFrame):
        name: ts.String
        n_rows: ts.Double

    df = person.union(person)

    tf = Aggregate(df, name=df.name.group(), n_rows=count(df.age).cast(DoubleType()))

    fields = tf.to_df().schema.fields

    assert isinstance(fields[0].dataType, StringType)
    assert isinstance(fields[1].dataType, DoubleType)
