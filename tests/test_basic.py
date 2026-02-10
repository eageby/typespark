from tests.conftest import Id, Person, Range
from tests.utils import collect_values
from typespark import Int, String
from typespark.base import BaseDataFrame
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
