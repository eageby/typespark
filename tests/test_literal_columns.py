"""Tests for literal-restricted columns (enum support): StringOf and IntOf."""

import textwrap
from typing import Literal

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from tests.utils import collect_values
from typespark import Integer, IntOf, String, StringOf
from typespark.dataframe import BaseDataFrame


class Ticket(BaseDataFrame):
    id: String
    status: StringOf[Literal["open", "closed"]]
    priority: IntOf[Literal[1, 2, 3]]


@pytest.fixture
def tickets(spark: SparkSession) -> Ticket:
    data = [("1", "open", 1), ("2", "closed", 2), ("3", "open", 3)]
    return Ticket.from_df(
        spark.createDataFrame(data, "id string, status string, priority int")
    )


def _mypy_error_lines(tmp_path, code: str) -> tuple[list[int], str]:
    api = pytest.importorskip("mypy.api")
    source = tmp_path / "snippet.py"
    source.write_text(textwrap.dedent(code))
    stdout, _, _ = api.run([str(source), "--no-incremental", "--show-error-codes"])
    errors = [line for line in stdout.splitlines() if ": error:" in line]
    return sorted({int(line.split(":")[1]) for line in errors}), stdout


def test_schema():
    assert Ticket.generate_schema() == StructType(
        [
            StructField("id", StringType()),
            StructField("status", StringType()),
            StructField("priority", IntegerType()),
        ]
    )


def test_cast_returns_base_primitive(tickets: Ticket):
    assert type(tickets.id.cast(StringType())) is String
    assert type(tickets.id.cast(IntegerType())) is Integer


# ── StringOf ──────────────────────────────────────────────────────────────────


def test_string_of_column_type(tickets: Ticket):
    assert isinstance(tickets.status, StringOf)
    assert isinstance(tickets.status, String)


def test_string_of_eq(tickets: Ticket):
    assert len(collect_values(tickets.filter(tickets.status == "open"))) == 2


def test_string_of_ne(tickets: Ticket):
    assert len(collect_values(tickets.filter(tickets.status != "open"))) == 1


def test_string_of_isin(tickets: Ticket):
    result = tickets.filter(tickets.status.isin("open", "closed"))
    assert len(collect_values(result)) == 3


def test_string_of_mypy_rejects_values_outside_literal(tmp_path):
    # A bare `col == "bogus"` is not an error by itself: mypy falls back to
    # str.__eq__ and types it as `bool`. The error surfaces where a `Bool`
    # column is required, e.g. filter().
    lines, stdout = _mypy_error_lines(
        tmp_path,
        """
        from typing import Literal
        from typespark import StringOf
        from typespark.dataframe import BaseDataFrame

        class Ticket(BaseDataFrame):
            status: StringOf[Literal["open", "closed"]]

        def check(t: Ticket) -> None:
            t.filter(t.status == "open")
            t.filter(t.status != "closed")
            t.filter(t.status.isin("open", "closed"))
            t.filter(t.status == "bogus")
            t.filter(t.status != "bogus")
            t.filter(t.status.isin("open", "bogus"))
        """,
    )
    assert lines == [13, 14, 15], stdout


# ── IntOf ─────────────────────────────────────────────────────────────────────


def test_int_of_column_type(tickets: Ticket):
    assert isinstance(tickets.priority, IntOf)
    assert isinstance(tickets.priority, Integer)


def test_int_of_eq(tickets: Ticket):
    assert len(collect_values(tickets.filter(tickets.priority == 1))) == 1


def test_int_of_ne(tickets: Ticket):
    assert len(collect_values(tickets.filter(tickets.priority != 1))) == 2


def test_int_of_isin(tickets: Ticket):
    result = tickets.filter(tickets.priority.isin(1, 3))
    assert len(collect_values(result)) == 2


def test_int_of_ordering(tickets: Ticket):
    assert len(collect_values(tickets.filter(tickets.priority >= 2))) == 2


@pytest.mark.parametrize(
    "op",
    [
        lambda c: c + 1,
        lambda c: c - 1,
        lambda c: c * 2,
        lambda c: c % 2,
        lambda c: -c,
    ],
    ids=["add", "sub", "mul", "mod", "neg"],
)
def test_int_of_arithmetic_returns_integer(tickets: Ticket, op):
    assert type(op(tickets.priority)) is Integer


def test_int_of_mypy_rejects_values_outside_literal(tmp_path):
    lines, stdout = _mypy_error_lines(
        tmp_path,
        """
        from typing import Literal
        from typespark import Integer, IntOf
        from typespark.dataframe import BaseDataFrame

        class Ticket(BaseDataFrame):
            priority: IntOf[Literal[1, 2, 3]]

        def check(t: Ticket) -> None:
            t.filter(t.priority == 1)
            t.filter(t.priority != 2)
            t.filter(t.priority.isin(1, 3))
            t.filter(t.priority > 5)
            plus: Integer = t.priority + 1
            t.filter(t.priority == 4)
            t.filter(t.priority != 4)
            t.filter(t.priority.isin(1, 4))
            narrowed: IntOf[Literal[1, 2, 3]] = t.priority + 1
        """,
    )
    assert lines == [15, 16, 17, 18], stdout
