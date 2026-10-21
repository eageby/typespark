# pyright: reportUnnecessaryTypeIgnoreComment=true

"""Enum-valued when() usage, checked with pyright by tests/test_typecheck.py.

Lines marked `# type: ignore` must produce an error.
"""

from enum import IntEnum, StrEnum

import typespark as ts
import typespark.functions as tsf


class Status(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


class Priority(IntEnum):
    LOW = 1
    HIGH = 2


class Ticket(ts.DataFrame):
    age: ts.Int


class Labelled(ts.DataFrame):
    status: ts.StringOf[Status]
    priority: ts.IntOf[Priority]


def labelled(t: Ticket) -> Labelled:
    status = tsf.when(t.age > 10, Status.CLOSED).otherwise(Status.OPEN)
    priority = (
        tsf.when(t.age > 5, Priority.HIGH)
        .when(t.age > 1, Priority.LOW)
        .otherwise(Priority.LOW)
    )
    return Labelled(t, status=status, priority=priority)


def no_otherwise(t: Ticket) -> ts.StringOf[Status]:
    return tsf.when(t.age > 10, Status.CLOSED)


def chain_column(t: Ticket, prev: ts.StringOf[Status]) -> ts.StringOf[Status]:
    return tsf.when(t.age > 10, Status.CLOSED).otherwise(prev)


def compare(t: Ticket) -> ts.Bool:
    status = tsf.when(t.age > 10, Status.CLOSED).otherwise(Status.OPEN)
    return status == Status.OPEN


def compare_raw_string(t: Ticket) -> ts.Bool:
    status = tsf.when(t.age > 10, Status.CLOSED).otherwise(Status.OPEN)
    return status == "open"  # type: ignore


def mixed_enums(t: Ticket) -> None:
    tsf.when(t.age > 10, Status.CLOSED).otherwise(Priority.LOW)  # type: ignore


def str_into_int(t: Ticket) -> ts.IntOf[Priority]:
    return tsf.when(t.age > 10, Status.CLOSED)  # type: ignore
