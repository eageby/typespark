# pyright: reportUnnecessaryTypeIgnoreComment=true

"""TimestampNTZ usage, checked with pyright by tests/test_typecheck.py.

Lines marked `# type: ignore` must produce an error.
"""

from pyspark.sql.types import TimestampNTZType

import typespark as ts
import typespark.functions as tsf


class Event(ts.DataFrame):
    raw: ts.String
    at: ts.TimestampNTZ


def cast_ntz(events: Event) -> ts.TimestampNTZ:
    return events.raw.cast(TimestampNTZType())


def parse(events: Event) -> ts.TimestampNTZ:
    return tsf.to_timestamp_ntz(events.raw)


def add_day(events: Event) -> ts.TimestampNTZ:
    return tsf.timestamp_add("DAY", ts.Int(1), events.at)


def mixed_up(events: Event) -> ts.Timestamp:
    return tsf.timestamp_add("DAY", ts.Int(1), events.at)  # type: ignore
