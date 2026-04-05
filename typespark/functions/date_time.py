from typing import Literal

import pyspark.sql.functions as F

import typespark

from ._type_aliases import DateOrTimestamp


def add_months(start: typespark.Date, months: typespark.Integer | int) -> typespark.Date:
    """Returns the date `months` months after `start`.

    Wrapper for :func:`pyspark.sql.functions.add_months`.
    """
    return typespark.Date(
        F.add_months(
            start.to_spark(),
            months.to_spark() if isinstance(months, typespark.TypedColumn) else months,
        )
    )


def date_add(start: typespark.Date, days: typespark.Int) -> typespark.Date:
    """Returns the date `days` days after `start`.

    Wrapper for :func:`pyspark.sql.functions.date_add`.
    """
    return typespark.Date(F.date_add(start.to_spark(), days.to_spark()))


def date_diff(end: DateOrTimestamp, start: DateOrTimestamp) -> typespark.Int:
    """Returns the number of days from `start` to `end`.

    Wrapper for :func:`pyspark.sql.functions.date_diff`.
    """
    return typespark.Int(F.date_diff(end.to_spark(), start.to_spark()))


def date_format(col: DateOrTimestamp, format: str) -> typespark.String:
    """Formats `col` as a string using the given datetime `format` pattern.

    Wrapper for :func:`pyspark.sql.functions.date_format`.
    """
    return typespark.String(F.date_format(col.to_spark(), format))


def date_from_unix_date(days: typespark.Integer) -> typespark.Date:
    """Converts a number of days since 1970-01-01 to a date.

    Wrapper for :func:`pyspark.sql.functions.date_from_unix_date`.
    """
    return typespark.Date(F.date_from_unix_date(days.to_spark()))


def date_sub(start: typespark.Date, days: typespark.Integer | int) -> typespark.Date:
    """Returns the date `days` days before `start`.

    Wrapper for :func:`pyspark.sql.functions.date_sub`.
    """
    return typespark.Date(
        F.date_sub(
            start.to_spark(),
            days.to_spark() if isinstance(days, typespark.TypedColumn) else days,
        )
    )


def date_trunc(format: str, timestamp: typespark.Timestamp) -> typespark.Timestamp:
    """Truncates `timestamp` to the granularity specified by `format` (e.g. 'year', 'month', 'day').

    Wrapper for :func:`pyspark.sql.functions.date_trunc`.
    """
    return typespark.Timestamp(F.date_trunc(format, timestamp.to_spark()))


def datediff(end: typespark.Date, start: typespark.Date) -> typespark.Int:
    """Returns the number of days from `start` to `end`.

    Wrapper for :func:`pyspark.sql.functions.datediff`.
    """
    return typespark.Int(F.datediff(end.to_spark(), start.to_spark()))


def day(col: DateOrTimestamp) -> typespark.Int:
    """Returns the day of the month from `col`.

    Wrapper for :func:`pyspark.sql.functions.day`.
    """
    return typespark.Int(F.day(col.to_spark()))


def dayofmonth(col: DateOrTimestamp) -> typespark.Int:
    """Returns the day of the month from `col`.

    Wrapper for :func:`pyspark.sql.functions.dayofmonth`.
    """
    return typespark.Int(F.dayofmonth(col.to_spark()))


def dayofweek(col: DateOrTimestamp) -> typespark.Int:
    """Returns the day of the week from `col` (1 = Sunday, 7 = Saturday).

    Wrapper for :func:`pyspark.sql.functions.dayofweek`.
    """
    return typespark.Int(F.dayofweek(col.to_spark()))


def dayofyear(col: DateOrTimestamp) -> typespark.Int:
    """Returns the day of the year from `col` (1-366).

    Wrapper for :func:`pyspark.sql.functions.dayofyear`.
    """
    return typespark.Int(F.dayofyear(col.to_spark()))


def from_unixtime(timestamp: typespark.Long | typespark.Integer, format: str | None = None) -> typespark.String:
    """Converts a Unix epoch timestamp (seconds since 1970-01-01) to a formatted string.

    Wrapper for :func:`pyspark.sql.functions.from_unixtime`.
    """
    if format is not None:
        return typespark.String(F.from_unixtime(timestamp.to_spark(), format))
    return typespark.String(F.from_unixtime(timestamp.to_spark()))


def from_utc_timestamp(timestamp: typespark.Timestamp, tz: str | typespark.String) -> typespark.Timestamp:
    """Interprets `timestamp` as UTC and converts it to the given timezone `tz`.

    Wrapper for :func:`pyspark.sql.functions.from_utc_timestamp`.
    """
    return typespark.Timestamp(
        F.from_utc_timestamp(
            timestamp.to_spark(),
            tz.to_spark() if isinstance(tz, typespark.TypedColumn) else tz,
        )
    )


def hour(col: typespark.Timestamp) -> typespark.Int:
    """Returns the hour component of `col` (0-23).

    Wrapper for :func:`pyspark.sql.functions.hour`.
    """
    return typespark.Int(F.hour(col.to_spark()))


def last_day(date: typespark.Date) -> typespark.Date:
    """Returns the last day of the month that `date` belongs to.

    Wrapper for :func:`pyspark.sql.functions.last_day`.
    """
    return typespark.Date(F.last_day(date.to_spark()))


def make_date(
    year: typespark.Integer | int,
    month: typespark.Integer | int,
    day: typespark.Integer | int,
) -> typespark.Date:
    """Creates a date from `year`, `month`, and `day` components.

    Wrapper for :func:`pyspark.sql.functions.make_date`.
    """

    def _to_col(v: typespark.Integer | int):
        return v.to_spark() if isinstance(v, typespark.TypedColumn) else F.lit(v)

    return typespark.Date(F.make_date(_to_col(year), _to_col(month), _to_col(day)))


def minute(col: typespark.Timestamp) -> typespark.Int:
    """Returns the minute component of `col` (0-59).

    Wrapper for :func:`pyspark.sql.functions.minute`.
    """
    return typespark.Int(F.minute(col.to_spark()))


def month(col: DateOrTimestamp) -> typespark.Int:
    """Returns the month component of `col` (1-12).

    Wrapper for :func:`pyspark.sql.functions.month`.
    """
    return typespark.Int(F.month(col.to_spark()))


def months_between(
    date1: DateOrTimestamp,
    date2: DateOrTimestamp,
    roundOff: bool = True,
) -> typespark.Double:
    """Returns the number of months between `date1` and `date2`.
    Rounds to 8 decimal places when `roundOff` is True.

    Wrapper for :func:`pyspark.sql.functions.months_between`.
    """
    return typespark.Double(F.months_between(date1.to_spark(), date2.to_spark(), roundOff))


def next_day(date: typespark.Date, dayOfWeek: str) -> typespark.Date:
    """Returns the first date after `date` that falls on `dayOfWeek` (e.g. 'Monday').

    Wrapper for :func:`pyspark.sql.functions.next_day`.
    """
    return typespark.Date(F.next_day(date.to_spark(), dayOfWeek))


def now() -> typespark.Timestamp:
    """Returns the current timestamp at query start time.

    Wrapper for :func:`pyspark.sql.functions.now`.
    """
    return typespark.Timestamp(F.now())


def quarter(col: DateOrTimestamp) -> typespark.Int:
    """Returns the quarter of the year from `col` (1-4).

    Wrapper for :func:`pyspark.sql.functions.quarter`.
    """
    return typespark.Int(F.quarter(col.to_spark()))


def second(col: typespark.Timestamp) -> typespark.Int:
    """Returns the second component of `col` (0-59).

    Wrapper for :func:`pyspark.sql.functions.second`.
    """
    return typespark.Int(F.second(col.to_spark()))


def timestamp_add(unit: str, quantity: typespark.Integer, ts: typespark.Timestamp) -> typespark.Timestamp:
    """Adds `quantity` units to `ts`. `unit` is a string such as 'YEAR', 'MONTH', 'DAY', 'HOUR', etc.

    Wrapper for :func:`pyspark.sql.functions.timestamp_add`.
    """
    return typespark.Timestamp(F.timestamp_add(unit, quantity.to_spark(), ts.to_spark()))


def timestamp_diff(
    unit: Literal[
        "YEAR",
        "QUARTER",
        "MONTH",
        "WEEK",
        "DAY",
        "HOUR",
        "MINUTE",
        "SECOND",
        "MILLISECOND",
        "MICROSECOND",
    ],
    start: typespark.Timestamp,
    end: typespark.Timestamp,
) -> typespark.Int:
    """Returns the difference between `end` and `start` expressed in the given `unit`.

    Wrapper for :func:`pyspark.sql.functions.timestamp_diff`.
    """
    return typespark.Int(F.timestamp_diff(unit, start.to_spark(), end.to_spark()))


def timestamp_micros(col: typespark.Long) -> typespark.Timestamp:
    """Converts microseconds since 1970-01-01 00:00:00 UTC to a timestamp.

    Wrapper for :func:`pyspark.sql.functions.timestamp_micros`.
    """
    return typespark.Timestamp(F.timestamp_micros(col.to_spark()))


def timestamp_millis(col: typespark.Long) -> typespark.Timestamp:
    """Converts milliseconds since 1970-01-01 00:00:00 UTC to a timestamp.

    Wrapper for :func:`pyspark.sql.functions.timestamp_millis`.
    """
    return typespark.Timestamp(F.timestamp_millis(col.to_spark()))


def timestamp_seconds(col: typespark.Long | typespark.Double) -> typespark.Timestamp:
    """Converts seconds (with optional fractional part) since 1970-01-01 00:00:00 UTC to a timestamp.

    Wrapper for :func:`pyspark.sql.functions.timestamp_seconds`.
    """
    return typespark.Timestamp(F.timestamp_seconds(col.to_spark()))


def to_date(col: typespark.String, format: str | None = None) -> typespark.Date:
    """Parses `col` as a date using the optional `format` pattern.

    Wrapper for :func:`pyspark.sql.functions.to_date`.
    """
    return typespark.Date(F.to_date(col.to_spark(), format))


def to_timestamp(col: typespark.String | typespark.Date, format: str | None = None) -> typespark.Timestamp:
    """Parses `col` as a timestamp using the optional `format` pattern.

    Wrapper for :func:`pyspark.sql.functions.to_timestamp`.
    """
    if format is not None:
        return typespark.Timestamp(F.to_timestamp(col.to_spark(), format))
    return typespark.Timestamp(F.to_timestamp(col.to_spark()))


def to_utc_timestamp(timestamp: typespark.Timestamp, tz: str | typespark.String) -> typespark.Timestamp:
    """Interprets `timestamp` as being in timezone `tz` and converts it to UTC.

    Wrapper for :func:`pyspark.sql.functions.to_utc_timestamp`.
    """
    return typespark.Timestamp(
        F.to_utc_timestamp(
            timestamp.to_spark(),
            tz.to_spark() if isinstance(tz, typespark.TypedColumn) else tz,
        )
    )


def trunc(date: typespark.Date, format: str) -> typespark.Date:
    """Truncates `date` to the unit specified by `format` (e.g. 'year', 'month').

    Wrapper for :func:`pyspark.sql.functions.trunc`.
    """
    return typespark.Date(F.trunc(date.to_spark(), format))


def unix_date(date: typespark.Date) -> typespark.Int:
    """Returns the number of days since 1970-01-01 for the given `date`.

    Wrapper for :func:`pyspark.sql.functions.unix_date`.
    """
    return typespark.Int(F.unix_date(date.to_spark()))


def unix_micros(col: typespark.Timestamp) -> typespark.Long:
    """Returns the number of microseconds since 1970-01-01 00:00:00 UTC for `col`.

    Wrapper for :func:`pyspark.sql.functions.unix_micros`.
    """
    return typespark.Long(F.unix_micros(col.to_spark()))


def unix_millis(col: typespark.Timestamp) -> typespark.Long:
    """Returns the number of milliseconds since 1970-01-01 00:00:00 UTC for `col`.

    Wrapper for :func:`pyspark.sql.functions.unix_millis`.
    """
    return typespark.Long(F.unix_millis(col.to_spark()))


def unix_seconds(col: typespark.Timestamp) -> typespark.Long:
    """Returns the number of seconds since 1970-01-01 00:00:00 UTC for `col`.

    Wrapper for :func:`pyspark.sql.functions.unix_seconds`.
    """
    return typespark.Long(F.unix_seconds(col.to_spark()))


def unix_timestamp(
    timestamp: typespark.Timestamp | typespark.String | None = None,
    format: str | None = None,
) -> typespark.Long:
    """Returns the current Unix timestamp in seconds. Parses `timestamp` with optional `format`
    when provided.

    Wrapper for :func:`pyspark.sql.functions.unix_timestamp`.
    """
    if timestamp is not None and format is not None:
        return typespark.Long(F.unix_timestamp(timestamp.to_spark(), format))
    if timestamp is not None:
        return typespark.Long(F.unix_timestamp(timestamp.to_spark()))
    return typespark.Long(F.unix_timestamp())


def weekofyear(col: DateOrTimestamp) -> typespark.Int:
    """Returns the week of the year from `col` (1-53).

    Wrapper for :func:`pyspark.sql.functions.weekofyear`.
    """
    return typespark.Int(F.weekofyear(col.to_spark()))


def year(col: DateOrTimestamp) -> typespark.Int:
    """Returns the year component of `col`.

    Wrapper for :func:`pyspark.sql.functions.year`.
    """
    return typespark.Int(F.year(col.to_spark()))
