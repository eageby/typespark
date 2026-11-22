"""Features typespark supports only on Spark 4.0+ raise a clear error on 3.5."""

import datetime

import pytest
from pyspark.sql import SparkSession

import typespark as ts
import typespark.functions as tsf
from tests.conftest import SPARK4


class _Frame(ts.DataFrame):
    d: ts.Double
    s: ts.String
    a: ts.Array[ts.Integer]
    t: ts.Timestamp


@pytest.fixture
def frame(spark: SparkSession) -> _Frame:
    return _Frame.create([(1.5, "xx", [1, 2], datetime.datetime(2025, 1, 1))], spark)


SPARK4_ONLY = {
    "tsf.ceil with scale": lambda f: tsf.ceil(f.d, 1),
    "tsf.floor with scale": lambda f: tsf.floor(f.d, 1),
    "tsf.ltrim with trim": lambda f: tsf.ltrim(f.s, f.s),
    "tsf.rtrim with trim": lambda f: tsf.rtrim(f.s, f.s),
    "tsf.array_position": lambda f: tsf.array_position(f.a, ts.Integer(1)),
    "tsf.array_remove": lambda f: tsf.array_remove(f.a, ts.Integer(1)),
    "tsf.timestamp_add": lambda f: tsf.timestamp_add("DAY", ts.Integer(1), f.t),
}


@pytest.mark.skipif(SPARK4, reason="supported on Spark 4.0")
@pytest.mark.parametrize("feature", SPARK4_ONLY)
def test_spark4_feature_raises_before_spark4(frame: _Frame, feature: str):
    with pytest.raises(ts.UnsupportedSparkFeatureError, match=r"Spark >= 4\.0"):
        SPARK4_ONLY[feature](frame)


@pytest.mark.parametrize(
    "call",
    [
        lambda f: tsf.ceil(f.d),
        lambda f: tsf.floor(f.d),
        lambda f: tsf.ltrim(f.s),
        lambda f: tsf.rtrim(f.s),
    ],
    ids=["ceil", "floor", "ltrim", "rtrim"],
)
def test_supported_on_every_spark(frame: _Frame, call):
    """The forms without the 4.0-only argument work on 3.5 too."""
    frame.to_df().select(call(frame).to_spark()).collect()
