import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql import Window as SparkWindow
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType

import typespark as ts
import typespark.functions as tsf
from tests.utils import collect_values
from typespark.columns._groups import _AggregateMixin


class Score(ts.DataFrame):
    g: ts.String
    v: ts.Int
    name: ts.String


@pytest.fixture(scope="function")
def scores(spark: SparkSession):
    data = [
        Row(g="a", v=1, name="x"),
        Row(g="a", v=3, name="y"),
        Row(g="b", v=2, name="z"),
        Row(g="b", v=5, name="w"),
    ]
    yield Score.from_df(spark.createDataFrame(data, schema=Score.generate_schema()))


def _window(scores: Score) -> ts.WindowSpec:
    return ts.Window.partitionBy(scores.g).orderBy(scores.v.desc())


@pytest.mark.parametrize(
    "func, expected, spark_type",
    [
        (lambda s: tsf.row_number(), ts.Int, IntegerType),
        (lambda s: tsf.rank(), ts.Int, IntegerType),
        (lambda s: tsf.dense_rank(), ts.Int, IntegerType),
        (lambda s: tsf.ntile(2), ts.Int, IntegerType),
        (lambda s: tsf.percent_rank(), ts.Double, DoubleType),
        (lambda s: tsf.cume_dist(), ts.Double, DoubleType),
        (lambda s: tsf.lag(s.v), ts.Int, IntegerType),
        (lambda s: tsf.lead(s.name), ts.String, StringType),
        (lambda s: tsf.nth_value(s.v, 1), ts.Int, IntegerType),
        (lambda s: tsf.sum(s.v), ts.Long, LongType),
        (lambda s: tsf.max(s.v), ts.Int, IntegerType),
    ],
)
def test_over_return_types(scores: Score, func, expected, spark_type):
    result = func(scores).over(_window(scores))

    assert type(result) is expected
    actual = scores.to_df().select(result.to_spark().alias("r")).schema["r"].dataType
    assert isinstance(actual, spark_type)


def test_over_strips_aggregate_marker(scores: Score):
    agg = tsf.sum(scores.v)
    assert isinstance(agg, _AggregateMixin)

    assert not isinstance(agg.over(_window(scores)), _AggregateMixin)


def test_qualify_with_typed_window(scores: Score):
    result = scores.qualify(tsf.row_number().over(_window(scores)) == 1)

    values = sorted(collect_values(result), key=lambda r: r["g"])
    assert values == [{"g": "a", "v": 3, "name": "y"}, {"g": "b", "v": 5, "name": "w"}]
    assert isinstance(result, Score)


def test_windowed_aggregate_in_constructor_does_not_group(scores: Score):
    class Totals(ts.DataFrame):
        name: ts.String
        total: ts.Long

    w = ts.Window.partitionBy(scores.g)
    result = Totals(scores, name=scores.name, total=tsf.sum(scores.v).over(w))

    values = sorted(collect_values(result), key=lambda r: r["name"])
    assert values == [
        {"name": "w", "total": 7},
        {"name": "x", "total": 4},
        {"name": "y", "total": 4},
        {"name": "z", "total": 7},
    ]


def test_rows_between_running_total(scores: Score):
    class Running(ts.DataFrame):
        name: ts.String
        running: ts.Long

    w = (
        ts.Window.partitionBy(scores.g)
        .orderBy(scores.v)
        .rowsBetween(ts.Window.unboundedPreceding, ts.Window.currentRow)
    )
    result = Running(scores, name=scores.name, running=tsf.sum(scores.v).over(w))

    values = sorted(collect_values(result), key=lambda r: r["name"])
    assert values == [
        {"name": "w", "running": 7},
        {"name": "x", "running": 1},
        {"name": "y", "running": 4},
        {"name": "z", "running": 2},
    ]


def test_over_accepts_raw_spark_window(scores: Score):
    w = SparkWindow.partitionBy(F.col("g")).orderBy(F.col("v").desc())

    result = scores.qualify(tsf.row_number().over(w) == 1)

    assert sorted(r["v"] for r in collect_values(result)) == [3, 5]
