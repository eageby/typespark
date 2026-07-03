import pytest
from pyspark.sql import Row, SparkSession, Window
from pyspark.sql import functions as F

from tests.utils import collect_values
from typespark import Int, String
from typespark.dataframe import BaseDataFrame


class Score(BaseDataFrame):
    g: String
    v: Int


@pytest.fixture(scope="function")
def scores(spark: SparkSession):
    data = [
        Row(g="a", v=1),
        Row(g="a", v=3),
        Row(g="b", v=2),
        Row(g="b", v=5),
    ]
    yield Score.from_df(spark.createDataFrame(data))


def test_qualify_keeps_top_row_per_group(scores: Score):
    w = Window.partitionBy(F.col("g")).orderBy(F.col("v").desc())

    result = scores.qualify(F.row_number().over(w) == 1)

    values = sorted(collect_values(result), key=lambda r: r["g"])

    assert values == [{"g": "a", "v": 3}, {"g": "b", "v": 5}]


def test_qualify_drops_temp_column(scores: Score):
    w = Window.partitionBy(F.col("g")).orderBy(F.col("v").desc())

    result = scores.qualify(F.row_number().over(w) == 1)

    # Schema unchanged — the internal predicate column must not leak.
    assert result.to_df().columns == ["g", "v"]


def test_qualify_preserves_typed_class(scores: Score):
    w = Window.partitionBy(F.col("g")).orderBy(F.col("v").desc())

    result = scores.qualify(F.row_number().over(w) == 1)

    assert isinstance(result, Score)
