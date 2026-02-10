import pytest

from tests.utils import collect_column
from typespark import BaseDataFrame, Integer, TypedColumn


class Data(BaseDataFrame):
    a: Integer
    b: Integer


@pytest.fixture
def df(spark):
    return Data.from_df(
        spark.createDataFrame(
            [
                (1, 2),
                (3, 4),
            ],
            ["a", "b"],
        )
    )


@pytest.mark.parametrize(
    "op, expected",
    [
        (lambda a, b: a + b, [3, 7]),
        (lambda a, b: a - b, [-1, -1]),
        (lambda a, b: a * b, [2, 12]),
        (lambda a, b: a / b, [0.5, 0.75]),
    ],
)
def test_arithmetic_ops(df: Data, op, expected):
    result = op(df.a, df.b)
    tf = df.select(result.alias("result"))

    result_values = collect_column(tf, "result")

    assert isinstance(result, TypedColumn)
    assert result_values == expected
