import inspect
from typing import Callable, cast, overload

import typespark as ts
from pyspark.sql import Column
from pyspark.sql import functions as F


def filter[T: ts.Column](col: ts.Array[T], f: Callable[[T], ts.Bool]) -> ts.Array[T]:
    """Returns elements of array `col` for which `f` returns true.

    Wrapper for :func:`pyspark.sql.functions.filter`.
    """
    elem_type = col._require_elem_type()

    def _spark_compatible_function(x: Column) -> Column:
        return f(elem_type.wrap(x)).to_spark()

    return type(col).wrap(F.filter(col.to_spark(), _spark_compatible_function))


def aggregate[T: ts.Column, I: ts.Column](
    col: ts.Array[T], init: I, f: Callable[[I, T], I]
) -> I:
    """Reduces array `col` to a single value by applying `f(accumulator, element)`,
    starting from `init`.

    Wrapper for :func:`pyspark.sql.functions.aggregate`.
    """
    elem_type = col._require_elem_type()
    acc_type = type(init)

    def _spark_compatible_function(acc: Column, elem: Column) -> Column:
        return f(acc_type.wrap(acc), elem_type.wrap(elem)).to_spark()

    return type(init).wrap(
        F.aggregate(col.to_spark(), init.to_spark(), _spark_compatible_function)
    )


@overload
def transform[I: ts.Column, O: ts.Column](
    col: ts.Array[I], f: Callable[[I], O]
) -> ts.Array[O]: ...


@overload
def transform[I: ts.Column, O: ts.Column](
    col: ts.Array[I], f: Callable[[I, ts.Integer], O]
) -> ts.Array[O]: ...


def transform[I: ts.Column, O: ts.Column](
    col: ts.Array[I], f: Callable[[I], O] | Callable[[I, ts.Integer], O]
) -> ts.Array[O]:
    """Applies `f` to each element of array `col` and returns a new array of the results.

    `f` takes either the element, or the element and its 0-based index
    (``ts.Integer``).

    Wrapper for :func:`pyspark.sql.functions.transform`.
    """
    elem_type = col._require_elem_type()
    spark_function: Callable[..., Column]

    if len(inspect.signature(f).parameters) == 2:
        f2 = cast(Callable[[I, ts.Integer], O], f)

        output_elem_type = type(
            f2(
                elem_type.wrap(F.col("__probe__")),
                ts.Integer.wrap(F.col("__probe_i__")),
            )
        )

        def _with_index(x: Column, i: Column) -> Column:
            return f2(elem_type.wrap(x), ts.Integer.wrap(i)).to_spark()

        spark_function = _with_index

    else:
        f1 = cast(Callable[[I], O], f)
        output_elem_type = type(f1(elem_type.wrap(F.col("__probe__"))))

        def _without_index(x: Column) -> Column:
            return f1(elem_type.wrap(x)).to_spark()

        spark_function = _without_index

    return ts.ArrayOf(output_elem_type).wrap(
        F.transform(col.to_spark(), spark_function)
    )
