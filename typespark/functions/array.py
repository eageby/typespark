import pyspark.sql.functions as F

import typespark
from typespark.columns.array import ArrayOf, TypedArrayType as _TAT


def array[C: typespark.Column](
    *cols: C,
) -> typespark.Array[C]:
    """Creates an array column from the given columns.

    Wrapper for :func:`pyspark.sql.functions.array`.
    """
    return ArrayOf(cols[0].__class__).wrap(F.array([c.to_spark() for c in cols]))


def array_append[T: typespark.Column](
    col: typespark.Array[T], value: T
) -> typespark.Array[T]:
    """Appends `value` to the end of array `col`.

    Wrapper for :func:`pyspark.sql.functions.array_append`.
    """
    return col.__class__.wrap(F.array_append(col.to_spark(), value.to_spark()))


def array_compact[A: _TAT](col: A) -> A:
    """Removes null elements from array `col`.

    Wrapper for :func:`pyspark.sql.functions.array_compact`.
    """
    return col.__class__.wrap(F.array_compact(col.to_spark()))  # type: ignore


def array_contains[T: typespark.Column](
    col: typespark.Array[T], value: T
) -> typespark.Bool:
    """Returns true if array `col` contains `value`.

    Wrapper for :func:`pyspark.sql.functions.array_contains`.
    """
    return typespark.Bool.wrap(F.array_contains(col.to_spark(), value.to_spark()))


def array_distinct[A: _TAT](col: A) -> A:
    """Returns an array with duplicate elements removed from `col`.

    Wrapper for :func:`pyspark.sql.functions.array_distinct`.
    """
    return col.__class__.wrap(F.array_distinct(col.to_spark()))  # type: ignore


def array_except[A: _TAT](col1: A, col2: A) -> A:
    """Returns elements in `col1` that are not in `col2`, without duplicates.

    Wrapper for :func:`pyspark.sql.functions.array_except`.
    """
    return col1.__class__.wrap(F.array_except(col1.to_spark(), col2.to_spark()))  # type: ignore


def array_insert[T: typespark.TypedColumn](
    arr: typespark.Array[T], pos: typespark.Integer | int, value: T
) -> typespark.Array[T]:
    """Inserts `value` into array `arr` at position `pos` (1-based). Negative positions count from the end.

    Wrapper for :func:`pyspark.sql.functions.array_insert`.
    """
    return arr.__class__.wrap(  # type: ignore
        F.array_insert(
            arr.to_spark(),
            pos.to_spark() if isinstance(pos, typespark.TypedColumn) else pos,
            value.to_spark(),
        ),
    )


def array_intersect[A: _TAT](col1: A, col2: A) -> A:
    """Returns the intersection of `col1` and `col2`, without duplicates.

    Wrapper for :func:`pyspark.sql.functions.array_intersect`.
    """
    return col1.__class__.wrap(F.array_intersect(col1.to_spark(), col2.to_spark()))  # type: ignore


def array_join(
    col: typespark.Array[typespark.String],
    delimiter: str,
    nullReplacement: str | None = None,
) -> typespark.String:
    """Joins elements of string array `col` into a single string using `delimiter`.
    Replaces nulls with `nullReplacement` when provided.

    Wrapper for :func:`pyspark.sql.functions.array_join`.
    """
    return typespark.String.wrap(
        F.array_join(col.to_spark(), delimiter, nullReplacement)
    )


def array_max[T: typespark.TypedColumn](col: typespark.Array[T]) -> T:
    """Returns the maximum element in array `col`.

    Wrapper for :func:`pyspark.sql.functions.array_max`.
    """
    elem_cls = col._elem_type
    return elem_cls.wrap(F.array_max(col.to_spark()))  # type: ignore


def array_min[T: typespark.TypedColumn](col: typespark.Array[T]) -> T:
    """Returns the minimum element in array `col`.

    Wrapper for :func:`pyspark.sql.functions.array_min`.
    """
    elem_cls = col._elem_type
    return elem_cls.wrap(F.array_min(col.to_spark()))  # type: ignore


def array_position[T: typespark.TypedColumn](
    col: typespark.Array[T], value: T
) -> typespark.Long:
    """Returns the 1-based position of the first occurrence of `value` in array `col`, or 0 if not found.

    Wrapper for :func:`pyspark.sql.functions.array_position`.
    """
    return typespark.Long.wrap(F.array_position(col.to_spark(), value.to_spark()))


def array_prepend[T: typespark.TypedColumn](
    col: typespark.Array[T], value: T
) -> typespark.Array[T]:
    """Prepends `value` to the beginning of array `col`.

    Wrapper for :func:`pyspark.sql.functions.array_prepend`.
    """
    return col.__class__.wrap(F.array_prepend(col.to_spark(), value.to_spark()))  # type: ignore


def array_remove[T: typespark.TypedColumn](
    col: typespark.Array[T], element: T
) -> typespark.Array[T]:
    """Removes all occurrences of `element` from array `col`.

    Wrapper for :func:`pyspark.sql.functions.array_remove`.
    """
    return col.__class__.wrap(F.array_remove(col.to_spark(), element.to_spark()))  # type: ignore


def array_repeat[T: typespark.TypedColumn](
    col: T, count: typespark.Integer | int
) -> typespark.Array[T]:
    """Returns an array containing `col` repeated `count` times.

    Wrapper for :func:`pyspark.sql.functions.array_repeat`.
    """
    return ArrayOf(col.__class__).wrap(
        F.array_repeat(
            col.to_spark(),
            count.to_spark() if isinstance(count, typespark.TypedColumn) else count,
        )
    )


def array_size[T: typespark.TypedColumn](col: typespark.Array[T]) -> typespark.Int:
    """Returns the number of elements in array `col`, or -1 for null.

    Wrapper for :func:`pyspark.sql.functions.array_size`.
    """
    return typespark.Int.wrap(F.array_size(col.to_spark()))


def array_sort[A: _TAT](col: A) -> A:
    """Returns array `col` sorted in ascending order, with nulls last.

    Wrapper for :func:`pyspark.sql.functions.array_sort`.
    """
    return col.__class__.wrap(F.array_sort(col.to_spark()))  # type: ignore


def array_union[A: _TAT](col1: A, col2: A) -> A:
    """Returns the union of `col1` and `col2`, without duplicates.

    Wrapper for :func:`pyspark.sql.functions.array_union`.
    """
    return col1.__class__.wrap(F.array_union(col1.to_spark(), col2.to_spark()))  # type: ignore


def arrays_overlap[A: _TAT](a1: A, a2: A) -> typespark.Bool:
    """Returns true if arrays `a1` and `a2` share at least one non-null element.

    Wrapper for :func:`pyspark.sql.functions.arrays_overlap`.
    """
    return typespark.Bool.wrap(F.arrays_overlap(a1.to_spark(), a2.to_spark()))


def broadcast[T: typespark.DataFrame](df: T) -> T:
    """Marks `df` for broadcast join, hinting the optimizer to replicate it to all workers.

    Wrapper for :func:`pyspark.sql.functions.broadcast`.
    """
    return df.broadcast()


def cardinality(col: typespark.Array) -> typespark.Int:
    """Returns the number of elements in array `col`, or -1 for null.

    Wrapper for :func:`pyspark.sql.functions.cardinality`.
    """
    return typespark.Int.wrap(F.cardinality(col.to_spark()))


def collect_list[T: typespark.TypedColumn](col: T) -> typespark.Array[T]:
    """Aggregates values of `col` into an array, preserving duplicates and order.

    Wrapper for :func:`pyspark.sql.functions.collect_list`.
    """
    return ArrayOf(col.__class__).wrap(F.collect_list(col.to_spark()))


def collect_set[T: typespark.TypedColumn](col: T) -> typespark.Array[T]:
    """Aggregates unique values of `col` into an array.

    Wrapper for :func:`pyspark.sql.functions.collect_set`.
    """
    return ArrayOf(col.__class__).wrap(F.collect_set(col.to_spark()))


def element_at[T: typespark.TypedColumn](
    col: typespark.Array[T], extraction: typespark.Integer | int
) -> T:
    """Returns the element at position `extraction` in array `col` (1-based; negative counts from end).

    Wrapper for :func:`pyspark.sql.functions.element_at`.
    """
    elem_cls = col._elem_type
    return elem_cls.wrap(  # type: ignore
        F.element_at(
            col.to_spark(),
            extraction.to_spark()
            if isinstance(extraction, typespark.TypedColumn)
            else extraction,
        )
    )


def try_element_at[T: typespark.TypedColumn](
    col: typespark.Array[T], extraction: typespark.Integer | int
) -> T:
    """Returns the element at position `extraction` in array `col` (1-based; negative counts from end).
    Returns null if the index exceeds the array length, instead of raising an error.

    Wrapper for :func:`pyspark.sql.functions.try_element_at`.
    """
    elem_cls = col._elem_type
    return elem_cls.wrap(  # type: ignore
        F.try_element_at(
            col.to_spark(),
            extraction.to_spark()
            if isinstance(extraction, typespark.TypedColumn)
            else F.lit(extraction),
        )
    )


def explode[T: typespark.TypedColumn](col: typespark.Array[T]) -> T:
    """Explodes array `col` into separate rows. Rows with null or empty arrays are dropped.

    Wrapper for :func:`pyspark.sql.functions.explode`.
    """
    elem_cls = col._elem_type
    return elem_cls.wrap(F.explode(col.to_spark()))  # type: ignore


def explode_outer[T: typespark.TypedColumn](col: typespark.Array[T]) -> T:
    """Explodes array `col` into separate rows, preserving null/empty arrays as a single null row.

    Wrapper for :func:`pyspark.sql.functions.explode_outer`.
    """
    elem_cls = col._elem_type
    return elem_cls.wrap(F.explode_outer(col.to_spark()))  # type: ignore


def flatten[T: typespark.TypedColumn](
    col: typespark.Array[typespark.Array[T]],
) -> typespark.Array[T]:
    """Flattens a nested array `col` into a single-level array.

    Wrapper for :func:`pyspark.sql.functions.flatten`.
    """
    inner = col._elem_type
    elem = inner._elem_type if issubclass(inner, _TAT) else None
    result_cls = ArrayOf(elem) if elem is not None else typespark.TypedArrayType
    return result_cls.wrap(F.flatten(col.to_spark()))  # type: ignore


def sequence[T: typespark.TypedColumn](
    start: T, stop: T, step: T | None = None
) -> typespark.Array[T]:
    """Generates an array of values from `start` to `stop` (inclusive), incrementing by `step`.

    Wrapper for :func:`pyspark.sql.functions.sequence`.
    """
    return ArrayOf(start.__class__).wrap(
        F.sequence(
            start.to_spark(),
            stop.to_spark(),
            step.to_spark() if step is not None else None,
        )
    )


def shuffle[A: _TAT](col: A) -> A:
    """Returns a randomly shuffled copy of array `col`.

    Wrapper for :func:`pyspark.sql.functions.shuffle`.
    """
    return col.__class__.wrap(F.shuffle(col.to_spark()))  # type: ignore


def size(col: typespark.Array) -> typespark.Int:
    """Returns the number of elements in array `col`, or -1 for null.

    Wrapper for :func:`pyspark.sql.functions.size`.
    """
    return typespark.Int.wrap(F.size(col.to_spark()))


def slice[A: _TAT](
    col: A,
    start: typespark.Integer | int,
    length: typespark.Integer | int,
) -> A:
    """Returns a sub-array of `col` starting at `start` (1-based) with `length` elements.

    Wrapper for :func:`pyspark.sql.functions.slice`.
    """
    return col.__class__.wrap(  # type: ignore
        F.slice(
            col.to_spark(),
            start.to_spark() if isinstance(start, typespark.TypedColumn) else start,
            length.to_spark() if isinstance(length, typespark.TypedColumn) else length,
        ),
    )


def sort_array[A: _TAT](col: A, asc: bool = True) -> A:
    """Returns array `col` sorted in ascending order by default; set `asc=False` for descending.

    Wrapper for :func:`pyspark.sql.functions.sort_array`.
    """
    return col.__class__.wrap(F.sort_array(col.to_spark(), asc))  # type: ignore
