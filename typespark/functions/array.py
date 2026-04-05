import pyspark.sql.functions as F
from pyspark.sql.types import DataType

import typespark


def array[T: DataType](
    *cols: typespark.Column[T],
) -> typespark.Array[typespark.Column[T]]:
    """Creates an array column from the given columns.

    Wrapper for :func:`pyspark.sql.functions.array`.
    """
    return typespark.Array(F.array([c.to_spark() for c in cols]))


def array_append[T: DataType](
    col: typespark.Array[typespark.Column[T]], value: typespark.Column[T]
) -> typespark.Array[typespark.Column[T]]:
    """Appends `value` to the end of array `col`.

    Wrapper for :func:`pyspark.sql.functions.array_append`.
    """
    return typespark.Array(F.array_append(col.to_spark(), value.to_spark()))


def array_compact[T: typespark.TypedColumn](
    col: typespark.Array[T],
) -> typespark.Array[T]:
    """Removes null elements from array `col`.

    Wrapper for :func:`pyspark.sql.functions.array_compact`.
    """
    return typespark.Array(F.array_compact(col.to_spark()), col._elem_type)  # type: ignore


def array_contains[T: DataType](
    col: typespark.Array[typespark.Column[T]], value: typespark.Column[T]
) -> typespark.Column:
    """Returns true if array `col` contains `value`.

    Wrapper for :func:`pyspark.sql.functions.array_contains`.
    """
    return typespark.Column(F.array_contains(col.to_spark(), value.to_spark()))


def array_distinct[T: typespark.TypedColumn](
    col: typespark.Array[T],
) -> typespark.Array[T]:
    """Returns an array with duplicate elements removed from `col`.

    Wrapper for :func:`pyspark.sql.functions.array_distinct`.
    """
    return typespark.Array(F.array_distinct(col.to_spark()), col._elem_type)  # type: ignore


def array_except[T: typespark.TypedColumn](col1: typespark.Array[T], col2: typespark.Array[T]) -> typespark.Array[T]:
    """Returns elements in `col1` that are not in `col2`, without duplicates.

    Wrapper for :func:`pyspark.sql.functions.array_except`.
    """
    return typespark.Array(F.array_except(col1.to_spark(), col2.to_spark()), col1._elem_type)  # type: ignore


def array_insert[T: typespark.TypedColumn](
    arr: typespark.Array[T], pos: typespark.Integer | int, value: T
) -> typespark.Array[T]:
    """Inserts `value` into array `arr` at position `pos` (1-based). Negative positions count from the end.

    Wrapper for :func:`pyspark.sql.functions.array_insert`.
    """
    return typespark.Array(  # type: ignore
        F.array_insert(
            arr.to_spark(),
            pos.to_spark() if isinstance(pos, typespark.TypedColumn) else pos,
            value.to_spark(),
        ),
        arr._elem_type,
    )


def array_intersect[T: typespark.TypedColumn](col1: typespark.Array[T], col2: typespark.Array[T]) -> typespark.Array[T]:
    """Returns the intersection of `col1` and `col2`, without duplicates.

    Wrapper for :func:`pyspark.sql.functions.array_intersect`.
    """
    return typespark.Array(F.array_intersect(col1.to_spark(), col2.to_spark()), col1._elem_type)  # type: ignore


def array_join(
    col: typespark.Array[typespark.String],
    delimiter: str,
    nullReplacement: str | None = None,
) -> typespark.String:
    """Joins elements of string array `col` into a single string using `delimiter`.
    Replaces nulls with `nullReplacement` when provided.

    Wrapper for :func:`pyspark.sql.functions.array_join`.
    """
    return typespark.String(F.array_join(col.to_spark(), delimiter, nullReplacement))


def array_max[T: typespark.TypedColumn](col: typespark.Array[T]) -> T:
    """Returns the maximum element in array `col`.

    Wrapper for :func:`pyspark.sql.functions.array_max`.
    """
    return typespark.Column(F.array_max(col.to_spark()))  # type: ignore


def array_min[T: typespark.TypedColumn](col: typespark.Array[T]) -> T:
    """Returns the minimum element in array `col`.

    Wrapper for :func:`pyspark.sql.functions.array_min`.
    """
    return typespark.Column(F.array_min(col.to_spark()))  # type: ignore


def array_position[T: typespark.TypedColumn](col: typespark.Array[T], value: T) -> typespark.Long:
    """Returns the 1-based position of the first occurrence of `value` in array `col`, or 0 if not found.

    Wrapper for :func:`pyspark.sql.functions.array_position`.
    """
    return typespark.Long(F.array_position(col.to_spark(), value.to_spark()))


def array_prepend[T: typespark.TypedColumn](col: typespark.Array[T], value: T) -> typespark.Array[T]:
    """Prepends `value` to the beginning of array `col`.

    Wrapper for :func:`pyspark.sql.functions.array_prepend`.
    """
    return typespark.Array(F.array_prepend(col.to_spark(), value.to_spark()), col._elem_type)  # type: ignore


def array_remove[T: typespark.TypedColumn](col: typespark.Array[T], element: T) -> typespark.Array[T]:
    """Removes all occurrences of `element` from array `col`.

    Wrapper for :func:`pyspark.sql.functions.array_remove`.
    """
    return typespark.Array(F.array_remove(col.to_spark(), element.to_spark()), col._elem_type)  # type: ignore


def array_repeat[T: typespark.TypedColumn](col: T, count: typespark.Integer | int) -> typespark.Array[T]:
    """Returns an array containing `col` repeated `count` times.

    Wrapper for :func:`pyspark.sql.functions.array_repeat`.
    """
    return typespark.Array(  # type: ignore
        F.array_repeat(
            col.to_spark(),
            count.to_spark() if isinstance(count, typespark.TypedColumn) else count,
        )
    )


def array_size[T: typespark.TypedColumn](col: typespark.Array[T]) -> typespark.Int:
    """Returns the number of elements in array `col`, or -1 for null.

    Wrapper for :func:`pyspark.sql.functions.array_size`.
    """
    return typespark.Int(F.array_size(col.to_spark()))


def array_sort[T: typespark.TypedColumn](col: typespark.Array[T]) -> typespark.Array[T]:
    """Returns array `col` sorted in ascending order, with nulls last.

    Wrapper for :func:`pyspark.sql.functions.array_sort`.
    """
    return typespark.Array(F.array_sort(col.to_spark()), col._elem_type)  # type: ignore


def array_union[T: typespark.TypedColumn](col1: typespark.Array[T], col2: typespark.Array[T]) -> typespark.Array[T]:
    """Returns the union of `col1` and `col2`, without duplicates.

    Wrapper for :func:`pyspark.sql.functions.array_union`.
    """
    return typespark.Array(F.array_union(col1.to_spark(), col2.to_spark()), col1._elem_type)  # type: ignore


def arrays_overlap[T: typespark.TypedColumn](a1: typespark.Array[T], a2: typespark.Array[T]) -> typespark.Bool:
    """Returns true if arrays `a1` and `a2` share at least one non-null element.

    Wrapper for :func:`pyspark.sql.functions.arrays_overlap`.
    """
    return typespark.Bool(F.arrays_overlap(a1.to_spark(), a2.to_spark()))


def broadcast[T: typespark.DataFrame](df: T) -> T:
    """Marks `df` for broadcast join, hinting the optimizer to replicate it to all workers.

    Wrapper for :func:`pyspark.sql.functions.broadcast`.
    """
    return df.broadcast()


def cardinality(col: typespark.Array) -> typespark.Int:
    """Returns the number of elements in array `col`.

    Wrapper for :func:`pyspark.sql.functions.cardinality`.
    """
    return typespark.Int(F.cardinality(col.to_spark()))


def coalesce[T: DataType](*cols: typespark.Column[T]) -> typespark.Column[T]:
    """Returns the first non-null value from the given columns.

    Wrapper for :func:`pyspark.sql.functions.coalesce`.
    """
    return typespark.Column(F.coalesce(*[c.to_spark() for c in cols]))


def collect_list[T: DataType](
    col: typespark.Column[T],
) -> typespark.Array[typespark.Column[T]]:
    """Aggregates values of `col` into an array, preserving duplicates and order.

    Wrapper for :func:`pyspark.sql.functions.collect_list`.
    """
    return typespark.Array(F.collect_list(col.to_spark()))


def collect_set[T: DataType](
    col: typespark.Column[T],
) -> typespark.Array[typespark.Column[T]]:
    """Aggregates unique values of `col` into an array.

    Wrapper for :func:`pyspark.sql.functions.collect_set`.
    """
    return typespark.Array(F.collect_set(col.to_spark()))


def element_at[T: typespark.TypedColumn](col: typespark.Array[T], extraction: typespark.Integer | int) -> T:
    """Returns the element at position `extraction` in array `col` (1-based; negative counts from end).

    Wrapper for :func:`pyspark.sql.functions.element_at`.
    """
    return typespark.Column(  # type: ignore
        F.element_at(
            col.to_spark(),
            extraction.to_spark() if isinstance(extraction, typespark.TypedColumn) else extraction,
        )
    )


def explode[T: DataType](
    col: typespark.Array[typespark.Column[T]],
) -> typespark.Column[T]:
    """Explodes array `col` into separate rows. Rows with null or empty arrays are dropped.

    Wrapper for :func:`pyspark.sql.functions.explode`.
    """
    return typespark.Column[T](F.explode(col.to_spark()))


def explode_outer[T: DataType](
    col: typespark.Array[typespark.Column[T]],
) -> typespark.Column[T]:
    """Explodes array `col` into separate rows, preserving null/empty arrays as a single null row.

    Wrapper for :func:`pyspark.sql.functions.explode_outer`.
    """
    return typespark.Column[T](F.explode_outer(col.to_spark()))


def flatten[T: typespark.TypedColumn](
    col: typespark.Array[typespark.Array[T]],
) -> typespark.Array[T]:
    """Flattens a nested array `col` into a single-level array.

    Wrapper for :func:`pyspark.sql.functions.flatten`.
    """
    return typespark.Array(F.flatten(col.to_spark()))  # type: ignore


def sequence[T: typespark.TypedColumn](start: T, stop: T, step: T | None = None) -> typespark.Array[T]:
    """Generates an array of values from `start` to `stop` (inclusive), incrementing by `step`.

    Wrapper for :func:`pyspark.sql.functions.sequence`.
    """
    return typespark.Array(  # type: ignore
        F.sequence(
            start.to_spark(),
            stop.to_spark(),
            step.to_spark() if step is not None else None,
        )
    )


def shuffle[T: typespark.TypedColumn](col: typespark.Array[T]) -> typespark.Array[T]:
    """Returns a randomly shuffled copy of array `col`.

    Wrapper for :func:`pyspark.sql.functions.shuffle`.
    """
    return typespark.Array(F.shuffle(col.to_spark()), col._elem_type)  # type: ignore


def size(col: typespark.Array) -> typespark.Int:
    """Returns the number of elements in array `col`, or -1 for null.

    Wrapper for :func:`pyspark.sql.functions.size`.
    """
    return typespark.Int(F.size(col.to_spark()))


def slice[T: typespark.TypedColumn](
    col: typespark.Array[T],
    start: typespark.Integer | int,
    length: typespark.Integer | int,
) -> typespark.Array[T]:
    """Returns a sub-array of `col` starting at `start` (1-based) with `length` elements.

    Wrapper for :func:`pyspark.sql.functions.slice`.
    """
    return typespark.Array(  # type: ignore
        F.slice(
            col.to_spark(),
            start.to_spark() if isinstance(start, typespark.TypedColumn) else start,
            length.to_spark() if isinstance(length, typespark.TypedColumn) else length,
        ),
        col._elem_type,
    )


def sort_array[T: typespark.TypedColumn](col: typespark.Array[T], asc: bool = True) -> typespark.Array[T]:
    """Returns array `col` sorted in ascending order by default; set `asc=False` for descending.

    Wrapper for :func:`pyspark.sql.functions.sort_array`.
    """
    return typespark.Array(F.sort_array(col.to_spark(), asc), col._elem_type)  # type: ignore
