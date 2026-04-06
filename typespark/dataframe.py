"""
Typed PySpark DataFrame wrapper.

BaseDataFrame provides an explicit, typed API over pyspark.sql.DataFrame.
All PySpark operations are wrapped explicitly — there is no __getattr__ fallback.
For operations not wrapped here, use .to_df() to access the underlying DataFrame.
"""

from __future__ import annotations

from typing import Any, Never, Self, overload

import pyspark.sql
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, DataType, StructType, TimestampType

from typespark.base import _Base
from typespark.columns import AliasedTypedColumn, TypedColumn
from typespark.columns.generator import DeferredColumn, Generator
from typespark.columns.groups import _AggregateColumn, _GroupColumn


class BaseDataFrame(_Base):
    def __getattr__(self, name: str) -> Never:
        if hasattr(pyspark.sql.DataFrame, name):
            raise AttributeError(
                f"'{name}' is not yet wrapped by BaseDataFrame. "
                f"Use .to_df().{name}() to access the underlying PySpark DataFrame."
            )
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def select(self, *cols: str | pyspark.sql.Column | TypedColumn[DataType]) -> BaseDataFrame:
        """Project a set of columns, returning a new untyped ``BaseDataFrame``.

        Beyond standard column projection, ``select`` handles two TypeSpark-specific
        patterns automatically:

        - **Group-by aggregation**: if any column is wrapped with ``.group()`` and
          at least one aggregate function (e.g. ``sum``, ``count``) is included,
          the call is promoted to a ``.groupBy().agg()`` under the hood.
        - **Deferred / explode columns**: columns produced by
          ``TypedArrayType.explode()`` are materialised in a two-step select so
          that the exploded values are available for subsequent projection.
        """
        aggregates = [c.column for c in cols if isinstance(c, _AggregateColumn)]
        groups = [c.column for c in cols if isinstance(c, _GroupColumn)]
        projections = {c.parent for c in cols if isinstance(c, DeferredColumn)}

        if (aggregates or groups) and projections:
            raise NotImplementedError("Support for groups and projections have not been implemented.")

        if aggregates or groups:
            if not aggregates:
                raise ValueError("Need to specify aggregates if using groups.")

            # combined_columns = set(groups + aggregates)
            # expected_columns = set(cols)
            # if not expected_columns == combined_columns:
            #     missing_columns = expected_columns - combined_columns
            #     raise ValueError(
            #         f"Missing {missing_columns} as group columns or aggregates."
            #     )

            df = self._dataframe.groupBy(*[g._col for g in groups]).agg(*[a._col for a in aggregates])
            return BaseDataFrame.from_df(df, disable_select=True)

        else:
            if len(projections) > 0:
                projected_cols = [c.column_operation() if isinstance(c, Generator) else c for c in projections]
                normal_cols = [c for c in cols if not (isinstance(c, DeferredColumn) or isinstance(c, Generator))]
                # Prevent aliasing normal cols twice
                original_named_cols = [
                    F.col(n.original_name) if isinstance(n, AliasedTypedColumn) else n for n in normal_cols
                ]
                # Step 1: materialize generators
                df = self._dataframe.select(*projected_cols, *original_named_cols)  # type: ignore

                # Step 2: select final materialized expressions
                final_cols = [c.col if isinstance(c, DeferredColumn) else c for c in cols]

            else:
                df = self._dataframe
                final_cols = [c.column_operation() if isinstance(c, Generator) else c for c in cols]

            return BaseDataFrame.from_df(
                df.select(*[f.to_spark() if isinstance(f, TypedColumn) else f for f in final_cols]),
                disable_select=True,
            )

    def withColumn(self, colName: str, col: pyspark.sql.Column) -> BaseDataFrame:
        return BaseDataFrame.from_df(self._dataframe.withColumn(colName, col), disable_select=True)

    def drop(self, *cols) -> BaseDataFrame:
        return BaseDataFrame.from_df(self._dataframe.drop(*cols), disable_select=True)

    def drop_duplicates(self, subset: list[str] | None = None) -> Self:
        return self.__class__.from_df(self._dataframe.drop_duplicates(subset))

    def dropDuplicates(self, subset: list[str] | None = None) -> Self:
        return self.__class__.from_df(self._dataframe.dropDuplicates(subset))

    def show(self, n: int = 20, truncate: bool = True, vertical: bool = False) -> None:
        self._dataframe.show(n, truncate, vertical)

    def distinct(self) -> Self:
        return self.from_df(self._dataframe.distinct())

    def filter(self, condition: str | pyspark.sql.Column | TypedColumn[BooleanType]) -> Self:
        return self.from_df(
            self._dataframe.filter(condition.to_spark() if isinstance(condition, TypedColumn) else condition)
        )

    def where(self, condition: str | pyspark.sql.Column | TypedColumn[BooleanType]) -> Self:
        return self.filter(condition)

    def orderBy(self, *cols: TypedColumn[DataType] | pyspark.sql.Column, **kwargs: Any) -> Self:
        spark_cols = [c.to_spark() if isinstance(c, TypedColumn) else c for c in cols]
        return self.from_df(self._dataframe.orderBy(*spark_cols, **kwargs))

    def sort(self, *cols: TypedColumn[DataType] | pyspark.sql.Column, **kwargs: Any) -> Self:
        spark_cols = [c.to_spark() if isinstance(c, TypedColumn) else c for c in cols]
        return self.from_df(self._dataframe.sort(*spark_cols, **kwargs))

    def limit(self, num: int) -> Self:
        return self.from_df(self._dataframe.limit(num))

    def coalesce(self, numPartitions: int) -> Self:
        return self.from_df(self._dataframe.coalesce(numPartitions))

    def repartition(self, numPartitions: int, *cols: TypedColumn[DataType] | pyspark.sql.Column) -> Self:
        spark_cols = [c.to_spark() if isinstance(c, TypedColumn) else c for c in cols]
        return self.from_df(self._dataframe.repartition(numPartitions, *spark_cols))

    def cache(self) -> Self:
        return self.from_df(self._dataframe.cache(), disable_select=True)

    def persist(self, storageLevel: Any | None = None) -> Self:
        if storageLevel is None:
            return self.from_df(self._dataframe.persist(), disable_select=True)
        return self.from_df(self._dataframe.persist(storageLevel), disable_select=True)

    def unpersist(self, blocking: bool = False) -> Self:
        return self.from_df(self._dataframe.unpersist(blocking), disable_select=True)

    def sample(
        self,
        withReplacement: bool = False,
        fraction: float = 0.5,
        seed: int | None = None,
    ) -> Self:
        if seed is None:
            return self.from_df(self._dataframe.sample(withReplacement, fraction))
        return self.from_df(self._dataframe.sample(withReplacement, fraction, seed))

    def alias(self, alias: str) -> Self:
        """Create an aliased copy of this DataFrame for use in joins.

        Prefixes all column references with ``alias`` so that columns from two
        instances of the same schema can be distinguished:

            a1 = Person.from_df(df).alias("a1")
            a2 = Person.from_df(df).alias("a2")
            a1.join(a2, a1.name == a2.name)

        This is the DataFrame-level alias used for join disambiguation.
        It is unrelated to the ``@alias`` decorator, which sets a serialisation
        alias for the data-contract schema.
        """
        return self.from_df(self._dataframe, alias)

    @overload
    def union(self, other: Self) -> Self: ...
    @overload
    def union(self, other: BaseDataFrame, allowMissingColumns: bool = False) -> BaseDataFrame: ...

    def union(self, other: Self | BaseDataFrame, allowMissingColumns: bool = False) -> Self | BaseDataFrame:
        """Union this DataFrame with another by column name.

        Delegates to ``unionByName`` rather than positional ``union``.

        Return type depends on ``other``:

        - Same concrete class → returns ``Self`` (typed schema preserved).
        - Plain ``BaseDataFrame`` → returns untyped ``BaseDataFrame``.

        Set ``allowMissingColumns=True`` to union DataFrames with different
        column sets (missing columns are filled with ``null``).
        """
        union = self._dataframe.unionByName(other.to_df(), allowMissingColumns=allowMissingColumns)
        if isinstance(other, self.__class__):
            return self.__class__.from_df(union)

        return BaseDataFrame.from_df(union)

    def join(
        self,
        other: BaseDataFrame | pyspark.sql.DataFrame,
        on: TypedColumn | str | list[str] | pyspark.sql.Column | None = None,
        how: str | None = None,
    ) -> BaseDataFrame:
        """Join with another DataFrame, returning an untyped ``BaseDataFrame``.

        The result is untyped because the combined schema depends on the join type
        and the schemas of both sides.
        """
        return BaseDataFrame.from_df(
            self._dataframe.join(
                other.to_df() if isinstance(other, _Base) else other,
                on._col if isinstance(on, TypedColumn) else on,
                how,
            ),
            disable_select=True,
        )

    def leftsemi(
        self,
        other: BaseDataFrame | pyspark.sql.DataFrame,
        on: TypedColumn | str | list[str] | pyspark.sql.Column | None = None,
    ) -> Self:
        """Filter rows using a left semi join, preserving this DataFrame's typed schema.

        A left semi join returns only rows from ``self`` that have a match in
        ``other``, without adding any columns from ``other``. Because the schema
        of ``self`` is unchanged, the result is returned as ``Self``.
        """
        return self.from_df(
            self._dataframe.join(
                other.to_df() if isinstance(other, _Base) else other,
                on._col if isinstance(on, TypedColumn) else on,
                "leftsemi",
            ),
            disable_select=True,
        )

    def leftanti(
        self,
        other: BaseDataFrame | pyspark.sql.DataFrame,
        on: TypedColumn | str | list[str] | pyspark.sql.Column | None = None,
    ) -> Self:
        """Filter rows using a left anti join, preserving this DataFrame's typed schema.

        A left anti join returns only rows from ``self`` that have no match in
        ``other``, without adding any columns from ``other``. Because the schema
        of ``self`` is unchanged, the result is returned as ``Self``.
        """
        return self.from_df(
            self._dataframe.join(
                other.to_df() if isinstance(other, _Base) else other,
                on._col if isinstance(on, TypedColumn) else on,
                "leftanti",
            ),
            disable_select=True,
        )

    def broadcast(self) -> Self:
        return self.from_df(F.broadcast(self.to_spark()))

    def withWatermark(self, eventTime: TypedColumn[TimestampType], delayThreshold: str) -> Self:
        return self.from_df(self.to_spark().withWatermark(eventTime._name, delayThreshold))

    @property
    def schema(self) -> StructType:
        return self._dataframe.schema

    @property
    def dtypes(self) -> list[tuple[str, str]]:
        """List of ``(column_name, type_string)`` tuples for the underlying DataFrame.

        Note: unlike the ``columns`` property (which returns ``list[TypedColumn]``),
        ``dtypes`` returns plain strings and is unaware of TypeSpark's type system.
        """
        return self._dataframe.dtypes

    def count(self) -> int:
        return self._dataframe.count()

    def collect(self) -> list[Row]:
        return self._dataframe.collect()

    def first(self) -> Row | None:
        return self._dataframe.first()

    def toPandas(self):
        return self._dataframe.toPandas()

    def printSchema(self) -> None:
        self._dataframe.printSchema()

    def createOrReplaceTempView(self, name: str) -> None:
        self._dataframe.createOrReplaceTempView(name)

    def createTempView(self, name: str) -> None:
        self._dataframe.createTempView(name)
