"""
Typed PySpark DataFrame wrapper.

BaseDataFrame provides an explicit, typed API over pyspark.sql.DataFrame.
All PySpark operations are wrapped explicitly — there is no __getattr__ fallback.
For operations not wrapped here, use .to_df() to access the underlying DataFrame.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Never,
    Optional,
    Self,
    dataclass_transform,
    overload,
)

import attr
import attrs
import pyspark.sql
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, DataType, StructType, TimestampType

from typespark.base import _Base
from typespark.columns import AliasedTypedColumn, TypedColumn
from typespark.columns.generator import DeferredColumn, Generator
from typespark.columns.groups import _AggregateColumn, _GroupColumn
from typespark.define import define
from typespark.exceptions import MissingColumnError
from typespark.metadata import decimal, field, foreign_key, primary_key

if TYPE_CHECKING:
    from typespark.field_transforms import FieldTransformer


def _dataframe_converter(df: "BaseDataFrame | pyspark.sql.DataFrame"):
    if isinstance(df, pyspark.sql.DataFrame):
        return df
    return df.to_df()


@attrs.define(frozen=True)
class _DataFrameFields:
    """Internal attrs base holding DataFrame-specific fields.

    NOT decorated with @dataclass_transform so Pylance won't include
    these fields in subclass __init__ signatures.
    """

    _dataframe: pyspark.sql.DataFrame = attrs.field(
        converter=_dataframe_converter, alias="df"
    )
    _alias: Optional[str] = attrs.field(init=False, default=None)


@dataclass_transform(
    frozen_default=True,
    kw_only_default=True,
    field_specifiers=(attrs.field, attr.ib, decimal, foreign_key, primary_key, field),
)
class BaseDataFrame(_DataFrameFields, _Base):
    def __attrs_post_init__(self):
        object.__setattr__(
            self, "_dataframe", self._resolve(*self.columns)
        )  # Circumventing frozen

    def to_spark(self):
        return self._dataframe

    def to_df(self):
        return self._dataframe

    @classmethod
    def _wrap(cls, df: pyspark.sql.DataFrame, alias: str | None = None) -> Self:
        """Internal: wrap a Spark DataFrame that already has correct columns.

        Unlike from_df, this does NOT check for missing columns or handle defaults.
        Used by filter, join, orderBy, select, etc. where the DataFrame is produced
        by a known operation and columns are guaranteed correct.
        """
        if alias is not None:
            df = df.alias(alias)

        col_ref = (lambda fa: F.col(f"{alias}.{fa}")) if alias else None
        new = cls._build(df, col_ref=col_ref)
        object.__setattr__(new, "_alias", alias)
        object.__setattr__(new, "_dataframe", df)
        return new

    @overload
    @classmethod
    def from_df(
        cls,
        df: pyspark.sql.DataFrame,
        alias: str | None = None,
        disable_select: bool = False,
    ) -> Self: ...

    @overload
    @classmethod
    def from_df(
        cls, df: BaseDataFrame, alias: str | None = None, disable_select: bool = False
    ) -> Self: ...

    @classmethod
    def from_df(
        cls,
        df: pyspark.sql.DataFrame | BaseDataFrame,
        alias: str | None = None,
        disable_select: bool = False,
    ) -> Self:
        """Wrap a raw Spark DataFrame into a typed instance.

        Validates that all required columns exist, applies defaults for missing
        columns, and optionally narrows the DataFrame to only declared columns.
        """
        if isinstance(df, BaseDataFrame):
            df = df.to_df()

        if alias is not None:
            df = df.alias(alias)

        col_ref = (lambda fa: F.col(f"{alias}.{fa}")) if alias else None

        try:
            new = cls._build(df, col_ref=col_ref)
        except MissingColumnError as e:
            e.available_columns = df.columns
            raise

        object.__setattr__(new, "_alias", alias)

        spark_columns = [c.to_spark() for c in new.columns]
        if disable_select:
            extra = set(df.columns) - set(cls._column_aliases())
            spark_columns.extend([F.col(c) for c in extra])
        object.__setattr__(new, "_dataframe", df.select(*spark_columns))

        return new

    # ── Internal projection ─────────────────────────────────────────

    def _project(
        self, *cols: str | pyspark.sql.Column | TypedColumn[DataType]
    ) -> pyspark.sql.DataFrame:
        """Pure column projection: convert TypedColumns to Spark columns and select.

        Returns a raw Spark DataFrame. Only handles plain columns — no groups,
        aggregates, or deferred/explode columns.
        """
        return self._dataframe.select(
            *[c.to_spark() if isinstance(c, TypedColumn) else c for c in cols]
        )

    def _resolve(
        self, *cols: str | pyspark.sql.Column | TypedColumn[DataType]
    ) -> pyspark.sql.DataFrame:
        """Resolve columns (including groups, deferred) into a raw Spark DataFrame.

        Handles all TypeSpark column types:
        - Groups + aggregates → groupBy().agg()
        - Deferred / explode → two-step materialization
        - Plain columns → delegates to _project()

        Used by __attrs_post_init__ and select().
        """
        aggregates = [c.column for c in cols if isinstance(c, _AggregateColumn)]
        groups = [c.column for c in cols if isinstance(c, _GroupColumn)]
        projections = {c.parent for c in cols if isinstance(c, DeferredColumn)}

        if (aggregates or groups) and projections:
            raise NotImplementedError(
                "Support for groups and projections have not been implemented."
            )

        if aggregates or groups:
            if not aggregates:
                raise ValueError("Need to specify aggregates if using groups.")

            return self._dataframe.groupBy(*[g._col for g in groups]).agg(
                *[a._col for a in aggregates]
            )

        if len(projections) > 0:
            projected_cols = [
                c.column_operation() if isinstance(c, Generator) else c
                for c in projections
            ]
            normal_cols = [
                c
                for c in cols
                if not (isinstance(c, DeferredColumn) or isinstance(c, Generator))
            ]
            # Prevent aliasing normal cols twice
            original_named_cols = [
                F.col(n.original_name) if isinstance(n, AliasedTypedColumn) else n
                for n in normal_cols
            ]
            # Step 1: materialize generators
            df = self._dataframe.select(*projected_cols, *original_named_cols)  # type: ignore

            # Step 2: select final materialized expressions
            final_cols = [c.col if isinstance(c, DeferredColumn) else c for c in cols]
            return df.select(
                *[f.to_spark() if isinstance(f, TypedColumn) else f for f in final_cols]
            )

        # Resolve any bare Generator objects (e.g. explode without DeferredColumn access)
        resolved = [
            c.column_operation() if isinstance(c, Generator) else c for c in cols
        ]
        return self._project(*resolved)

    # ── User-facing select ───────────────────────────────────────────

    def __getattr__(self, name: str) -> Never:
        if hasattr(pyspark.sql.DataFrame, name):
            raise AttributeError(
                f"'{name}' is not yet wrapped by BaseDataFrame. "
                f"Use .to_df().{name}() to access the underlying PySpark DataFrame."
            )
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def select(
        self, *cols: str | pyspark.sql.Column | TypedColumn[DataType]
    ) -> BaseDataFrame:
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
        return BaseDataFrame._wrap(self._resolve(*cols))

    # ── Transforms ───────────────────────────────────────────────────

    def withColumn(self, colName: str, col: pyspark.sql.Column) -> BaseDataFrame:
        return BaseDataFrame._wrap(self._dataframe.withColumn(colName, col))

    def drop(self, *cols) -> BaseDataFrame:
        return BaseDataFrame._wrap(self._dataframe.drop(*cols))

    def drop_duplicates(self, subset: list[str] | None = None) -> Self:
        return self.__class__._wrap(self._dataframe.drop_duplicates(subset))

    def dropDuplicates(self, subset: list[str] | None = None) -> Self:
        return self.__class__._wrap(self._dataframe.dropDuplicates(subset))

    def distinct(self) -> Self:
        return self.__class__._wrap(self._dataframe.distinct())

    def filter(
        self, condition: str | pyspark.sql.Column | TypedColumn[BooleanType]
    ) -> Self:
        return self.__class__._wrap(
            self._dataframe.filter(
                condition.to_spark()
                if isinstance(condition, TypedColumn)
                else condition
            )
        )

    def where(
        self, condition: str | pyspark.sql.Column | TypedColumn[BooleanType]
    ) -> Self:
        return self.filter(condition)

    def orderBy(
        self, *cols: TypedColumn[DataType] | pyspark.sql.Column, **kwargs: Any
    ) -> Self:
        spark_cols = [c.to_spark() if isinstance(c, TypedColumn) else c for c in cols]
        return self.__class__._wrap(self._dataframe.orderBy(*spark_cols, **kwargs))

    def sort(
        self, *cols: TypedColumn[DataType] | pyspark.sql.Column, **kwargs: Any
    ) -> Self:
        spark_cols = [c.to_spark() if isinstance(c, TypedColumn) else c for c in cols]
        return self.__class__._wrap(self._dataframe.sort(*spark_cols, **kwargs))

    def limit(self, num: int) -> Self:
        return self.__class__._wrap(self._dataframe.limit(num))

    def coalesce(self, numPartitions: int) -> Self:
        return self.__class__._wrap(self._dataframe.coalesce(numPartitions))

    def repartition(
        self, numPartitions: int, *cols: TypedColumn[DataType] | pyspark.sql.Column
    ) -> Self:
        spark_cols = [c.to_spark() if isinstance(c, TypedColumn) else c for c in cols]
        return self.__class__._wrap(
            self._dataframe.repartition(numPartitions, *spark_cols)
        )

    def cache(self) -> Self:
        return self.__class__._wrap(self._dataframe.cache())

    def persist(self, storageLevel: Any | None = None) -> Self:
        if storageLevel is None:
            return self.__class__._wrap(self._dataframe.persist())
        return self.__class__._wrap(self._dataframe.persist(storageLevel))

    def unpersist(self, blocking: bool = False) -> Self:
        return self.__class__._wrap(self._dataframe.unpersist(blocking))

    def sample(
        self,
        withReplacement: bool = False,
        fraction: float = 0.5,
        seed: int | None = None,
    ) -> Self:
        if seed is None:
            return self.__class__._wrap(
                self._dataframe.sample(withReplacement, fraction)
            )
        return self.__class__._wrap(
            self._dataframe.sample(withReplacement, fraction, seed)
        )

    # ── Aliases ──────────────────────────────────────────────────────

    def alias(self, alias: str) -> Self:
        """Create an aliased copy of this DataFrame for use in joins.

        Prefixes all column references with ``alias`` so that columns from two
        instances of the same schema can be distinguished.
        """
        return self.__class__._wrap(self._dataframe, alias=alias)

    # ── Unions ────────────────────────────────────────────────────────

    @overload
    def union(self, other: Self) -> Self: ...
    @overload
    def union(
        self, other: BaseDataFrame, allowMissingColumns: bool = False
    ) -> BaseDataFrame: ...

    def union(
        self, other: Self | BaseDataFrame, allowMissingColumns: bool = False
    ) -> Self | BaseDataFrame:
        """Union this DataFrame with another by column name.

        Delegates to ``unionByName`` rather than positional ``union``.

        Return type depends on ``other``:

        - Same concrete class → returns ``Self`` (typed schema preserved).
        - Plain ``BaseDataFrame`` → returns untyped ``BaseDataFrame``.

        Set ``allowMissingColumns=True`` to union DataFrames with different
        column sets (missing columns are filled with ``null``).
        """
        union = self._dataframe.unionByName(
            other.to_df(), allowMissingColumns=allowMissingColumns
        )
        if isinstance(other, self.__class__):
            return self.__class__._wrap(union)

        return BaseDataFrame._wrap(union)

    # ── Joins ─────────────────────────────────────────────────────────

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
        return BaseDataFrame._wrap(
            self._dataframe.join(
                other.to_df() if isinstance(other, BaseDataFrame) else other,
                on._col if isinstance(on, TypedColumn) else on,
                how,
            ),
        )

    def leftsemi(
        self,
        other: BaseDataFrame | pyspark.sql.DataFrame,
        on: TypedColumn | str | list[str] | pyspark.sql.Column | None = None,
    ) -> Self:
        """Filter rows using a left semi join, preserving this DataFrame's typed schema."""
        return self.__class__._wrap(
            self._dataframe.join(
                other.to_df() if isinstance(other, BaseDataFrame) else other,
                on._col if isinstance(on, TypedColumn) else on,
                "leftsemi",
            ),
        )

    def leftanti(
        self,
        other: BaseDataFrame | pyspark.sql.DataFrame,
        on: TypedColumn | str | list[str] | pyspark.sql.Column | None = None,
    ) -> Self:
        """Filter rows using a left anti join, preserving this DataFrame's typed schema."""
        return self.__class__._wrap(
            self._dataframe.join(
                other.to_df() if isinstance(other, BaseDataFrame) else other,
                on._col if isinstance(on, TypedColumn) else on,
                "leftanti",
            ),
        )

    # ── Misc ──────────────────────────────────────────────────────────

    def broadcast(self) -> Self:
        return self.__class__._wrap(F.broadcast(self.to_spark()))

    def withWatermark(
        self, eventTime: TypedColumn[TimestampType], delayThreshold: str
    ) -> Self:
        return self.__class__._wrap(
            self.to_spark().withWatermark(eventTime._name, delayThreshold)
        )

    def show(self, n: int = 20, truncate: bool = True, vertical: bool = False) -> None:
        self._dataframe.show(n, truncate, vertical)

    @property
    def schema(self) -> StructType:
        return self._dataframe.schema

    @property
    def dtypes(self) -> list[tuple[str, str]]:
        """List of ``(column_name, type_string)`` tuples for the underlying DataFrame."""
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
