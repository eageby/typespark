"""Tests for class-based column types: explicit subclasses, ArrayOf caching, and casting."""

import pytest
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, LongType, StringType

from typespark.columns.array import ArrayOf, TypedArrayType
from typespark.columns.columns import TypedColumn
from typespark.columns.primitives import Integer, String
from typespark import UnnamedColumnError


# ── Primitive subclassing ─────────────────────────────────────────────────────


def test_primitive_has_data_type():
    assert String._data_type is StringType
    assert Integer._data_type is IntegerType


def test_primitive_is_subclass_of_typed_column():
    assert issubclass(String, TypedColumn)
    assert issubclass(Integer, TypedColumn)


def test_explicit_subclass_inherits_data_type():
    class MyString(String):
        pass

    assert MyString._data_type is StringType
    assert issubclass(MyString, String)
    assert issubclass(MyString, TypedColumn)


def test_explicit_subclass_with_override():
    class MyInteger(Integer):
        _data_type = IntegerType  # explicit, same value

    assert MyInteger._data_type is IntegerType
    assert issubclass(MyInteger, Integer)


# ── ArrayOf factory and caching ───────────────────────────────────────────────


def test_array_of_sets_elem_type():
    assert ArrayOf(String)._elem_type is String


def test_array_of_is_cached():
    assert ArrayOf(String) is ArrayOf(String)
    assert ArrayOf(Integer) is ArrayOf(Integer)


def test_array_class_getitem_delegates_to_array_of():
    import typespark as ts

    assert ts.Array[String] is ArrayOf(String)
    assert ts.Array[Integer] is ArrayOf(Integer)


def test_array_of_different_types_are_distinct():
    assert ArrayOf(String) is not ArrayOf(Integer)


def test_array_of_is_subclass_of_typed_array_type():
    assert issubclass(ArrayOf(String), TypedArrayType)
    assert issubclass(ArrayOf(String), TypedColumn)


# ── __init__ casting and wrap ─────────────────────────────────────────────────


def test_init_casts_to_declared_type(spark):
    raw = spark.range(1)  # id: LongType
    col = String(raw["id"])
    df = raw.select(col.to_spark().alias("v"))
    assert df.schema["v"].dataType == StringType()


def test_wrap_preserves_original_type(spark):
    raw = spark.range(1)  # id: LongType
    col = String.wrap(raw["id"])
    df = raw.select(col.to_spark().alias("v"))
    assert df.schema["v"].dataType == LongType()


def test_base_typed_column_init_does_not_cast(spark):
    # TypedColumn base has no _data_type, so no cast should happen
    raw = spark.range(1)  # id: LongType
    col = TypedColumn(raw["id"])
    df = raw.select(col.to_spark().alias("v"))
    assert df.schema["v"].dataType == LongType()


# ── Unnamed column guards ─────────────────────────────────────────────────────


def test_unnamed_column_repr():
    col = String(F.col("x"))
    assert repr(col) == "String<>"


def test_named_column_repr():
    col = String._set_column(F.col("x"), "x")
    assert repr(col) == "String<'x'>"


def test_unnamed_column_cast_preserves_no_name():
    col = String(F.col("x"))
    casted = col.cast(IntegerType())
    assert casted._name is None


def test_unnamed_column_alias_sets_name():
    col = String(F.col("x"))
    aliased = col.alias("y")
    assert aliased._name == "y"


def test_explode_unnamed_raises():
    col = ArrayOf(String)(F.col("tags"))
    with pytest.raises(UnnamedColumnError, match="explode"):
        col.explode()
