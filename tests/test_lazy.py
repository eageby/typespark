"""Tests for DataFrame.lazy() — column-only handles with no underlying data.

Pure expression building needs no Spark session.
"""

import pytest

from pyspark.sql import Row

from tests.conftest import Person
from typespark import BaseDataFrame, Int, LazyDataFrameError, String
from typespark.columns.struct import Struct
from typespark.metadata import field


class _Addr(Struct):
    city: String
    zip: String


class _Cust(BaseDataFrame):
    id: Int
    address: _Addr


class _AliasedAddr(Struct):
    city: String = field(df_alias="city_raw")


class _AliasedCust(BaseDataFrame):
    id: Int
    address: _AliasedAddr = field(df_alias="addr_raw")


@pytest.fixture(autouse=True)
def _active_spark(spark):
    """F.col() requires an active SparkContext even for pure expression building."""
    yield


def test_aliased_column_renders_qualified():
    t = Person.lazy("t")
    assert str(t.name.to_spark()) == "Column<'t.name'>"


def test_unaliased_column_renders_bare():
    t = Person.lazy()
    assert str(t.name.to_spark()) == "Column<'name'>"


def test_single_key_condition():
    t = Person.lazy("t")
    s = Person.lazy("s")
    assert str((t.name == s.name).to_spark()) == "Column<'=(t.name, s.name)'>"


def test_multi_key_condition():
    t = Person.lazy("t")
    s = Person.lazy("s")
    cond = (t.name == s.name) & (t.age == s.age)
    assert str(cond.to_spark()) == "Column<'and(=(t.name, s.name), =(t.age, s.age))'>"


def test_on_callable_pattern():
    """The consuming sink pattern: on=lambda t, s: t.id == s.id."""
    on = lambda t, s: t.name == s.name  # noqa: E731
    cond = on(Person.lazy("t"), Person.lazy("s"))
    assert str(cond.to_spark()) == "Column<'=(t.name, s.name)'>"


# ── Guardrails: data access raises ──────────────────────────────────────


def test_to_df_raises():
    with pytest.raises(LazyDataFrameError):
        Person.lazy("t").to_df()


def test_to_spark_raises():
    with pytest.raises(LazyDataFrameError):
        Person.lazy("t").to_spark()


def test_filter_raises():
    t = Person.lazy("t")
    with pytest.raises(LazyDataFrameError):
        t.filter(t.age == t.age)


def test_dataframe_attr_raises():
    with pytest.raises(LazyDataFrameError):
        _ = Person.lazy("t")._dataframe


def test_error_reports_origin_class_and_alias():
    with pytest.raises(LazyDataFrameError) as exc:
        Person.lazy("t").to_df()
    assert exc.value.model is Person
    assert exc.value.alias == "t"
    assert "Person" in str(exc.value)


# ── Identity / typing shape ─────────────────────────────────────────────


def test_isinstance_of_origin():
    assert isinstance(Person.lazy("t"), Person)


def test_lazy_class_is_cached():
    assert type(Person.lazy("t")) is type(Person.lazy("s"))


def test_alias_recorded():
    assert Person.lazy("t")._alias == "t"
    assert Person.lazy()._alias is None


def test_columns_accessible():
    t = Person.lazy("t")
    assert {c._name for c in t.columns} == {"name", "age"}


# ── Nested struct sub-fields ────────────────────────────────────────────


def test_nested_struct_column():
    t = _Cust.lazy("t")
    assert str(t.address.to_spark()) == "Column<'t.address'>"


def test_nested_struct_subfield():
    t = _Cust.lazy("t")
    assert str(t.address.city.to_spark()) == "Column<'t.address['city']'>"


def test_nested_struct_condition():
    t = _Cust.lazy("t")
    s = _Cust.lazy("s")
    cond = t.address.city == s.address.city
    assert str(cond.to_spark()) == "Column<'=(t.address['city'], s.address['city'])'>"


def test_nested_struct_df_alias():
    t = _AliasedCust.lazy("t")
    assert str(t.address.to_spark()) == "Column<'t.addr_raw'>"
    assert str(t.address.city.to_spark()) == "Column<'t.addr_raw['city_raw']'>"


def test_lazy_subfield_matches_real_aliased_frame(spark):
    """Conditions built from lazy must bind against a real aliased frame."""
    df = spark.createDataFrame([Row(id=1, addr_raw=Row(city_raw="NYC"))])
    real = _AliasedCust.from_df(df, alias="t")
    lazy = _AliasedCust.lazy("t")
    assert str(real.address.city.to_spark()) == str(lazy.address.city.to_spark())
