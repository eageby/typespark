# pyright: reportUnnecessaryTypeIgnoreComment=true
# mypy: warn-unused-ignores

"""The DataFrame constructor, checked with pyright and mypy by tests/test_typecheck.py.

Lines marked `# type: ignore` must produce an error in both checkers.
"""

import typespark as ts


class Raw(ts.DataFrame):
    a: ts.Int


class Out(ts.DataFrame):
    a: ts.Int
    b: ts.String = ts.field(default=None)


def from_typed_frame(raw: Raw) -> Out:
    return Out(raw, a=raw.a)


def from_spark_frame(raw: Raw) -> Out:
    return Out(raw.to_df(), a=raw.a)


def with_default_given(raw: Raw) -> Out:
    return Out(raw, a=raw.a, b=ts.String("x"))


def df_is_not_an_attribute(out: Out) -> object:
    return out.df  # type: ignore


def df_is_required(raw: Raw) -> None:
    Out(a=raw.a)  # type: ignore


def fields_are_keyword_only(raw: Raw) -> None:
    Out(raw, raw.a)  # type: ignore


def field_type_is_checked(raw: Raw) -> None:
    Out(raw, a=raw.a, b=raw.a)  # type: ignore
