# pyright: reportUnnecessaryTypeIgnoreComment=true

"""Array[Decimal] precision/scale usage, checked with pyright by tests/test_typecheck.py.

Lines marked `# type: ignore` must produce an error.
"""

import typespark as ts


class Raw(ts.DataFrame):
    flat: ts.Array[ts.String]
    nested: ts.Array[ts.Array[ts.String]]


class Amounts(ts.DataFrame):
    amounts: ts.Array[ts.Decimal] = ts.decimal(18, 4)
    nested: ts.Array[ts.Array[ts.Decimal]] = ts.decimal(18, 4)


def build(raw: Raw) -> Amounts:
    amounts = ts.Array[ts.Decimal](raw.flat, precision=18, scale=4)
    first: ts.Decimal = amounts.getItem(0)
    _ = first
    return Amounts(
        raw,
        amounts=amounts,
        nested=ts.Array[ts.Array[ts.Decimal]](raw.nested, precision=18, scale=4),
    )


def plain(raw: Raw) -> ts.Array[ts.Decimal]:
    return ts.Array[ts.Decimal](raw.flat)


def non_decimal(raw: Raw) -> None:
    ts.Array[ts.String](raw.flat, precision=18, scale=4)  # type: ignore
