from pyspark.sql.functions import lit
from pyspark.sql.types import IntegerType, StringType

from typespark.columns import TypedColumn
from typespark.type_alias import Integer, String

LiteralType = str | int | float | bool | None


def _lit(value: LiteralType):
    return TypedColumn(lit(value))


def string_literal(value: str | None) -> String:
    return _lit(value).cast(StringType())


def int_literal(value: int | None) -> Integer:
    return _lit(value).cast(IntegerType())
