from pyspark.sql.functions import lit
from pyspark.sql.types import BooleanType, IntegerType, StringType

from typespark.columns import TypedColumn
from typespark.type_alias import Bool, Integer, String

LiteralType = str | int | float | bool | None


def _lit(value: LiteralType):
    return TypedColumn(lit(value))


def string_literal(value: str | None) -> String:
    return _lit(value).cast(StringType())


def int_literal(value: int | None) -> Integer:
    return _lit(value).cast(IntegerType())


def bool_literal(value: bool | None) -> Bool:
    return _lit(value).cast(BooleanType())
