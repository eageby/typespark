from typing_extensions import deprecated

from typespark.type_alias import Bool, Integer, String

LiteralType = str | int | float | bool | None


@deprecated("Use String(value) instead")
def string_literal(value: str | None) -> String:
    return String(value)


@deprecated("Use Integer(value) instead")
def int_literal(value: int | None) -> Integer:
    return Integer(value)


@deprecated("Use Bool(value) instead")
def bool_literal(value: bool | None) -> Bool:
    return Bool(value)
