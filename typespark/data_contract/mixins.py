from typing import ClassVar


class Aliasable:
    """Optional base class for schema serialization alias support via @alias."""

    __serialization_alias__: ClassVar[str | None]


def alias(name: str | None = None):
    def decorator(cls: type) -> type:
        cls.__serialization_alias__ = name
        return cls

    return decorator
