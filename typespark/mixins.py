from typing import ClassVar, Optional
import attrs


class Aliasable:
    __serialization_alias__: ClassVar[str | None]


def alias(name: Optional[str] = None):
    def decorator(cls: type) -> type:
        if not issubclass(cls, Aliasable):
            raise TypeError(
                f"Cannot apply @{alias.__name__} to class '{cls.__name__}': "
                f"it must inherit from '{Aliasable.__name__}'."
            )
        cls.__serialization_alias__ = name
        return cls

    return decorator


class SchemaDefaults:
    """Not implemented"""

    __default_nullable__: ClassVar[bool | None] = None


def nullable(default: bool = True):
    def decorator(cls: type) -> type:
        if not issubclass(cls, SchemaDefaults):
            raise TypeError(
                f"Cannot apply @{nullable.__name__} to class '{cls.__name__}': "
                f"it must inherit from '{SchemaDefaults.__name__}'."
            )
        cls.__default_nullable__ = default
        return cls

    return decorator
