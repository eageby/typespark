from typing import ClassVar, Optional


class Aliasable:
    __serialization_alias__: ClassVar[str | None]


def alias(name: Optional[str] = None):
    def decorator(cls: type) -> type:
        if not issubclass(cls, Aliasable):
            raise TypeError(
                f"Cannot apply @aliasable to class '{cls.__name__}': "
                "it must inherit from 'Aliasable'."
            )
        cls.__serialization_alias__ = name
        return cls

    return decorator
