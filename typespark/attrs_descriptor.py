from typing import Type

import attrs


def is_generic(field_type: Type):
    return hasattr(field_type, "__origin__")


def get_descriptor(field_type: Type | None):
    if not field_type:
        return
    if is_generic(field_type):
        return field_type.__origin__
    return field_type


# Based on https://github.com/python-attrs/attrs/issues/1232#issuecomment-2187420239


def use_descriptor(annotation: Type, descriptor: Type):
    def hook(cls: type, fields: list[attrs.Attribute]) -> list[attrs.Attribute]:

        new_fields = []
        for def_field in fields:
            field_type = get_descriptor(def_field.type)
            if field_type == annotation:
                if not hasattr(descriptor, "__set_name__"):
                    raise ValueError(
                        "Descriptor must have __set_name__ to work with this transformer"
                    )
                descriptor_instance = descriptor()  # type: ignore
                getattr(descriptor_instance, "__set_name__")(
                    cls, def_field.alias or def_field.name
                )
                setattr(cls, def_field.name, descriptor_instance)
                # create a "shadow" field that accepts the value in the init
                ca = attrs.field(
                    init=False,
                    repr=False,
                    default=None,
                    kw_only=True,
                    metadata=def_field.metadata,
                    alias=def_field.alias,
                )
                a = attrs.Attribute.from_counting_attr(  # type: ignore
                    name=f"_{def_field.name}", ca=ca, type=def_field.type
                )
                new_fields.append(a)
            else:
                new_fields.append(def_field)
        return new_fields

    return hook
