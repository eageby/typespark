import attrs

from typespark.metadata import MetaData


def get_field_name(field: attrs.Attribute):
    return MetaData(**field.metadata).df_alias or field.name
