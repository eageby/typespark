import inspect
from types import ModuleType
from typing import List, Optional, Type, get_args, get_origin

import attr
from datacontract_specification.model import DataContractSpecification, Field, Model
from pydantic.alias_generators import to_snake
from pyspark.sql.types import (
    BinaryType,
    BooleanType,
    DataType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    TimestampType,
)
from simple_parsing import docstring

from typespark.base import BaseDataFrame
from typespark.metadata import MetaData
from typespark.columns import TypedColumn


def pyspark_type_to_contract(pyspark_type: DataType) -> str:
    """Convert a PySpark DataType to a data contract type string."""
    if pyspark_type is StringType:
        return "string"
    elif pyspark_type is IntegerType:
        return "int"
    elif pyspark_type is LongType:
        return "long"
    elif pyspark_type is ShortType:
        return "short"
    elif pyspark_type is FloatType:
        return "float"
    elif pyspark_type is DoubleType:
        return "double"
    elif pyspark_type is BooleanType:
        return "bool"
    elif pyspark_type is TimestampType:
        return "timestamp"
    elif pyspark_type is DateType:
        return "date"
    elif pyspark_type is BinaryType:
        return "bytes"
    elif pyspark_type is DecimalType:
        return "decimal"
    # elif pyspark_type is ArrayType:
    #     element_type = pyspark_type_to_contract(pyspark_type.elementType)
    #     return f"array<{element_type}>"
    # elif pyspark_type is MapType:
    #     key_type = pyspark_type_to_contract(pyspark_type.keyType)
    #     value_type = pyspark_type_to_contract(pyspark_type.valueType)
    #     return f"map<{key_type}, {value_type}>"
    # elif pyspark_type is StructType:
    #     return "struct"
    else:
        raise TypeError(f"Unsupported PySpark type: {pyspark_type}")


def resolve_pk(t: type[BaseDataFrame]):

    pks = [i for i in attr.fields(t) if MetaData(**i.metadata).primary_key is not None]

    if len(pks) == 1:
        return pks[0]
    else:
        raise ValueError(
            f"Cant reference composite primary keys of {t.__name__ or t.__serialization_alias__}"
        )


def foreign_key(t: Optional[type[BaseDataFrame]]):
    if t is None:
        return None
    pk = resolve_pk(t)
    return f"{t.__serialization_alias__ or to_snake(t.__name__)}.{pk.alias or pk.name}"


def resolve_type(t: type | None):
    if t is None or get_origin(t) is not TypedColumn:
        return None
    return pyspark_type_to_contract(get_args(t)[0])


def field(cls: type[BaseDataFrame], f: attr.Attribute):

    if f.alias is None:
        raise ValueError("Field alias missing")

    metadata = MetaData(**f.metadata)
    doc = docstring.get_attribute_docstring(cls, f.alias).help_string
    return Field(
        primaryKey=metadata.primary_key,
        references=foreign_key(metadata.foreign_key),
        type=resolve_type(f.type),
        description=doc,
    )


def serialize_class_metadata(cls: type[BaseDataFrame]):

    fields = {
        v.alias or v.name or k: field(cls, v) for k, v in attr.fields_dict(cls).items()
    }

    name = cls.__serialization_alias__ or to_snake(cls.__name__)

    return name, Model(description=cls.__doc__, fields=fields, type="table")


def find_exposed_subclasses(pkg: ModuleType, base_cls: Type) -> List[Type]:
    subclasses = []

    for name in getattr(pkg, "__all__", dir(pkg)):
        try:
            obj = getattr(pkg, name)
        except AttributeError:
            continue

        if inspect.isclass(obj) and issubclass(obj, base_cls) and obj is not base_cls:
            subclasses.append(obj)

    return subclasses


def serialize_product(pkg: ModuleType):
    models = find_exposed_subclasses(pkg, BaseDataFrame)

    return DataContractSpecification(
        models={k: v for k, v in [serialize_class_metadata(i) for i in models]},
    ).to_yaml()
