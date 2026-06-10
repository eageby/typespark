from pyspark.sql import types
from pyspark.sql.types import ArrayType, IntegerType, NullType, ShortType, StringType, StructType

import typespark as ts
from typespark.columns.struct import Struct
from typespark.metadata import field


def test_schema():
    class Data(ts.DataFrame):
        a: ts.String
        b: ts.Integer

    schema = Data.generate_schema()

    assert schema.fields[0].dataType == types.StringType()
    assert schema.fields[1].dataType == types.IntegerType()


def test_schema_with_struct():
    class Info(ts.Struct):
        a: ts.String
        b: ts.Integer

    class Data(ts.DataFrame):
        info: Info

    schema = Data.generate_schema()
    struct = schema.fields[0].dataType

    assert schema.fields[0].name == "info"
    assert isinstance(struct, types.StructType)

    assert struct[0].dataType == types.StringType()
    assert struct[0].name == "a"

    assert struct[1].dataType == types.IntegerType()
    assert struct[1].name == "b"


def test_array_schema():
    class Data(ts.DataFrame):
        a: ts.Array[ts.String]
        b: ts.Array[ts.Integer]

    schema = Data.generate_schema()

    assert isinstance(schema.fields[0].dataType, types.ArrayType)
    assert isinstance(schema.fields[1].dataType, types.ArrayType)

    assert isinstance(schema.fields[0].dataType.elementType, types.StringType)
    assert isinstance(schema.fields[1].dataType.elementType, types.IntegerType)


def test_schema_element_type_equality():
    class Data(ts.DataFrame):
        tags: ts.Array[ts.String]
        counts: ts.Array[ts.Integer]

    schema = Data.generate_schema()

    assert schema["tags"].dataType.elementType == StringType()
    assert schema["counts"].dataType.elementType == IntegerType()


def test_schema_nested_struct():
    class Info(Struct):
        label: ts.String
        value: ts.Integer

    class Data(ts.DataFrame):
        info: Info

    schema = Data.generate_schema()

    struct_type = schema["info"].dataType
    assert isinstance(struct_type, StructType)
    assert struct_type["label"].dataType == StringType()
    assert struct_type["value"].dataType == IntegerType()


def test_schema_array_of_struct():
    class Item(Struct):
        name: ts.String
        count: ts.Integer

    class Data(ts.DataFrame):
        items: ts.Array[Item]

    schema = Data.generate_schema()

    array_type = schema["items"].dataType
    assert isinstance(array_type, ArrayType)
    elem = array_type.elementType
    assert isinstance(elem, StructType)
    assert elem["name"].dataType == StringType()
    assert elem["count"].dataType == IntegerType()


def test_schema_with_df_alias():
    class Data(ts.DataFrame):
        name_: ts.String = field(df_alias="name")

    schema = Data.generate_schema()

    assert schema.fieldNames() == ["name"]
    assert schema["name"].dataType == StringType()


def test_schema_bare_typed_column_class():
    """Bare TypedColumn[T] with a DataType class should resolve via get_args fallback."""

    class Data(ts.DataFrame):
        a: ts.Column[ShortType]

    schema = Data.generate_schema()
    assert schema["a"].dataType == ShortType()


def test_schema_bare_typed_column_instance():
    """Bare TypedColumn[T] with a DataType instance should be returned as-is."""

    class Data(ts.DataFrame):
        a: ts.Column[NullType()]  # type: ignore[type-arg]

    schema = Data.generate_schema()
    assert schema["a"].dataType == NullType()


def test_array_of_struct_generate_schema():
    class Item(Struct):
        label: ts.String

    schema = ts.ArrayOf(Item).generate_schema()

    assert isinstance(schema, ArrayType)
    assert isinstance(schema.elementType, StructType)
    assert schema.elementType["label"].dataType == StringType()


def test_nested_array_generate_schema():
    schema = ts.ArrayOf(ts.ArrayOf(ts.Integer)).generate_schema()

    assert isinstance(schema, ArrayType)
    assert isinstance(schema.elementType, ArrayType)
    assert schema.elementType.elementType == IntegerType()


def test_nested_array_field_in_dataframe():
    class Data(ts.DataFrame):
        matrix: ts.ArrayOf(ts.ArrayOf(ts.Integer))

    schema = Data.generate_schema()

    outer = schema["matrix"].dataType
    assert isinstance(outer, ArrayType)
    assert isinstance(outer.elementType, ArrayType)
    assert outer.elementType.elementType == IntegerType()
