from pyspark.sql import types

import typespark as ts


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
