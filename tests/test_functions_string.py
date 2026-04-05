from pyspark.sql import Row, SparkSession
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    IntegerType,
    StringType,
)

import typespark
from tests.utils import collect_column
from typespark import functions as tsf
from typespark.columns.array import TypedArrayType


def test_concat(spark: SparkSession):
    class Data(typespark.DataFrame):
        s: typespark.String
        t: typespark.String

    data = [
        ("hello", "lo"),
        ("world", "zz"),
    ]
    strings = Data.from_df(spark.createDataFrame(data, schema=Data.generate_schema()))

    result = strings.select(
        tsf.concat(strings.s).alias("no-op"),
        tsf.concat(strings.s, strings.t).alias("concat"),
    )

    values = collect_column(result, "no-op")
    concated_values = collect_column(result, "concat")
    assert values == ["hello", "world"]
    assert concated_values == ["hellolo", "worldzz"]

    assert isinstance(result.to_spark().schema["concat"].dataType, StringType)
    assert isinstance(result.to_spark().schema["no-op"].dataType, StringType)


def test_contains_string_column(spark: SparkSession):
    class Data(typespark.DataFrame):
        s: typespark.String
        t: typespark.String

    data = [
        ("hello", "lo"),
        ("world", "zz"),
    ]

    strings = Data.from_df(spark.createDataFrame(data, schema=Data.generate_schema()))

    result = strings.select(
        tsf.contains(strings.s, strings.t).alias("has"),
        strings.s.alias("s"),
        strings.t.alias("t"),
    )

    from tests.utils import collect_values
    rows = collect_values(result)
    assert rows[0]["has"] is True
    assert rows[1]["has"] is False

    assert isinstance(result.to_spark().schema["has"].dataType, BooleanType)


def test_contains_binary_column(spark: SparkSession):
    class BinaryTestData(typespark.DataFrame):
        b: typespark.Binary
        needle: typespark.Binary

    data = [
        (bytearray(b"abcdef"), bytearray(b"bc")),
        (bytearray(b"1234"), bytearray(b"zz")),
    ]

    binaries = BinaryTestData.from_df(spark.createDataFrame(data, schema=BinaryTestData.generate_schema()))

    result = binaries.select(tsf.contains(binaries.b, binaries.needle).alias("has"))

    values = collect_column(result, "has")
    assert values == [True, False]

    assert isinstance(result.to_spark().schema["has"].dataType, BooleanType)


def test_lower(spark: SparkSession):
    class Data(typespark.DataFrame):
        s: typespark.String

    data = [
        ("hello",),
        ("WORLD",),
        ("MiXeD CaSe",),
    ]

    strings = Data.from_df(spark.createDataFrame(data, schema=Data.generate_schema()))

    result = strings.select(tsf.lower(strings.s).alias("lowered"))

    values = collect_column(result, "lowered")

    assert values == ["hello", "world", "mixed case"]

    assert isinstance(result.to_spark().schema["lowered"].dataType, StringType)


def test_ltrim(spark: SparkSession):
    class Data(typespark.DataFrame):
        s: typespark.String
        trim: typespark.String

    data = [
        ("xxhello", "x"),
        ("xyxworld  ", "xy"),
        ("   both  ", " "),
        ("nochange", "x"),
    ]

    df = Data.from_df(spark.createDataFrame(data, schema=Data.generate_schema()))

    result = df.select(
        tsf.ltrim(df.s).alias("default_trimmed"),
        tsf.ltrim(df.s, df.trim).alias("trimmed"),
    )

    default = collect_column(result, "default_trimmed")
    trimmed = collect_column(result, "trimmed")

    assert default == [
        "xxhello",
        "xyxworld  ",
        "both  ",
        "nochange",
    ]

    assert trimmed == [
        "hello",
        "world  ",
        "both  ",
        "nochange",
    ]

    assert isinstance(
        result.to_spark().schema["default_trimmed"].dataType,
        StringType,
    )
    assert isinstance(
        result.to_spark().schema["trimmed"].dataType,
        StringType,
    )


def test_ascii(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    df = Strings.from_df(
        spark.createDataFrame([("A",), ("a",)], schema=Strings.generate_schema())
    )
    result = df.select(tsf.ascii(df.s).alias("v"))
    assert collect_column(result, "v") == [65, 97]
    assert isinstance(result.to_spark().schema["v"].dataType, IntegerType)


def test_bin(spark: SparkSession):
    class Nums(typespark.DataFrame):
        x: typespark.Int

    df = Nums.from_df(
        spark.createDataFrame([(5,), (10,)], schema=Nums.generate_schema())
    )
    result = df.select(tsf.bin(df.x).alias("v"))
    assert collect_column(result, "v") == ["101", "1010"]
    assert isinstance(result.to_spark().schema["v"].dataType, StringType)


def test_btrim(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    df = Strings.from_df(
        spark.createDataFrame(
            [("xxhelloxx",), ("  world  ",)], schema=Strings.generate_schema()
        )
    )
    result = df.select(
        tsf.btrim(df.s).alias("default"),
    )
    assert collect_column(result, "default") == ["xxhelloxx", "world"]
    assert isinstance(result.to_spark().schema["default"].dataType, StringType)


def test_char(spark: SparkSession):
    class Nums(typespark.DataFrame):
        x: typespark.Int

    df = Nums.from_df(
        spark.createDataFrame([(65,), (97,)], schema=Nums.generate_schema())
    )
    result = df.select(tsf.char(df.x).alias("v"))
    assert collect_column(result, "v") == ["A", "a"]
    assert isinstance(result.to_spark().schema["v"].dataType, StringType)


def test_char_length(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    df = Strings.from_df(
        spark.createDataFrame([("hello",), ("",)], schema=Strings.generate_schema())
    )
    result = df.select(tsf.char_length(df.s).alias("v"))
    assert collect_column(result, "v") == [5, 0]
    assert isinstance(result.to_spark().schema["v"].dataType, IntegerType)


def test_decode_encode(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    df = Strings.from_df(
        spark.createDataFrame([("hello",)], schema=Strings.generate_schema())
    )

    class Encoded(typespark.DataFrame):
        b: typespark.Binary

    encoded = Encoded(df, b=tsf.encode(df.s, "UTF-8")).alias("e")
    decoded = encoded.select(tsf.decode(encoded.b, "UTF-8").alias("s"))

    assert collect_column(decoded, "s") == ["hello"]
    assert isinstance(encoded.to_spark().schema["b"].dataType, BinaryType)
    assert isinstance(decoded.to_spark().schema["s"].dataType, StringType)


def test_endswith(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String
        suffix: typespark.String

    df = Strings.from_df(
        spark.createDataFrame(
            [("hello world", "world"), ("hello world", "hello")],
            schema=Strings.generate_schema(),
        )
    )
    result = df.select(tsf.endswith(df.s, df.suffix).alias("v"))
    assert collect_column(result, "v") == [True, False]
    assert isinstance(result.to_spark().schema["v"].dataType, BooleanType)


def test_find_in_set(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String
        set_str: typespark.String

    df = Strings.from_df(
        spark.createDataFrame(
            [("b", "a,b,c"), ("d", "a,b,c")],
            schema=Strings.generate_schema(),
        )
    )
    result = df.select(tsf.find_in_set(df.s, df.set_str).alias("v"))
    assert collect_column(result, "v") == [2, 0]
    assert isinstance(result.to_spark().schema["v"].dataType, IntegerType)


def test_format_number(spark: SparkSession):
    class Nums(typespark.DataFrame):
        x: typespark.Double

    df = Nums.from_df(
        spark.createDataFrame([(1234567.89,)], schema=Nums.generate_schema())
    )
    result = df.select(tsf.format_number(df.x, 2).alias("v"))
    assert collect_column(result, "v") == ["1,234,567.89"]
    assert isinstance(result.to_spark().schema["v"].dataType, StringType)


def test_format_string(spark: SparkSession):
    class Data(typespark.DataFrame):
        name: typespark.String
        age: typespark.Int

    df = Data.from_df(
        spark.createDataFrame([("Alice", 30)], schema=Data.generate_schema())
    )
    result = df.select(
        tsf.format_string("%s is %d years old", df.name, df.age).alias("v")
    )
    assert collect_column(result, "v") == ["Alice is 30 years old"]
    assert isinstance(result.to_spark().schema["v"].dataType, StringType)


def test_hex_unhex(spark: SparkSession):
    class Nums(typespark.DataFrame):
        x: typespark.Int

    df = Nums.from_df(spark.createDataFrame([(255,)], schema=Nums.generate_schema()))
    result = df.select(tsf.hex(df.x).alias("v"))
    assert collect_column(result, "v") == ["FF"]
    assert isinstance(result.to_spark().schema["v"].dataType, StringType)


def test_instr(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    df = Strings.from_df(
        spark.createDataFrame(
            [("hello world",), ("abcdef",)], schema=Strings.generate_schema()
        )
    )
    result = df.select(tsf.instr(df.s, "world").alias("v"))
    assert collect_column(result, "v") == [7, 0]
    assert isinstance(result.to_spark().schema["v"].dataType, IntegerType)


def test_left(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    df = Strings.from_df(
        spark.createDataFrame(
            [("hello",), ("world",)], schema=Strings.generate_schema()
        )
    )
    result = df.select(tsf.left(df.s, 3).alias("v"))
    assert collect_column(result, "v") == ["hel", "wor"]
    assert isinstance(result.to_spark().schema["v"].dataType, StringType)


def test_length(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    df = Strings.from_df(
        spark.createDataFrame([("hello",), ("",)], schema=Strings.generate_schema())
    )
    result = df.select(tsf.length(df.s).alias("v"))
    assert collect_column(result, "v") == [5, 0]
    assert isinstance(result.to_spark().schema["v"].dataType, IntegerType)


def test_levenshtein(spark: SparkSession):
    class Strings(typespark.DataFrame):
        a: typespark.String
        b: typespark.String

    df = Strings.from_df(
        spark.createDataFrame(
            [("kitten", "sitting"), ("hello", "hello")],
            schema=Strings.generate_schema(),
        )
    )
    result = df.select(tsf.levenshtein(df.a, df.b).alias("v"))
    assert collect_column(result, "v") == [3, 0]
    assert isinstance(result.to_spark().schema["v"].dataType, IntegerType)


def test_locate(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    df = Strings.from_df(
        spark.createDataFrame(
            [("hello world",), ("abcdef",)], schema=Strings.generate_schema()
        )
    )
    result = df.select(tsf.locate("world", df.s).alias("v"))
    assert collect_column(result, "v") == [7, 0]
    assert isinstance(result.to_spark().schema["v"].dataType, IntegerType)


def test_octet_length(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    df = Strings.from_df(
        spark.createDataFrame([("hello",)], schema=Strings.generate_schema())
    )
    result = df.select(tsf.octet_length(df.s).alias("v"))
    assert collect_column(result, "v") == [5]
    assert isinstance(result.to_spark().schema["v"].dataType, IntegerType)


def test_overlay(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String
        r: typespark.String

    df = Strings.from_df(
        spark.createDataFrame(
            [("hello world", "there")], schema=Strings.generate_schema()
        )
    )
    result = df.select(tsf.overlay(df.s, df.r, 7, 5).alias("v"))
    assert collect_column(result, "v") == ["hello there"]
    assert isinstance(result.to_spark().schema["v"].dataType, StringType)


def test_repeat(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    df = Strings.from_df(
        spark.createDataFrame([("ab",), ("x",)], schema=Strings.generate_schema())
    )
    result = df.select(tsf.repeat(df.s, 3).alias("v"))
    assert collect_column(result, "v") == ["ababab", "xxx"]
    assert isinstance(result.to_spark().schema["v"].dataType, StringType)


def test_reverse_string(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    df = Strings.from_df(
        spark.createDataFrame([("hello",), ("abc",)], schema=Strings.generate_schema())
    )
    result = df.select(tsf.reverse(df.s).alias("v"))
    assert collect_column(result, "v") == ["olleh", "cba"]
    assert isinstance(result.to_spark().schema["v"].dataType, StringType)


def test_reverse_array(spark: SparkSession):
    class Arrays(typespark.DataFrame):
        a: TypedArrayType[typespark.Int]

    df = Arrays.from_df(
        spark.createDataFrame([Row(a=[1, 2, 3])], schema=Arrays.generate_schema())
    )
    result = df.select(tsf.reverse(df.a).alias("v"))
    assert collect_column(result, "v") == [[3, 2, 1]]
    schema_type = result.to_spark().schema["v"].dataType
    assert isinstance(schema_type, ArrayType)
    assert isinstance(schema_type.elementType, IntegerType)


def test_right(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    df = Strings.from_df(
        spark.createDataFrame(
            [("hello",), ("world",)], schema=Strings.generate_schema()
        )
    )
    result = df.select(tsf.right(df.s, 3).alias("v"))
    assert collect_column(result, "v") == ["llo", "rld"]
    assert isinstance(result.to_spark().schema["v"].dataType, StringType)


def test_rpad(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    df = Strings.from_df(
        spark.createDataFrame([("hi",)], schema=Strings.generate_schema())
    )
    result = df.select(tsf.rpad(df.s, 5, ".").alias("v"))
    assert collect_column(result, "v") == ["hi..."]
    assert isinstance(result.to_spark().schema["v"].dataType, StringType)


def test_rtrim(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    df = Strings.from_df(
        spark.createDataFrame([("  hello  ",)], schema=Strings.generate_schema())
    )
    result = df.select(tsf.rtrim(df.s).alias("v"))
    assert collect_column(result, "v") == ["  hello"]
    assert isinstance(result.to_spark().schema["v"].dataType, StringType)


def test_sha1(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    df = Strings.from_df(
        spark.createDataFrame([("hello",)], schema=Strings.generate_schema())
    )
    result = df.select(tsf.sha1(df.s).alias("v"))
    vals = collect_column(result, "v")
    assert vals == ["aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d"]
    assert isinstance(result.to_spark().schema["v"].dataType, StringType)


def test_sha2(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    df = Strings.from_df(
        spark.createDataFrame([("hello",)], schema=Strings.generate_schema())
    )
    result = df.select(tsf.sha2(df.s, 256).alias("v"))
    vals = collect_column(result, "v")
    assert vals == ["2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"]
    assert isinstance(result.to_spark().schema["v"].dataType, StringType)


def test_soundex(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    df = Strings.from_df(
        spark.createDataFrame(
            [("Robert",), ("Rupert",)], schema=Strings.generate_schema()
        )
    )
    result = df.select(tsf.soundex(df.s).alias("v"))
    vals = collect_column(result, "v")
    # Both Robert and Rupert have the same Soundex code
    assert vals == ["R163", "R163"]
    assert isinstance(result.to_spark().schema["v"].dataType, StringType)


def test_substr(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    df = Strings.from_df(
        spark.createDataFrame([("hello world",)], schema=Strings.generate_schema())
    )
    result = df.select(
        tsf.substr(df.s, 1, 5).alias("first"),
        tsf.substr(df.s, 7).alias("rest"),
    )
    assert collect_column(result, "first") == ["hello"]
    assert collect_column(result, "rest") == ["world"]
    assert isinstance(result.to_spark().schema["first"].dataType, StringType)


def test_substring_index(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    df = Strings.from_df(
        spark.createDataFrame([("a.b.c.d",)], schema=Strings.generate_schema())
    )
    result = df.select(
        tsf.substring_index(df.s, ".", 2).alias("first2"),
        tsf.substring_index(df.s, ".", -2).alias("last2"),
    )
    assert collect_column(result, "first2") == ["a.b"]
    assert collect_column(result, "last2") == ["c.d"]
    assert isinstance(result.to_spark().schema["first2"].dataType, StringType)


def test_translate(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    df = Strings.from_df(
        spark.createDataFrame([("hello",)], schema=Strings.generate_schema())
    )
    result = df.select(tsf.translate(df.s, "aeiou", "AEIOU").alias("v"))
    assert collect_column(result, "v") == ["hEllO"]
    assert isinstance(result.to_spark().schema["v"].dataType, StringType)


def test_unhex(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    df = Strings.from_df(
        spark.createDataFrame([("48656C6C6F",)], schema=Strings.generate_schema())
    )
    result = df.select(tsf.unhex(df.s).alias("v"))
    assert isinstance(result.to_spark().schema["v"].dataType, BinaryType)
    vals = collect_column(result, "v")
    assert bytes(vals[0]) == b"Hello"


def test_url_encode_decode(spark: SparkSession):
    class Strings(typespark.DataFrame):
        s: typespark.String

    class Encoded(typespark.DataFrame):
        v: typespark.String

    df = Strings.from_df(
        spark.createDataFrame([("hello world",)], schema=Strings.generate_schema())
    )
    encoded = Encoded(df, v=tsf.url_encode(df.s)).alias("e")

    assert collect_column(encoded, "v") == ["hello+world"]

    decoded = encoded.select(tsf.url_decode(encoded.v).alias("v"))
    assert collect_column(decoded, "v") == ["hello world"]
    assert isinstance(encoded.to_spark().schema["v"].dataType, StringType)
