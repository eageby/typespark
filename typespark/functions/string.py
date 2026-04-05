from typing import overload

import pyspark.sql.functions as F

from typespark import Array, Binary, Bool, Column, Int, Integer, String, TypedArrayType, TypedColumn, int_literal

from ._type_aliases import Numeric


def ascii(col: String) -> Int:
    """Returns the ASCII code of the first character of `col`.

    Wrapper for :func:`pyspark.sql.functions.ascii`.
    """
    return Int(F.ascii(col.to_spark()))


def bin(col: Numeric) -> String:
    """Returns the binary representation of `col` as a string.

    Wrapper for :func:`pyspark.sql.functions.bin`.
    """
    return String(F.bin(col.to_spark()))


def btrim(col: String, trim: String | None = None) -> String:
    """Removes leading and trailing characters from `col`. Removes whitespace when `trim` is omitted.

    Wrapper for :func:`pyspark.sql.functions.btrim`.
    """
    return String(F.btrim(col.to_spark(), trim.to_spark() if trim else None))


def char(col: Integer) -> String:
    """Returns the character for the given ASCII code point `col`.

    Wrapper for :func:`pyspark.sql.functions.char`.
    """
    return String(F.char(col.to_spark()))


def char_length(col: String) -> Int:
    """Returns the number of characters in `col`.

    Wrapper for :func:`pyspark.sql.functions.char_length`.
    """
    return Int(F.char_length(col.to_spark()))


def concat(*cols: String | Array) -> String | Array:
    """Concatenates multiple string or array columns.

    Wrapper for :func:`pyspark.sql.functions.concat`.
    """
    if isinstance(cols[0], Array):
        elem_type = cols[0]._elem_type
        return Array(F.concat(*[c.to_spark() for c in cols]), elem_type)
    return String(F.concat(*[c.to_spark() for c in cols]))


def concat_ws(sep: str, *cols: String) -> String:
    """Concatenates string columns using `sep` as separator, skipping nulls.

    Wrapper for :func:`pyspark.sql.functions.concat_ws`.
    """
    return String(F.concat_ws(sep, *[c.to_spark() for c in cols]))


@overload
def contains(left: String, right: String) -> Bool: ...


@overload
def contains(left: Binary, right: Binary) -> Bool: ...


def contains(left: String | Binary, right: String | Binary) -> Bool:
    """Returns true if `left` contains `right`.

    Wrapper for :func:`pyspark.sql.functions.contains`.
    """
    return Bool(F.contains(left.to_spark(), right.to_spark()))


def decode(col: Binary, charset: str) -> String:
    """Decodes a binary column `col` to a string using the given `charset`.

    Wrapper for :func:`pyspark.sql.functions.decode`.
    """
    return String(F.decode(col.to_spark(), charset))


def encode(col: String, charset: str) -> Binary:
    """Encodes string column `col` to binary using the given `charset`.

    Wrapper for :func:`pyspark.sql.functions.encode`.
    """
    return Binary(F.encode(col.to_spark(), charset))


def endswith(col: String, suffix: String) -> Bool:
    """Returns true if `col` ends with `suffix`.

    Wrapper for :func:`pyspark.sql.functions.endswith`.
    """
    return Bool(F.endswith(col.to_spark(), suffix.to_spark()))


def find_in_set(col: String, str: String) -> Int:
    """Returns the 1-based position of `col` in the comma-separated list `str`, or 0 if not found.

    Wrapper for :func:`pyspark.sql.functions.find_in_set`.
    """
    return Int(F.find_in_set(col.to_spark(), str.to_spark()))


def format_number(col: Numeric, d: int) -> String:
    """Formats `col` as a number with `d` decimal places using US locale grouping (e.g. 1,234.56).

    Wrapper for :func:`pyspark.sql.functions.format_number`.
    """
    return String(F.format_number(col.to_spark(), d))


def format_string(format: str, *cols: Column) -> String:
    """Returns a formatted string using printf-style `format` and the given columns.

    Wrapper for :func:`pyspark.sql.functions.format_string`.
    """
    return String(F.format_string(format, *[c.to_spark() for c in cols]))


def hex(col: Column) -> String:
    """Returns the hexadecimal string representation of `col`.

    Wrapper for :func:`pyspark.sql.functions.hex`.
    """
    return String(F.hex(col.to_spark()))


def initcap(col: String) -> String:
    """Returns `col` with the first letter of each word capitalised.

    Wrapper for :func:`pyspark.sql.functions.initcap`.
    """
    return String(F.initcap(col.to_spark()))


def instr(col: String, substr: str) -> Int:
    """Returns the 1-based position of the first occurrence of `substr` in `col`, or 0 if not found.

    Wrapper for :func:`pyspark.sql.functions.instr`.
    """
    return Int(F.instr(col.to_spark(), substr))


def left(col: String, len: Integer | int) -> String:
    """Returns the leftmost `len` characters of `col`.

    Wrapper for :func:`pyspark.sql.functions.left`.
    """
    return String(
        F.left(
            col.to_spark(),
            len.to_spark() if isinstance(len, TypedColumn) else F.lit(len),
        )
    )


def length(col: String | Binary) -> Int:
    """Returns the character length of a string, or the byte length of a binary column.

    Wrapper for :func:`pyspark.sql.functions.length`.
    """
    return Int(F.length(col.to_spark()))


def levenshtein(col1: String, col2: String, threshold: int | None = None) -> Int:
    """Returns the Levenshtein distance between `col1` and `col2`. Returns -1 if the distance exceeds `threshold`.

    Wrapper for :func:`pyspark.sql.functions.levenshtein`.
    """
    return Int(F.levenshtein(col1.to_spark(), col2.to_spark(), threshold))


def locate(substr: str, col: String, pos: int = 1) -> Int:
    """Returns the 1-based position of the first occurrence of `substr` in `col` at or after `pos`.

    Wrapper for :func:`pyspark.sql.functions.locate`.
    """
    return Int(F.locate(substr, col.to_spark(), pos))


def lower(col: String) -> String:
    """Returns `col` converted to lowercase.

    Wrapper for :func:`pyspark.sql.functions.lower`.
    """
    return String(F.lower(col.to_spark()))


def lpad(
    col: String,
    len: Integer | int,
    pad: String | str,
) -> String:
    """Left-pads `col` with `pad` to a total length of `len` characters.

    Wrapper for :func:`pyspark.sql.functions.lpad`.
    """
    return String(
        F.lpad(
            col.to_spark(),
            len.to_spark() if isinstance(len, TypedColumn) else len,
            pad.to_spark() if isinstance(pad, TypedColumn) else pad,
        )
    )


def ltrim(col: String, trim: String | None = None) -> String:
    """Removes leading characters from `col`. Removes whitespace when `trim` is omitted.

    Wrapper for :func:`pyspark.sql.functions.ltrim`.
    """
    return String(F.ltrim(col.to_spark(), trim.to_spark() if trim else None))


def octet_length(col: String | Binary) -> Int:
    """Returns the byte length of `col`.

    Wrapper for :func:`pyspark.sql.functions.octet_length`.
    """
    return Int(F.octet_length(col.to_spark()))


def overlay(
    col: String,
    replace: String,
    pos: Integer | int,
    len: Integer | int = -1,
) -> String:
    """Replaces a substring of `col` starting at `pos` with `replace`. Replaces `len` characters when provided.

    Wrapper for :func:`pyspark.sql.functions.overlay`.
    """
    return String(
        F.overlay(
            col.to_spark(),
            replace.to_spark(),
            pos.to_spark() if isinstance(pos, TypedColumn) else pos,
            len.to_spark() if isinstance(len, TypedColumn) else len,
        )
    )


def regexp_extract(string: String, pattern: str, idx: int) -> String:
    """Extracts the `idx`-th regex group matched by `pattern` from `string`.

    Wrapper for :func:`pyspark.sql.functions.regexp_extract`.
    """
    return String(
        F.regexp_extract(
            string.to_spark(),
            pattern,
            idx,
        )
    )


def regexp_replace(string: String, pattern: str | String, replacement: str | String) -> String:
    """Replaces all substrings of `string` matching `pattern` with `replacement`.

    Wrapper for :func:`pyspark.sql.functions.regexp_replace`.
    """
    return String(
        F.regexp_replace(
            string.to_spark(),
            pattern.to_spark() if isinstance(pattern, Column) else pattern,
            replacement.to_spark() if isinstance(replacement, Column) else replacement,
        )
    )


def repeat(col: String, n: Integer | int) -> String:
    """Returns `col` repeated `n` times.

    Wrapper for :func:`pyspark.sql.functions.repeat`.
    """
    return String(F.repeat(col.to_spark(), n.to_spark() if isinstance(n, TypedColumn) else n))


def replace(src: String, search: String, replace: String | None = None) -> String:
    """Replaces all occurrences of `search` in `src` with `replace`. Removes matches when `replace` is omitted.

    Wrapper for :func:`pyspark.sql.functions.replace`.
    """
    return String(F.replace(src.to_spark(), search.to_spark(), replace.to_spark() if replace else None))


@overload
def reverse(col: String) -> String: ...


@overload
def reverse[T: TypedColumn](col: Array[T]) -> Array[T]: ...


def reverse(col: String | Array) -> String | Array:
    """Reverses the characters of a string column, or the elements of an array column.

    Wrapper for :func:`pyspark.sql.functions.reverse`.
    """
    if isinstance(col, TypedArrayType):
        return Array(F.reverse(col.to_spark()), col._elem_type)
    return String(F.reverse(col.to_spark()))


def right(col: String, len: Integer | int) -> String:
    """Returns the rightmost `len` characters of `col`.

    Wrapper for :func:`pyspark.sql.functions.right`.
    """
    return String(
        F.right(
            col.to_spark(),
            len.to_spark() if isinstance(len, TypedColumn) else F.lit(len),
        )
    )


def rpad(
    col: String,
    len: Integer | int,
    pad: String | str,
) -> String:
    """Right-pads `col` with `pad` to a total length of `len` characters.

    Wrapper for :func:`pyspark.sql.functions.rpad`.
    """
    return String(
        F.rpad(
            col.to_spark(),
            len.to_spark() if isinstance(len, TypedColumn) else len,
            pad.to_spark() if isinstance(pad, TypedColumn) else pad,
        )
    )


def rtrim(col: String, trim: String | None = None) -> String:
    """Removes trailing characters from `col`. Removes whitespace when `trim` is omitted.

    Wrapper for :func:`pyspark.sql.functions.rtrim`.
    """
    return String(F.rtrim(col.to_spark(), trim.to_spark() if trim else None))


def sha1(col: Binary | String) -> String:
    """Returns the SHA-1 hex digest of `col`.

    Wrapper for :func:`pyspark.sql.functions.sha1`.
    """
    return String(F.sha1(col.to_spark()))


def sha2(col: Binary | String, numBits: int) -> String:
    """Returns the SHA-2 hex digest of `col` using `numBits`-bit hash (224, 256, 384, or 512).

    Wrapper for :func:`pyspark.sql.functions.sha2`.
    """
    return String(F.sha2(col.to_spark(), numBits))


def soundex(col: String) -> String:
    """Returns the Soundex phonetic code for `col`.

    Wrapper for :func:`pyspark.sql.functions.soundex`.
    """
    return String(F.soundex(col.to_spark()))


def split(
    col: String,
    pattern: Column | str,
    limit: Int | int = -1,
):
    """Splits `col` around occurrences of `pattern` (regex), returning an array of strings.

    Wrapper for :func:`pyspark.sql.functions.split`.
    """
    return Array(
        F.split(
            col.to_spark(),
            pattern.to_spark() if isinstance(pattern, Column) else pattern,
            limit.to_spark() if isinstance(limit, Column) else limit,
        ),
        String,
    )


def split_part(src: String, delimiter: String, partNum: Int) -> String:
    """Splits `src` by `delimiter` and returns the `partNum`-th part (1-based).

    Wrapper for :func:`pyspark.sql.functions.split_part`.
    """
    return String(F.split_part(src.to_spark(), delimiter.to_spark(), partNum.to_spark()))


def startswith(str: String, prefix: String) -> Bool:
    """Returns true if `str` starts with `prefix`.

    Wrapper for :func:`pyspark.sql.functions.startswith`.
    """
    return Bool(F.startswith(str.to_spark(), prefix.to_spark()))


def substr(
    col: String,
    startPos: Integer | int,
    length: Integer | int | None = None,
) -> String:
    """Returns the substring of `col` from `startPos` (1-based).
    Includes all remaining characters when `length` is omitted.

    Wrapper for :func:`pyspark.sql.functions.substr`.
    """
    args = [
        col.to_spark(),
        startPos.to_spark() if isinstance(startPos, TypedColumn) else int_literal(startPos).to_spark(),
    ]
    if length is not None:
        args.append(length.to_spark() if isinstance(length, TypedColumn) else int_literal(length).to_spark())

    return String(F.substr(*args))


def substring(
    str: String,
    pos: Int | int,
    len: Int | int,
) -> String:
    """Returns the substring of `str` starting at `pos` (1-based) with at most `len` characters.

    Wrapper for :func:`pyspark.sql.functions.substring`.
    """
    return String(
        F.substring(
            str.to_spark(),
            pos.to_spark() if isinstance(pos, TypedColumn) else pos,
            len.to_spark() if isinstance(len, TypedColumn) else len,
        )
    )


def substring_index(col: String, delim: str, count: int) -> String:
    """Returns everything to the left of the `count`-th occurrence of `delim` in `col`.
    Negative `count` counts from the right.

    Wrapper for :func:`pyspark.sql.functions.substring_index`.
    """
    return String(F.substring_index(col.to_spark(), delim, count))


def translate(col: String, matchingString: str, replaceString: str) -> String:
    """Translates characters in `col` by replacing each character in `matchingString` with
    the corresponding character in `replaceString`.

    Wrapper for :func:`pyspark.sql.functions.translate`.
    """
    return String(F.translate(col.to_spark(), matchingString, replaceString))


def trim(col: String, trim: String | None = None):
    """Removes leading and trailing whitespace from `col`, or the characters in `trim` when provided.

    Wrapper for :func:`pyspark.sql.functions.trim`.
    """
    return String(F.trim(col.to_spark(), trim.to_spark() if trim else None))


def unhex(col: String) -> Binary:
    """Decodes a hexadecimal string `col` into binary.

    Wrapper for :func:`pyspark.sql.functions.unhex`.
    """
    return Binary(F.unhex(col.to_spark()))


def upper(col: String) -> String:
    """Returns `col` converted to uppercase.

    Wrapper for :func:`pyspark.sql.functions.upper`.
    """
    return String(F.upper(col.to_spark()))


def url_decode(col: String) -> String:
    """Decodes a URL-encoded string column `col`.

    Wrapper for :func:`pyspark.sql.functions.url_decode`.
    """
    return String(F.url_decode(col.to_spark()))


def url_encode(col: String) -> String:
    """URL-encodes string column `col` using application/x-www-form-urlencoded format.

    Wrapper for :func:`pyspark.sql.functions.url_encode`.
    """
    return String(F.url_encode(col.to_spark()))


def unbase64(col: String) -> Binary:
    """Decodes a Base64-encoded string column `col` into binary.

    Wrapper for :func:`pyspark.sql.functions.unbase64`.
    """
    return Binary(F.unbase64(col.to_spark()))
