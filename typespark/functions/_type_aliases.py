from typespark import (
    Column,
    Date,
    Decimal,
    Double,
    Float,
    Int,
    Long,
    Timestamp,
    TimestampNTZ,
)

type Numeric = Int | Long | Float | Double | Decimal | Column
type AnyTimestamp = Timestamp | TimestampNTZ
type DateOrTimestamp = Date | Timestamp | TimestampNTZ
