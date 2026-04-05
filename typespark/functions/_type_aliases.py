from typespark import (
    Byte,
    Date,
    Decimal,
    Double,
    Float,
    Int,
    Long,
    Short,
    Timestamp,
)

type Numeric = Byte | Short | Int | Long | Float | Double | Decimal
type DateOrTimestamp = Date | Timestamp
