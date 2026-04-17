from typespark import (
    Column,
    Date,
    Decimal,
    Double,
    Float,
    Int,
    Long,
    Timestamp,
)

type Numeric = Int | Long | Float | Double | Decimal | Column
type DateOrTimestamp = Date | Timestamp
