# pyright: reportUnnecessaryTypeIgnoreComment=true

"""Typed window usage, checked with pyright by tests/test_typecheck.py.

Lines marked `# type: ignore` must produce an error.
"""

import typespark as ts
import typespark.functions as tsf


class Order(ts.DataFrame):
    order_id: ts.String
    customer_id: ts.String
    amount: ts.Double


class Ranked(ts.DataFrame):
    order_id: ts.String
    rank: ts.Int
    running: ts.Double


def latest(orders: Order) -> Order:
    w = ts.Window.partitionBy(orders.customer_id).orderBy(orders.amount.desc())
    return orders.qualify(tsf.row_number().over(w) == 1)


def ranked(orders: Order) -> Ranked:
    w = ts.Window.partitionBy(orders.customer_id).orderBy(orders.amount)
    return Ranked(
        orders,
        order_id=orders.order_id,
        rank=tsf.rank().over(w),
        running=tsf.sum(orders.amount).over(w),
    )


def wrong_type(orders: Order) -> Ranked:
    w = ts.Window.partitionBy(orders.customer_id)
    return Ranked(
        orders,
        order_id=orders.order_id,
        rank=tsf.percent_rank().over(w),  # type: ignore
        running=tsf.lag(orders.order_id).over(w),  # type: ignore
    )
