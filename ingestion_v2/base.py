from typing import List
from schema.ingestion_schema import ContentUnit


def assign_order(units: List[ContentUnit], start: int) -> int:
    idx = start
    for u in units:
        idx += 1
        u.order_index = idx
    return idx