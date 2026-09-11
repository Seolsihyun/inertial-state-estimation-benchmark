from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timer():
    start = time.perf_counter()
    result = {"start_time": start, "elapsed_s": 0.0}
    try:
        yield result
    finally:
        end = time.perf_counter()
        result["end_time"] = end
        result["elapsed_s"] = end - start
