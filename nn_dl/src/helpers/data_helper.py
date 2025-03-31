from typing import List


def array_to_csv(array: List[float]) -> str:
    return ",".join(map(str, array)) if array else ""
