import enum
from typing import Iterable, List, Tuple, Union


def tuples_list(a: Iterable, b: Iterable) -> List[Tuple]:
    return list(zip(a, b))


def parse_str_from_enum(lst: List[Union[enum.Enum, str]]) -> List[str]:
    def parse(x: Union[enum.Enum, str]) -> str:
        if isinstance(x, enum.Enum):
            return str(x.value)
        else:
            return x

    return list(map(parse, lst))


__all__ = [
    "tuples_list",
    "parse_str_from_enum"
]
