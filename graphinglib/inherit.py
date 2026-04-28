from typing import TypeGuard


class Inherit:
    __slots__ = ()

    def __repr__(self) -> str:
        return "Inherit"

    __str__ = __repr__

    def __copy__(self) -> "Inherit":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "Inherit":
        return self


INHERIT = Inherit()


def is_inherit(value: object) -> TypeGuard[Inherit]:
    return value is INHERIT
