from typing import Any, Protocol

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class MathematicalObject(Protocol):
    """
    This class implements the __r* (reverse) and __i* (inplace) methods for adding, subtracting, multiplying, dividing
    and raising to a power an object that implements the direct methods.

    .. warning::
        This class works only if the children class implements the `__add__`, `__sub__`, `__mul__`, `__truediv__` and
        `__pow__` methods. Note also that the `__rpow__` method is not implemented by this class.
    """

    def __add__(self, other: Any) -> Self:
        raise NotImplementedError

    def __radd__(self, other: Any) -> Self:
        return self.__add__(other)

    def __iadd__(self, other: Any) -> Self:
        self = self.__add__(other)
        return self

    def __sub__(self, other: Any) -> Self:
        raise NotImplementedError

    def __rsub__(self, other: Any) -> Self:
        return self.__sub__(other) * (-1)

    def __isub__(self, other: Any) -> Self:
        self = self.__sub__(other)
        return self

    def __mul__(self, other: Any) -> Self:
        raise NotImplementedError

    def __rmul__(self, other: Any) -> Self:
        return self.__mul__(other)

    def __imul__(self, other: Any) -> Self:
        self = self.__mul__(other)
        return self

    def __truediv__(self, other: Any) -> Self:
        raise NotImplementedError

    def __rtruediv__(self, other: Any) -> Self:
        return self.__truediv__(other) ** (-1)

    def __itruediv__(self, other: Any) -> Self:
        self = self.__truediv__(other)
        return self

    def __pow__(self, other: Any) -> Self:
        raise NotImplementedError

    def __rpow__(self, other: Any) -> Self:
        raise NotImplementedError

    def __ipow__(self, other: Any) -> Self:
        self = self.__pow__(other)
        return self
