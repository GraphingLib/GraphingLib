from typing import Any, Protocol

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from matplotlib.colors import to_rgba_array


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


def get_contrasting_shade(color: str | tuple[int, int, int]) -> str:
    """
    Gives the most contrasting shade (black/white) for a given color. The algorithm used comes from this Stack
    Exchange answer : https://ux.stackexchange.com/a/82068.

    Parameters
    ----------
    color : str or tuple[int, int, int]
        Color that needs to be contrasted. This can either be a known matplotlib color string or a RGB code, given
        as a tuple of integers that take 0-255.

    Returns
    -------
    shade : str
        Shade (black/white) that contrasts the most with the given color.
    """
    if isinstance(color, str):
        color = to_rgba_array(color)[0, :3] * 255

    R, G, B = color

    if R <= 10:
        Rg = R / 3294
    else:
        Rg = (R / 269 + 0.0513) ** 2.4

    if G <= 10:
        Gg = G / 3294
    else:
        Gg = (G / 269 + 0.0513) ** 2.4

    if B <= 10:
        Bg = B / 3294
    else:
        Bg = (B / 269 + 0.0513) ** 2.4

    L = 0.2126 * Rg + 0.7152 * Gg + 0.0722 * Bg
    if L < 0.5:
        return "white"
    else:
        return "black"
