from copy import deepcopy
from difflib import get_close_matches
from typing import Any, Protocol, TypeVar

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from matplotlib.colors import to_rgba_array

T = TypeVar("T")


def _copy_with_overrides(instance: T, **kwargs: Any) -> T:
    """
    Returns a deep copy of an instance with selected public writable properties overridden.
    """
    class_name = instance.__class__.__name__
    properties: dict[str, property] = {}
    for attr in dir(instance.__class__):
        if attr.startswith("_"):
            continue
        property_ = getattr(instance.__class__, attr, None)
        if isinstance(property_, property):
            properties[attr] = property_
    writable_properties = {
        attr: property_
        for attr, property_ in properties.items()
        if property_.fset is not None
    }
    property_names = list(properties)
    for key in kwargs:
        if key.startswith("_"):
            raise AttributeError(
                f"{class_name} has no public writable property '{key}'."
            )
        if key in writable_properties:
            continue
        if key in properties:
            raise AttributeError(
                f"{class_name}.{key} is a read-only property and cannot be set."
            )
        close_match = get_close_matches(key, property_names, n=1, cutoff=0.6)
        if close_match:
            raise AttributeError(
                f"{class_name} has no public writable property '{key}'. "
                f"Did you mean '{close_match[0]}'?"
            )
        raise AttributeError(f"{class_name} has no public writable property '{key}'.")

    new_copy = deepcopy(instance)
    for key, value in kwargs.items():
        setattr(new_copy, key, value)
    return new_copy


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
