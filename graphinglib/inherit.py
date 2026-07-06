from typing import Any, NoReturn, TypeGuard, TypeVar, cast

T = TypeVar("T")


class Inherit:
    __slots__ = ()

    def __repr__(self) -> str:
        return "Inherit"

    __str__ = __repr__

    def __copy__(self) -> "Inherit":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "Inherit":
        return self

    def __bool__(self) -> NoReturn:
        raise TypeError(
            "The INHERIT sentinel cannot be used in a boolean context. Use "
            "is_inherit(value) to test for it, or resolve_or(value, default) to "
            "obtain a concrete value."
        )


INHERIT = Inherit()

type Styled[V] = V | Inherit
"""A style parameter that either holds a concrete value or defers to the figure style."""


def is_inherit(value: object) -> TypeGuard[Inherit]:
    return value is INHERIT


def resolved(value: Styled[T]) -> T:
    """
    Asserts that a style parameter has been resolved and narrows its type.

    Raises if the value is still the ``INHERIT`` sentinel; only use this where style
    resolution is guaranteed to have run.
    """
    if is_inherit(value):
        raise TypeError("Style parameter is still INHERIT; it has not been resolved.")
    # The type checker cannot narrow a TypeVar union through the TypeGuard above.
    return cast(T, value)


def resolve_or(value: Styled[T], default: T) -> T:
    """Returns the value, or the given default if it is still the ``INHERIT`` sentinel."""
    if is_inherit(value):
        return default
    return cast(T, value)


def strip_inherit(params: dict[str, Any]) -> dict[str, Any]:
    """
    Returns a copy of a keyword-argument dict without the ``INHERIT``-valued entries.

    Uses identity to detect the sentinel, which is safe for array-valued parameters
    (unlike ``!=``, which numpy arrays broadcast into an ambiguous element-wise result).
    """
    return {key: value for key, value in params.items() if not is_inherit(value)}
