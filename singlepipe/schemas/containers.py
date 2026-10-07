"""Containers of predictions.

``PredictionList`` is a value type, not a pydantic model: an immutable,
ordered sequence of predictions with the operations every task needs
(top-k, filters). It plugs into pydantic through
``__get_pydantic_core_schema__``, so it works as a field type in output schemas
and in FastAPI.

Task-specific operations go to subclasses, created only when a task needs an
operation the base cannot express, for example
``class DetectionList(PredictionList[Detection])`` with box-based methods.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Self, TypeVar, get_args, get_origin, overload

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from singlepipe.schemas.primitives import Prediction

# Covariant: the container only gives predictions out, so
# PredictionList[Detection] can be used where PredictionList[Prediction] is expected.
PredictionT_co = TypeVar("PredictionT_co", bound=Prediction, covariant=True)

_NO_SCORES_MESSAGE = (
    "Operation requires confidence scores, but some predictions have "
    "score=None. This backend did not provide confidence values."
)
_REPR_LIMIT = 3


class PredictionList(Sequence[PredictionT_co]):
    """Immutable ordered sequence of predictions.

    Behaves like a read-only sequence: ``len``, indexing, slicing (returns the
    same container type), iteration, ``in``. There is no ``append`` or any
    other mutation; every operation returns a new container.

    Score-based operations (``top``, ``top1``, ``filter_by_score``) require
    every prediction to have a score and raise ``ValueError`` otherwise.

    In pydantic schemas the field accepts a list of predictions or of dicts and
    is dumped as a list of dicts; the JSON schema is an array of predictions.
    """

    __slots__ = ("_items",)

    def __init__(self, items: Iterable[PredictionT_co] = ()) -> None:
        self._items: tuple[PredictionT_co, ...] = tuple(items)
        for item in self._items:
            if not isinstance(item, Prediction):
                raise TypeError(
                    f"PredictionList items must be Prediction instances, "
                    f"got {type(item).__name__}"
                )

    # --------------------------------------------------------- sequence

    @overload
    def __getitem__(self, index: int) -> PredictionT_co: ...
    @overload
    def __getitem__(self, index: slice) -> Self: ...
    def __getitem__(self, index: int | slice) -> PredictionT_co | Self:
        if isinstance(index, slice):
            return type(self)(self._items[index])
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[PredictionT_co]:
        # Sequence would iterate via __getitem__ until IndexError; this is faster.
        return iter(self._items)

    # ----------------------------------------------- equality, hash, repr

    def __eq__(self, other: object) -> bool:
        """Equal to another PredictionList with the same items in the same order.

        Not equal to a plain list or tuple, even with the same items.
        """
        if not isinstance(other, PredictionList):
            return NotImplemented
        return self._items == other._items

    def __hash__(self) -> int:
        return hash(self._items)

    def __repr__(self) -> str:
        shown = ", ".join(repr(item) for item in self._items[:_REPR_LIMIT])
        if len(self._items) > _REPR_LIMIT:
            shown += f", ... ({len(self._items)} total)"
        return f"{type(self).__name__}([{shown}])"

    # ------------------------------------------------------------ reading

    @property
    def labels(self) -> list[str]:
        """Labels of all predictions, in order."""
        return [item.label for item in self._items]

    @property
    def top1(self) -> PredictionT_co | None:
        """Highest-scoring prediction, or None if the container is empty.

        Raises ValueError if predictions have no scores; use ``[0]`` for those.
        For multi-label outputs a single "best" label may be misleading;
        prefer ``filter_by_score`` there.
        """
        if not self._items:
            return None
        return self.top(1)[0]

    # ----------------------------------------------------- transformations

    def top(self, k: int = 1) -> Self:
        """Return the k highest-scoring predictions, best first.

        Ties keep their original order. Raises ValueError if ``k <= 0`` or if
        any prediction has score=None.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        ranked = sorted(self._scored(), key=lambda pair: pair[0], reverse=True)
        return type(self)(item for _, item in ranked[:k])

    def filter(self, predicate: Callable[[PredictionT_co], bool]) -> Self:
        """Return predictions for which ``predicate`` is true, in order.

        Example: ``detections.filter(lambda d: d.box.area > 1000)``.
        """
        return type(self)(item for item in self._items if predicate(item))

    def filter_by_score(self, threshold: float) -> Self:
        """Return predictions with ``score >= threshold``, in order.

        Raises ValueError if any prediction has score=None.
        """
        return type(self)(item for score, item in self._scored() if score >= threshold)

    def filter_by_label(self, labels: str | Iterable[str]) -> Self:
        """Return predictions whose label is in ``labels``, in order.

        A single string is one label, not an iterable of characters.
        """
        wanted = {labels} if isinstance(labels, str) else set(labels)
        return type(self)(item for item in self._items if item.label in wanted)

    def _scored(self) -> list[tuple[float, PredictionT_co]]:
        """Pair each prediction with its score, failing if any score is missing."""
        scored: list[tuple[float, PredictionT_co]] = []
        for item in self._items:
            if item.score is None:
                raise ValueError(_NO_SCORES_MESSAGE)
            scored.append((item.score, item))
        return scored

    # ------------------------------------------------------------ pydantic

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Validate as ``list[item type]`` and wrap; dump as a list of dicts.

        The item type comes from the annotation (``PredictionList[Detection]``)
        or, for a subclass such as ``DetectionList(PredictionList[Detection])``,
        from its base class. A bare ``PredictionList`` holds ``Prediction``.
        """
        container = get_origin(source) or source
        list_schema = handler.generate_schema(list[_item_type(source)])  # type: ignore[misc]

        return core_schema.no_info_before_validator_function(
            _unwrap,
            core_schema.no_info_after_validator_function(container, list_schema),
            serialization=core_schema.plain_serializer_function_ser_schema(
                list, return_schema=list_schema
            ),
        )


def _unwrap(value: object) -> object:
    """Turn a PredictionList into a list, so its items are validated again.

    Without this, a PredictionList[Prediction] would be accepted unchecked by
    a field annotated PredictionList[Detection].
    """
    return list(value) if isinstance(value, PredictionList) else value


def _item_type(source: Any) -> type[Prediction]:
    """Find the prediction type a PredictionList annotation holds."""
    args = get_args(source)
    if not args:
        for base in getattr(source, "__orig_bases__", ()):
            origin = get_origin(base)
            if isinstance(origin, type) and issubclass(origin, PredictionList):
                args = get_args(base)
                break

    item = args[0] if args else Prediction
    if isinstance(item, TypeVar):
        item = item.__bound__ or Prediction
    if not (isinstance(item, type) and issubclass(item, Prediction)):
        raise TypeError(
            f"PredictionList items must be a Prediction subclass, got {item!r}"
        )
    return item