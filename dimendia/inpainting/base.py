"""Inpainter interface and backend factory (Stage D).

An inpainter reconstructs the pixels hidden behind nearer layers so that, when the
primary object is pushed forward, the revealed background looks natural. The
per-frame interface is stateful so backends can propagate texture across time and
suppress flicker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from dimendia.config import InpaintingBackend
from dimendia.logging import get_logger
from dimendia.types import DepthMap, Frame, Mask

log = get_logger(__name__)


class Inpainter(ABC):
    name: str = "base"

    @abstractmethod
    def inpaint(self, frame: Frame, mask: Mask, depth: DepthMap | None = None) -> Frame:
        """Fill the ``True`` region of ``mask`` in ``frame`` and return RGB uint8.

        ``depth`` is an optional ``[0, 1]`` (1 == near) map letting backends bias
        the fill toward background-colored source pixels.
        """
        raise NotImplementedError

    def reset(self) -> None:  # noqa: B027 - optional hook, concrete no-op by design
        """Clear per-clip temporal state."""


def build_inpainter(backend: InpaintingBackend = InpaintingBackend.AUTO) -> Inpainter:
    from dimendia.inpainting.classical import ClassicalInpainter

    def classical() -> Inpainter:
        log.info("inpainting backend: classical (Telea + temporal propagation)")
        return ClassicalInpainter()

    if backend == InpaintingBackend.CLASSICAL:
        return classical()

    if backend in (InpaintingBackend.AUTO, InpaintingBackend.PROPAINTER):
        try:
            from dimendia.inpainting.propainter_adapter import ProPainterAdapter

            adapter = ProPainterAdapter()
            log.info("inpainting backend: ProPainter")
            return adapter
        except Exception as exc:  # noqa: BLE001 - graceful fallback
            log.warning("ProPainter unavailable (%s)", exc)

    return classical()
