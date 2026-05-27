"""High-level Emulator facade for PyGear.

Wraps GameGearConsole so callers can drive the emulator without importing
internal subsystem classes.  numpy is the only runtime dependency; pygame is
not required.

Typical usage::

    import pygear

    with pygear.Emulator("roms/game.gg") as emu:
        for _ in range(300):          # 5 seconds at 60 fps
            emu.press(pygear.RIGHT)
            frame, audio = emu.step()
            emu.release(pygear.RIGHT)
            if frame is not None:
                process(frame)        # numpy (144, 160, 3) uint8
"""

from __future__ import annotations

import os

import numpy as np

from .cartridge import Cartridge
from .console import GameGearConsole


class Emulator:
    """Drive a Game Gear emulator headlessly from Python."""

    def __init__(
        self,
        rom: str | os.PathLike,
        *,
        bios: str | os.PathLike | None = None,
        no_bios: bool = False,
    ) -> None:
        """Load *rom* and power on the console.

        Parameters
        ----------
        rom:
            Path to a .gg ROM file.
        bios:
            Explicit path to the Game Gear BIOS ROM.  If omitted, the BIOS is
            auto-discovered from the ROM directory and ``~/.pygear/``.
        no_bios:
            Skip BIOS emulation entirely and boot directly from the cartridge.
        """
        cart = Cartridge(os.fspath(rom))
        self._console = GameGearConsole(
            cart,
            bios_path=os.fspath(bios) if bios is not None else None,
            no_bios=no_bios,
        )

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def step(self) -> tuple[np.ndarray | None, list[tuple[float, float]]]:
        """Advance one video frame (262 scanlines).

        Returns
        -------
        frame:
            uint8 numpy array of shape ``(144, 160, 3)`` when a new frame is
            ready (once per call under normal operation), or ``None`` if VBlank
            did not fire.
        audio:
            List of ``(left, right)`` float pairs in ``[-1.0, +1.0]`` at
            44 100 Hz (~735 samples per call).
        """
        audio = self._console.step_frame()
        vdp = self._console.vdp
        if vdp.frame_ready:
            frame = vdp.frame
            vdp.frame_ready = False
        else:
            frame = None
        return frame, audio

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def press(self, *buttons: str) -> None:
        """Press one or more buttons.

        Valid button names: ``UP``, ``DOWN``, ``LEFT``, ``RIGHT``,
        ``BUTTON1``, ``BUTTON2``, ``START``.  Use the constants exported from
        the ``pygear`` package rather than raw strings.
        """
        for b in buttons:
            self._console.joypad.press(b)

    def release(self, *buttons: str) -> None:
        """Release one or more buttons."""
        for b in buttons:
            self._console.joypad.release(b)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the console to power-on state."""
        self._console.reset()

    def save_state(self, slot: int = 0) -> str:
        """Serialize the full console state to disk.  Returns the file path."""
        return self._console.save_state(slot)

    def load_state(self, slot: int = 0) -> bool:
        """Restore console state from disk.  Returns ``True`` on success."""
        return self._console.load_state(slot)

    # ------------------------------------------------------------------
    # Context manager — flushes battery-backed SRAM on exit
    # ------------------------------------------------------------------

    def __enter__(self) -> "Emulator":
        return self

    def __exit__(self, *_) -> None:
        self._console.save_sav()

    # ------------------------------------------------------------------
    # Power-user escape hatch
    # ------------------------------------------------------------------

    @property
    def console(self) -> GameGearConsole:
        """The underlying :class:`~pygear.console.GameGearConsole` instance."""
        return self._console
