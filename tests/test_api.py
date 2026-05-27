"""Tests for the Emulator facade (pygear/emulator.py)."""

import os

import numpy as np
import pytest

import pygear
from pygear import (
    Emulator,
    UP, DOWN, LEFT, RIGHT, BUTTON1, BUTTON2, START,
    SCREEN_W, SCREEN_H,
)
from pygear.console import GameGearConsole


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rom_path(tmp_path):
    """32 KB .gg ROM filled with incrementing bytes."""
    data = bytes(i & 0xFF for i in range(32 * 1024))
    p = tmp_path / "test.gg"
    p.write_bytes(data)
    return str(p)


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------

class TestEmulatorStep:

    def test_step_returns_tuple(self, rom_path):
        emu = Emulator(rom_path, no_bios=True)
        result = emu.step()
        assert isinstance(result, tuple) and len(result) == 2

    def test_step_produces_audio(self, rom_path):
        emu = Emulator(rom_path, no_bios=True)
        _, audio = emu.step()
        assert isinstance(audio, list) and len(audio) > 0

    def test_audio_samples_are_stereo_pairs(self, rom_path):
        emu = Emulator(rom_path, no_bios=True)
        _, audio = emu.step()
        left, right = audio[0]
        assert isinstance(left, float) and isinstance(right, float)

    def test_frame_ready_after_vblank(self, rom_path):
        emu = Emulator(rom_path, no_bios=True)
        frame, _ = emu.step()
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (SCREEN_H, SCREEN_W, 3)
        assert frame.dtype == np.uint8

    def test_frame_ready_cleared_after_step(self, rom_path):
        emu = Emulator(rom_path, no_bios=True)
        emu.step()
        assert emu.console.vdp.frame_ready is False


# ---------------------------------------------------------------------------
# press() / release()
# ---------------------------------------------------------------------------

class TestEmulatorInput:

    def test_press_release_does_not_crash(self, rom_path):
        emu = Emulator(rom_path, no_bios=True)
        emu.press(UP)
        emu.release(UP)

    def test_press_multiple_buttons(self, rom_path):
        emu = Emulator(rom_path, no_bios=True)
        emu.press(UP, DOWN, LEFT, RIGHT, BUTTON1, BUTTON2, START)
        emu.release(UP, DOWN, LEFT, RIGHT, BUTTON1, BUTTON2, START)

    def test_press_affects_joypad_state(self, rom_path):
        emu = Emulator(rom_path, no_bios=True)
        emu.press(UP)
        assert UP in emu.console.joypad._pressed

    def test_release_clears_joypad_state(self, rom_path):
        emu = Emulator(rom_path, no_bios=True)
        emu.press(BUTTON1)
        emu.release(BUTTON1)
        assert BUTTON1 not in emu.console.joypad._pressed


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestEmulatorLifecycle:

    def test_reset_works(self, rom_path):
        emu = Emulator(rom_path, no_bios=True)
        emu.step()
        emu.reset()
        frame, audio = emu.step()
        assert isinstance(audio, list)

    def test_reset_clears_frame_state(self, rom_path):
        emu = Emulator(rom_path, no_bios=True)
        emu.step()
        emu.reset()
        # After reset VDP frame_ready should be False before next step
        assert emu.console.vdp.frame_ready is False

    def test_no_bios_flag(self, rom_path):
        emu = Emulator(rom_path, no_bios=True)
        assert emu.console.bus._bios_active is False

    def test_context_manager(self, rom_path):
        with Emulator(rom_path, no_bios=True) as emu:
            emu.step()

    def test_context_manager_returns_emulator(self, rom_path):
        with Emulator(rom_path, no_bios=True) as emu:
            assert isinstance(emu, Emulator)

    def test_console_property(self, rom_path):
        emu = Emulator(rom_path, no_bios=True)
        assert isinstance(emu.console, GameGearConsole)

    def test_save_load_state_roundtrip(self, rom_path):
        emu = Emulator(rom_path, no_bios=True)
        emu.step()
        path = emu.save_state(0)
        assert os.path.exists(path)
        ok = emu.load_state(0)
        assert ok is True

    def test_load_state_missing_slot(self, rom_path):
        emu = Emulator(rom_path, no_bios=True)
        assert emu.load_state(99) is False

    def test_step_after_load_state(self, rom_path):
        emu = Emulator(rom_path, no_bios=True)
        emu.step()
        emu.save_state(0)
        emu.load_state(0)
        frame, audio = emu.step()
        assert isinstance(audio, list)


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------

class TestTopLevelImports:

    def test_import_emulator(self):
        from pygear import Emulator as E
        assert E is Emulator

    def test_import_button_constants(self):
        assert UP      == "UP"
        assert DOWN    == "DOWN"
        assert LEFT    == "LEFT"
        assert RIGHT   == "RIGHT"
        assert BUTTON1 == "BUTTON1"
        assert BUTTON2 == "BUTTON2"
        assert START   == "START"

    def test_import_screen_dimensions(self):
        assert SCREEN_W == 160
        assert SCREEN_H == 144

    def test_import_cartridge(self):
        from pygear import Cartridge as C
        assert C is not None

    def test_all_exports_accessible(self):
        for name in pygear.__all__:
            assert hasattr(pygear, name), f"pygear.{name} missing"
