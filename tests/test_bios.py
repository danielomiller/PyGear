"""Tests for Game Gear BIOS overlay (bus.py, ports.py, console.py)."""

import os
import tempfile

import pytest

from pygear.cartridge import Cartridge
from pygear.memory.bus import MemoryBus
from pygear.io.ports import IOPorts
from pygear.console import GameGearConsole, _find_bios


# ---------------------------------------------------------------------------
# Minimal stub helpers
# ---------------------------------------------------------------------------

def _make_cart(size: int = 32 * 1024) -> Cartridge:
    """Create a temp .gg file filled with incrementing byte values and return a Cartridge."""
    data = bytes(i & 0xFF for i in range(size))
    with tempfile.NamedTemporaryFile(suffix='.gg', delete=False) as f:
        f.write(data)
        path = f.name
    cart = Cartridge(path)
    os.unlink(path)
    return cart


def _make_bios(fill: int = 0xAB) -> bytes:
    """Return 1 KB of BIOS data filled with *fill*."""
    return bytes([fill] * 0x400)


# ---------------------------------------------------------------------------
# MemoryBus BIOS overlay
# ---------------------------------------------------------------------------

class TestMemoryBusBios:

    def _bus(self) -> MemoryBus:
        return MemoryBus(_make_cart())

    def test_no_bios_reads_cart(self):
        bus = self._bus()
        # Without a BIOS, reads from $0000 come from the cartridge.
        assert bus._bios_active is False
        # Just verify no exception and returns an int.
        val = bus.read(0x0000)
        assert isinstance(val, int)

    def test_load_bios_activates_overlay(self):
        bus = self._bus()
        bios = _make_bios(0xCC)
        bus.load_bios(bios)
        assert bus._bios_active is True
        assert bus.read(0x0000) == 0xCC
        assert bus.read(0x01FF) == 0xCC
        assert bus.read(0x03FF) == 0xCC

    def test_bios_overlay_boundary(self):
        """$0400 must NOT read from BIOS; it comes from the cartridge mapper."""
        bus = self._bus()
        bus.load_bios(_make_bios(0xBB))
        # $03FF is the last BIOS byte; $0400 bypasses the overlay.
        assert bus.read(0x03FF) == 0xBB
        # Cartridge byte at $0400 differs from BIOS fill.
        cart_val = bus.read(0x0400)
        assert cart_val != 0xBB

    def test_bios_truncated_to_1k(self):
        bus = self._bus()
        big = bytes([0xDE] * 0x800)  # 2 KB — should be trimmed to 1 KB
        bus.load_bios(big)
        assert bus.read(0x03FF) == 0xDE

    def test_set_mem_ctrl_bit4_disables_bios(self):
        bus = self._bus()
        bus.load_bios(_make_bios(0x11))
        assert bus._bios_active is True
        bus.set_mem_ctrl(0x10)
        assert bus._bios_active is False
        # After disable, $0000 should no longer return BIOS fill.
        assert bus.read(0x0000) != 0x11

    def test_set_mem_ctrl_without_bit4_keeps_bios(self):
        bus = self._bus()
        bus.load_bios(_make_bios(0x22))
        bus.set_mem_ctrl(0x00)
        assert bus._bios_active is True
        assert bus.read(0x0000) == 0x22

    def test_set_mem_ctrl_other_bits_ignored(self):
        bus = self._bus()
        bus.load_bios(_make_bios(0x33))
        bus.set_mem_ctrl(0xEF)   # all bits except bit 4
        assert bus._bios_active is True

    def test_reset_reenables_bios(self):
        bus = self._bus()
        bus.load_bios(_make_bios(0x44))
        bus.set_mem_ctrl(0x10)
        assert bus._bios_active is False
        bus.reset()
        assert bus._bios_active is True
        assert bus.read(0x0000) == 0x44

    def test_reset_without_bios_leaves_inactive(self):
        bus = self._bus()
        bus.reset()
        assert bus._bios_active is False

    def test_get_state_includes_bios_active(self):
        bus = self._bus()
        bus.load_bios(_make_bios())
        state = bus.get_state()
        assert state['bios_active'] is True

    def test_set_state_restores_bios_active(self):
        bus = self._bus()
        bus.load_bios(_make_bios(0x55))
        state = bus.get_state()
        # Disable, then restore via set_state.
        bus.set_mem_ctrl(0x10)
        assert bus._bios_active is False
        bus.set_state(state)
        assert bus._bios_active is True

    def test_set_state_no_bios_loaded_stays_false(self):
        """set_state with bios_active=True but no BIOS loaded → stays False."""
        bus = self._bus()
        state = bus.get_state()
        state['bios_active'] = True   # inject True without loading a BIOS
        bus.set_state(state)
        assert bus._bios_active is False

    def test_set_state_missing_key_defaults_to_false(self):
        bus = self._bus()
        state = bus.get_state()
        del state['bios_active']
        bus.set_state(state)
        assert bus._bios_active is False


# ---------------------------------------------------------------------------
# IOPorts — port $3E dispatch
# ---------------------------------------------------------------------------

class _StubBus:
    def __init__(self):
        self.last_mem_ctrl = None

    def set_mem_ctrl(self, value: int) -> None:
        self.last_mem_ctrl = value


class _StubVDP:
    def port_read(self, port): return 0xFF
    def port_write(self, port, value): pass


class _StubJoypad:
    def port_00(self): return 0xFF
    def port_c0(self): return 0xFF


class TestIOPortsMemCtrl:

    def test_port_3e_calls_set_mem_ctrl(self):
        stub_bus = _StubBus()
        ports = IOPorts(_StubVDP(), _StubJoypad(), bus=stub_bus)
        ports.write(0x3E, 0x10)
        assert stub_bus.last_mem_ctrl == 0x10

    def test_port_3e_without_bus_no_error(self):
        ports = IOPorts(_StubVDP(), _StubJoypad())
        ports.write(0x3E, 0x10)  # should be silently ignored

    def test_port_3e_value_masked_to_byte(self):
        stub_bus = _StubBus()
        ports = IOPorts(_StubVDP(), _StubJoypad(), bus=stub_bus)
        ports.write(0x3E, 0x110)   # value > 0xFF
        assert stub_bus.last_mem_ctrl == 0x10

    def test_other_ports_not_routed_to_bus(self):
        stub_bus = _StubBus()
        ports = IOPorts(_StubVDP(), _StubJoypad(), bus=stub_bus)
        ports.write(0x7E, 0x00)   # PSG port — should not touch bus
        assert stub_bus.last_mem_ctrl is None


# ---------------------------------------------------------------------------
# _find_bios search helper
# ---------------------------------------------------------------------------

class TestFindBios:

    def test_explicit_path_found(self):
        data = b'\xAB' * 0x400
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            path = f.name
        try:
            result = _find_bios(None, explicit=path)
            assert result == data
        finally:
            os.unlink(path)

    def test_explicit_path_missing_returns_none(self):
        assert _find_bios(None, explicit='/nonexistent/path/bios.gg') is None

    def test_no_bios_file_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_rom = os.path.join(tmpdir, 'game.gg')
            result = _find_bios(fake_rom, explicit=None)
        assert result is None

    def test_bios_found_in_rom_dir(self):
        data = b'\xCC' * 0x400
        with tempfile.TemporaryDirectory() as tmpdir:
            bios_path = os.path.join(tmpdir, 'bios.gg')
            with open(bios_path, 'wb') as f:
                f.write(data)
            fake_rom = os.path.join(tmpdir, 'game.gg')
            result = _find_bios(fake_rom, explicit=None)
        assert result == data

    def test_none_cart_path_searches_home(self):
        # Without a cart path and no BIOS in ~/.pygear, result should be None
        # (or whatever is found there — we just verify it doesn't raise).
        result = _find_bios(None, explicit=None)
        assert result is None or isinstance(result, bytes)


# ---------------------------------------------------------------------------
# GameGearConsole integration
# ---------------------------------------------------------------------------

class TestConsoleNoBiosFlag:

    def test_no_bios_flag_leaves_inactive(self):
        cart = _make_cart()
        console = GameGearConsole(cart, no_bios=True)
        assert console.bus._bios_active is False

    def test_explicit_bios_missing_path_is_graceful(self):
        """A missing explicit bios path should not crash — BIOS just won't load."""
        cart = _make_cart()
        console = GameGearConsole(cart, bios_path='/nonexistent/bios.gg')
        assert console.bus._bios_active is False

    def test_explicit_bios_loads(self):
        cart = _make_cart()
        data = bytes([0xDD] * 0x400)
        with tempfile.NamedTemporaryFile(suffix='.gg', delete=False) as f:
            f.write(data)
            bios_file = f.name
        try:
            console = GameGearConsole(cart, bios_path=bios_file)
            assert console.bus._bios_active is True
            assert console.bus.read(0x0000) == 0xDD
        finally:
            os.unlink(bios_file)

    def test_ports_wired_to_bus(self):
        """port $3E write via console.ports should reach the bus."""
        cart = _make_cart()
        data = bytes([0xEE] * 0x400)
        with tempfile.NamedTemporaryFile(suffix='.gg', delete=False) as f:
            f.write(data)
            bios_file = f.name
        try:
            console = GameGearConsole(cart, bios_path=bios_file)
            assert console.bus._bios_active is True
            console.ports.write(0x3E, 0x10)
            assert console.bus._bios_active is False
        finally:
            os.unlink(bios_file)
