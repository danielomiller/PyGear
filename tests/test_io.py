"""Tests for pygear.io: Joypad and IOPorts."""
import pytest
from pygear.io.joypad import (
    Joypad, START, UP, DOWN, LEFT, RIGHT, BUTTON1, BUTTON2,
)
from pygear.io.ports import IOPorts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockVDP:
    def __init__(self):
        self._reads: dict[int, int] = {}
        self.writes: list[tuple[int, int]] = []

    def set_read(self, port: int, value: int) -> None:
        self._reads[port] = value

    def port_read(self, port: int) -> int:
        return self._reads.get(port, 0xFF)

    def port_write(self, port: int, value: int) -> None:
        self.writes.append((port, value))


class MockJoypad:
    def __init__(self, p00: int = 0xFF, pc0: int = 0xFF):
        self._p00 = p00
        self._pc0 = pc0

    def port_00(self) -> int:
        return self._p00

    def port_c0(self) -> int:
        return self._pc0


def make_io(vdp=None, joypad=None):
    if vdp    is None: vdp    = MockVDP()
    if joypad is None: joypad = MockJoypad()
    return IOPorts(vdp, joypad), vdp, joypad


# ---------------------------------------------------------------------------
# TestJoypad
# ---------------------------------------------------------------------------

class TestJoypad:
    # --- defaults -----------------------------------------------------------

    def test_default_port_00_all_released(self):
        assert Joypad().port_00() == 0xFF

    def test_default_port_c0_all_released(self):
        assert Joypad().port_c0() == 0xFF

    # --- START (port 0x00 bit 7) -------------------------------------------

    def test_start_pressed_clears_bit7(self):
        j = Joypad()
        j.press(START)
        assert j.port_00() == 0x7F

    def test_start_released_restores_bit7(self):
        j = Joypad()
        j.press(START)
        j.release(START)
        assert j.port_00() == 0xFF

    def test_start_does_not_affect_port_c0(self):
        j = Joypad()
        j.press(START)
        assert j.port_c0() == 0xFF

    # --- directional / fire buttons (port 0xC0 bits 0–5) -------------------

    def test_up_pressed_clears_bit0(self):
        j = Joypad()
        j.press(UP)
        assert j.port_c0() == 0xFE

    def test_down_pressed_clears_bit1(self):
        j = Joypad()
        j.press(DOWN)
        assert j.port_c0() == 0xFD

    def test_left_pressed_clears_bit2(self):
        j = Joypad()
        j.press(LEFT)
        assert j.port_c0() == 0xFB

    def test_right_pressed_clears_bit3(self):
        j = Joypad()
        j.press(RIGHT)
        assert j.port_c0() == 0xF7

    def test_button1_pressed_clears_bit4(self):
        j = Joypad()
        j.press(BUTTON1)
        assert j.port_c0() == 0xEF

    def test_button2_pressed_clears_bit5(self):
        j = Joypad()
        j.press(BUTTON2)
        assert j.port_c0() == 0xDF

    def test_directions_do_not_affect_port_00(self):
        j = Joypad()
        for b in [UP, DOWN, LEFT, RIGHT, BUTTON1, BUTTON2]:
            j.press(b)
        assert j.port_00() == 0xFF

    # --- multiple simultaneous presses -------------------------------------

    def test_two_directions_both_bits_clear(self):
        j = Joypad()
        j.press(UP)
        j.press(RIGHT)
        # bit 0 and bit 3 cleared
        assert j.port_c0() == 0xFF ^ 0x01 ^ 0x08

    def test_all_c0_buttons_pressed_bits5_0_clear(self):
        j = Joypad()
        for b in [UP, DOWN, LEFT, RIGHT, BUTTON1, BUTTON2]:
            j.press(b)
        # bits 5-0 all clear; bits 7-6 always 1
        assert j.port_c0() == 0xC0

    def test_start_and_direction_independent(self):
        j = Joypad()
        j.press(START)
        j.press(DOWN)
        assert j.port_00() == 0x7F
        assert j.port_c0() == 0xFD

    # --- release -----------------------------------------------------------

    def test_release_reverts_single_bit_among_multiple(self):
        j = Joypad()
        j.press(LEFT)
        j.press(RIGHT)
        j.release(LEFT)
        assert j.port_c0() == 0xF7  # only RIGHT (bit 3) still clear

    def test_release_unpressed_button_no_effect(self):
        j = Joypad()
        j.release(DOWN)
        assert j.port_c0() == 0xFF

    # --- press idempotency -------------------------------------------------

    def test_press_same_button_twice_idempotent(self):
        j = Joypad()
        j.press(UP)
        j.press(UP)
        assert j.port_c0() == 0xFE

    # --- reset -------------------------------------------------------------

    def test_reset_clears_all_buttons(self):
        j = Joypad()
        j.press(START)
        j.press(UP)
        j.press(BUTTON2)
        j.reset()
        assert j.port_00() == 0xFF
        assert j.port_c0() == 0xFF

    def test_reset_on_fresh_joypad_is_safe(self):
        j = Joypad()
        j.reset()
        assert j.port_00() == 0xFF
        assert j.port_c0() == 0xFF

    # --- return value bounds -----------------------------------------------

    def test_port_00_always_byte(self):
        j = Joypad()
        j.press(START)
        assert 0 <= j.port_00() <= 0xFF

    def test_port_c0_always_byte(self):
        j = Joypad()
        for b in [UP, DOWN, LEFT, RIGHT, BUTTON1, BUTTON2]:
            j.press(b)
        assert 0 <= j.port_c0() <= 0xFF


# ---------------------------------------------------------------------------
# TestIOPorts
# ---------------------------------------------------------------------------

class TestIOPorts:
    # --- read routing -------------------------------------------------------

    def test_read_port_00_routes_to_joypad(self):
        io, _, _ = make_io(joypad=MockJoypad(p00=0x7F))
        assert io.read(0x00) == 0x7F

    def test_read_port_7e_routes_to_vdp(self):
        vdp = MockVDP()
        vdp.set_read(0x7E, 0xAB)
        io, _, _ = make_io(vdp=vdp)
        assert io.read(0x7E) == 0xAB

    def test_read_port_7f_routes_to_vdp(self):
        vdp = MockVDP()
        vdp.set_read(0x7F, 0x42)
        io, _, _ = make_io(vdp=vdp)
        assert io.read(0x7F) == 0x42

    def test_read_port_be_routes_to_vdp(self):
        vdp = MockVDP()
        vdp.set_read(0xBE, 0x55)
        io, _, _ = make_io(vdp=vdp)
        assert io.read(0xBE) == 0x55

    def test_read_port_bf_routes_to_vdp(self):
        vdp = MockVDP()
        vdp.set_read(0xBF, 0x80)
        io, _, _ = make_io(vdp=vdp)
        assert io.read(0xBF) == 0x80

    def test_read_port_c0_routes_to_joypad(self):
        io, _, _ = make_io(joypad=MockJoypad(pc0=0xEF))
        assert io.read(0xC0) == 0xEF

    def test_read_port_dc_routes_to_joypad(self):
        io, _, _ = make_io(joypad=MockJoypad(pc0=0xDF))
        assert io.read(0xDC) == 0xDF

    def test_read_port_c1_returns_ff(self):
        io, _, _ = make_io()
        assert io.read(0xC1) == 0xFF

    def test_read_port_dd_returns_ff(self):
        io, _, _ = make_io()
        assert io.read(0xDD) == 0xFF

    def test_read_unknown_ports_return_ff(self):
        io, _, _ = make_io()
        for port in [0x01, 0x10, 0x3F, 0x40, 0x80, 0xA0, 0xFE]:
            assert io.read(port) == 0xFF, f"port {port:#04x} should return 0xFF"

    # --- read uses 8-bit port masking --------------------------------------

    def test_read_port_masked_to_8_bit(self):
        io, _, _ = make_io(joypad=MockJoypad(p00=0x7F))
        assert io.read(0x100) == 0x7F   # 0x100 & 0xFF == 0x00

    # --- write routing ------------------------------------------------------

    def test_write_port_be_routes_to_vdp(self):
        io, vdp, _ = make_io()
        io.write(0xBE, 0x42)
        assert vdp.writes == [(0xBE, 0x42)]

    def test_write_port_bf_routes_to_vdp(self):
        io, vdp, _ = make_io()
        io.write(0xBF, 0x81)
        assert vdp.writes == [(0xBF, 0x81)]

    def test_write_multiple_vdp_calls_ordered(self):
        io, vdp, _ = make_io()
        io.write(0xBF, 0x10)
        io.write(0xBF, 0x81)
        io.write(0xBE, 0xFF)
        assert vdp.writes == [(0xBF, 0x10), (0xBF, 0x81), (0xBE, 0xFF)]

    def test_write_unknown_port_not_dispatched_to_vdp(self):
        io, vdp, _ = make_io()
        for port in [0x00, 0x7E, 0x7F, 0xC0, 0xC1, 0xDC, 0xDD, 0xFE]:
            io.write(port, 0x00)
        assert vdp.writes == []

    # --- write uses 8-bit port and value masking ---------------------------

    def test_write_port_masked_to_8_bit(self):
        io, vdp, _ = make_io()
        io.write(0x1BE, 0x99)   # 0x1BE & 0xFF == 0xBE
        assert vdp.writes == [(0xBE, 0x99)]

    def test_write_value_masked_to_8_bit(self):
        io, vdp, _ = make_io()
        io.write(0xBE, 0x1FF)   # 0x1FF & 0xFF == 0xFF
        assert vdp.writes == [(0xBE, 0xFF)]

    # --- integration with real Joypad --------------------------------------

    def test_real_joypad_start_press_read_via_port_00(self):
        j = Joypad()
        j.press(START)
        io, _, _ = make_io(joypad=j)
        assert io.read(0x00) == 0x7F

    def test_real_joypad_direction_read_via_port_c0_and_dc(self):
        j = Joypad()
        j.press(RIGHT)
        io, _, _ = make_io(joypad=j)
        assert io.read(0xC0) == 0xF7
        assert io.read(0xDC) == 0xF7

    def test_real_joypad_reset_restores_ff_on_both_mirror_ports(self):
        j = Joypad()
        j.press(UP)
        j.press(BUTTON1)
        io, _, _ = make_io(joypad=j)
        j.reset()
        assert io.read(0xC0) == 0xFF
        assert io.read(0xDC) == 0xFF
