"""VDP test suite.

Structure
---------
TestVDPPorts  — control/data port I/O, VRAM, CRAM, registers, address latch,
                auto-increment, status clear-on-read, V/H counters.

More test classes will be added in subsequent tasks (tile decoder, background
renderer, timing/interrupts).
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pygear.vdp.vdp import VDP, VRAM_SIZE, CRAM_SIZE, NUM_REGS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_vdp() -> VDP:
    v = VDP()
    v.reset()
    return v


def vdp_set_addr(vdp: VDP, addr: int, code: int) -> None:
    """Write the two-byte command word that sets address+code."""
    vdp.port_write(0xBF, addr & 0xFF)
    vdp.port_write(0xBF, ((code & 0x03) << 6) | ((addr >> 8) & 0x3F))


def vdp_write_vram(vdp: VDP, addr: int, data: bytes) -> None:
    """Write *data* bytes to VRAM starting at *addr* (code=01)."""
    vdp_set_addr(vdp, addr, 0b01)
    for b in data:
        vdp.port_write(0xBE, b)


def vdp_read_vram(vdp: VDP, addr: int, length: int) -> list:
    """Read *length* bytes from VRAM starting at *addr* (code=00)."""
    vdp_set_addr(vdp, addr, 0b00)
    return [vdp.port_read(0xBE) for _ in range(length)]


# ---------------------------------------------------------------------------
class TestVDPPorts:
    # -----------------------------------------------------------------------
    # Sanity / reset
    # -----------------------------------------------------------------------

    def test_reset_clears_vram(self):
        vdp = make_vdp()
        vdp.vram[0x100] = 0xAB
        vdp.reset()
        assert vdp.vram[0x100] == 0

    def test_reset_clears_cram(self):
        vdp = make_vdp()
        vdp.cram[4] = 0xFF
        vdp.reset()
        assert vdp.cram[4] == 0

    def test_reset_clears_regs(self):
        vdp = make_vdp()
        vdp.regs[3] = 0x5A
        vdp.reset()
        assert vdp.regs[3] == 0

    def test_reset_clears_status(self):
        vdp = make_vdp()
        vdp.status = 0xFF
        vdp.reset()
        assert vdp.status == 0

    # -----------------------------------------------------------------------
    # VRAM write via data port (code=01)
    # -----------------------------------------------------------------------

    def test_vram_write_single_byte(self):
        vdp = make_vdp()
        vdp_set_addr(vdp, 0x0010, 0b01)
        vdp.port_write(0xBE, 0xAB)
        assert vdp.vram[0x0010] == 0xAB

    def test_vram_write_auto_increment(self):
        vdp = make_vdp()
        vdp_set_addr(vdp, 0x0020, 0b01)
        vdp.port_write(0xBE, 0x11)
        vdp.port_write(0xBE, 0x22)
        vdp.port_write(0xBE, 0x33)
        assert vdp.vram[0x0020] == 0x11
        assert vdp.vram[0x0021] == 0x22
        assert vdp.vram[0x0022] == 0x33

    def test_vram_write_address_wraps(self):
        vdp = make_vdp()
        vdp_set_addr(vdp, VRAM_SIZE - 1, 0b01)
        vdp.port_write(0xBE, 0xAA)
        vdp.port_write(0xBE, 0xBB)
        assert vdp.vram[VRAM_SIZE - 1] == 0xAA
        assert vdp.vram[0]             == 0xBB

    def test_vram_write_code00_also_writes(self):
        # Setting code=00 prefetches addr and increments; subsequent data write
        # lands at addr+1 (not addr, which was consumed by the prefetch).
        vdp = make_vdp()
        vdp_set_addr(vdp, 0x0005, 0b00)
        vdp.port_write(0xBE, 0x77)
        assert vdp.vram[0x0006] == 0x77

    # -----------------------------------------------------------------------
    # VRAM read via data port (code=00)
    # -----------------------------------------------------------------------

    def test_vram_read_prefetch_on_address_set(self):
        # Setting code=00 immediately prefetches byte at addr; first data read
        # returns that prefetched value and fetches addr+1.
        vdp = make_vdp()
        vdp.vram[0x0030] = 0xCD
        vdp.vram[0x0031] = 0xEF
        vdp_set_addr(vdp, 0x0030, 0b00)
        assert vdp.port_read(0xBE) == 0xCD

    def test_vram_read_sequential(self):
        vdp = make_vdp()
        vdp.vram[0x0050] = 0x01
        vdp.vram[0x0051] = 0x02
        vdp.vram[0x0052] = 0x03
        vdp_set_addr(vdp, 0x0050, 0b00)
        assert vdp.port_read(0xBE) == 0x01
        assert vdp.port_read(0xBE) == 0x02
        assert vdp.port_read(0xBE) == 0x03

    def test_vram_read_address_wraps(self):
        vdp = make_vdp()
        vdp.vram[VRAM_SIZE - 1] = 0x55
        vdp.vram[0]             = 0x66
        vdp_set_addr(vdp, VRAM_SIZE - 1, 0b00)
        assert vdp.port_read(0xBE) == 0x55
        assert vdp.port_read(0xBE) == 0x66

    def test_vram_write_then_read_roundtrip(self):
        vdp = make_vdp()
        payload = bytes(range(16))
        vdp_write_vram(vdp, 0x0100, payload)
        result = vdp_read_vram(vdp, 0x0100, 16)
        assert result == list(payload)

    # -----------------------------------------------------------------------
    # CRAM write via data port (code=11)
    # -----------------------------------------------------------------------

    def test_cram_write_single_byte(self):
        vdp = make_vdp()
        vdp_set_addr(vdp, 0x00, 0b11)
        vdp.port_write(0xBE, 0xAB)
        assert vdp.cram[0] == 0xAB

    def test_cram_write_two_bytes_one_colour(self):
        # A GG colour entry is 2 bytes; writing consecutive bytes to CRAM addr 0
        # fills cram[0] and cram[1].
        vdp = make_vdp()
        vdp_set_addr(vdp, 0x00, 0b11)
        vdp.port_write(0xBE, 0x0F)   # low byte: R=F G=0
        vdp.port_write(0xBE, 0x00)   # high byte: B=0
        assert vdp.cram[0] == 0x0F
        assert vdp.cram[1] == 0x00

    def test_cram_write_auto_increment(self):
        vdp = make_vdp()
        vdp_set_addr(vdp, 0x00, 0b11)
        for i in range(CRAM_SIZE):
            vdp.port_write(0xBE, i)
        assert list(vdp.cram) == list(range(CRAM_SIZE))

    def test_cram_address_wraps_at_64(self):
        # CRAM is 64 bytes; address wraps within that window
        vdp = make_vdp()
        vdp_set_addr(vdp, 63, 0b11)
        vdp.port_write(0xBE, 0xAA)   # writes to cram[63]
        vdp.port_write(0xBE, 0xBB)   # address auto-increments; 64 & 0x3F = 0 → cram[0]
        assert vdp.cram[63] == 0xAA
        assert vdp.cram[0]  == 0xBB

    def test_cram_write_does_not_touch_vram(self):
        vdp = make_vdp()
        vdp_set_addr(vdp, 0x00, 0b11)
        vdp.port_write(0xBE, 0xFF)
        assert vdp.vram[0] == 0

    # -----------------------------------------------------------------------
    # Register write via control port (code=10)
    # -----------------------------------------------------------------------

    def test_reg_write_r0(self):
        vdp = make_vdp()
        vdp.port_write(0xBF, 0xE4)          # value for R0
        vdp.port_write(0xBF, (0b10 << 6) | 0)  # code=10, reg=0
        assert vdp.regs[0] == 0xE4

    def test_reg_write_r1(self):
        vdp = make_vdp()
        vdp.port_write(0xBF, 0xA0)
        vdp.port_write(0xBF, (0b10 << 6) | 1)
        assert vdp.regs[1] == 0xA0

    def test_reg_write_all_regs(self):
        vdp = make_vdp()
        for reg in range(NUM_REGS):
            vdp.port_write(0xBF, reg * 11)
            vdp.port_write(0xBF, (0b10 << 6) | reg)
        for reg in range(NUM_REGS):
            assert vdp.regs[reg] == reg * 11

    def test_reg_write_out_of_range_ignored(self):
        # Register 11 (0x0B) does not exist; write should not raise
        vdp = make_vdp()
        vdp.port_write(0xBF, 0xFF)
        vdp.port_write(0xBF, (0b10 << 6) | 11)
        # registers 0–10 must be unchanged
        assert list(vdp.regs) == [0] * NUM_REGS

    # -----------------------------------------------------------------------
    # Address latch behaviour
    # -----------------------------------------------------------------------

    def test_latch_requires_two_writes(self):
        # A single control write does not commit the address
        vdp = make_vdp()
        vdp.port_write(0xBF, 0x40)   # first byte only
        # Latch is pending; data write should still go to old address (0)
        vdp.port_write(0xBE, 0x99)
        assert vdp.vram[0] == 0x99

    def test_data_write_resets_latch(self):
        # After a data write, the next control write starts a fresh latch sequence
        vdp = make_vdp()
        vdp.port_write(0xBF, 0x10)   # first byte: addr low = 0x10
        vdp.port_write(0xBE, 0x00)   # data write resets latch
        # Now write a fresh address 0x0020 with code=01
        vdp.port_write(0xBF, 0x20)
        vdp.port_write(0xBF, (0b01 << 6) | 0x00)
        vdp.port_write(0xBE, 0xCC)
        assert vdp.vram[0x0020] == 0xCC

    def test_data_read_resets_latch(self):
        vdp = make_vdp()
        vdp.vram[0x0010] = 0x77
        vdp.port_write(0xBF, 0x05)   # first byte
        vdp.port_read(0xBE)          # data read resets latch
        # Fresh address sequence for address 0x0010
        vdp_set_addr(vdp, 0x0010, 0b00)
        assert vdp.port_read(0xBE) == 0x77

    def test_status_read_resets_latch(self):
        vdp = make_vdp()
        vdp.port_write(0xBF, 0x05)   # start latch
        vdp.port_read(0xBF)          # status read resets latch
        # Fresh two-byte sequence should now set address 0x0030
        vdp_set_addr(vdp, 0x0030, 0b01)
        vdp.port_write(0xBE, 0xDD)
        assert vdp.vram[0x0030] == 0xDD

    def test_second_control_write_commits_address(self):
        vdp = make_vdp()
        vdp.vram[0x0123] = 0xAB
        vdp_set_addr(vdp, 0x0123, 0b00)
        assert vdp.port_read(0xBE) == 0xAB

    # -----------------------------------------------------------------------
    # Status register
    # -----------------------------------------------------------------------

    def test_status_read_returns_value(self):
        vdp = make_vdp()
        vdp.status = 0xE0
        assert vdp.port_read(0xBF) == 0xE0

    def test_status_cleared_on_read(self):
        vdp = make_vdp()
        vdp.status = 0xFF
        vdp.port_read(0xBF)
        assert vdp.status == 0

    def test_status_read_twice_gives_zero_second_time(self):
        vdp = make_vdp()
        vdp.status = 0x80
        vdp.port_read(0xBF)
        assert vdp.port_read(0xBF) == 0

    # -----------------------------------------------------------------------
    # V-counter
    # -----------------------------------------------------------------------

    def test_vcounter_zero_at_reset(self):
        vdp = make_vdp()
        assert vdp.port_read(0x7E) == 0

    def test_vcounter_linear_region(self):
        vdp = make_vdp()
        for line in range(0, 0xDB):    # 0x00–0xDA inclusive
            vdp._line = line
            assert vdp.port_read(0x7E) == line

    def test_vcounter_at_jump_boundary(self):
        # Line 0xDA → V-counter 0xDA; line 0xDB (=219) → V-counter 0xD5
        vdp = make_vdp()
        vdp._line = 0xDA
        assert vdp.port_read(0x7E) == 0xDA
        vdp._line = 0xDB
        assert vdp.port_read(0x7E) == 0xD5

    def test_vcounter_end_of_frame(self):
        # Line 261 → V-counter 0xFF
        vdp = make_vdp()
        vdp._line = 261
        assert vdp.port_read(0x7E) == 0xFF

    def test_vcounter_non_linear_region(self):
        vdp = make_vdp()
        for line in range(0xDB, 262):  # lines 219–261
            vdp._line = line
            expected = (0xD5 + (line - 0xDB)) & 0xFF
            assert vdp.port_read(0x7E) == expected, f"line={line}"

    # -----------------------------------------------------------------------
    # H-counter
    # -----------------------------------------------------------------------

    def test_hcounter_zero_at_start_of_line(self):
        vdp = make_vdp()
        assert vdp.port_read(0x7F) == 0

    def test_hcounter_increments_with_cycle(self):
        vdp = make_vdp()
        vdp._cycle = 10
        assert vdp.port_read(0x7F) == 5   # 10 >> 1

    def test_hcounter_wraps_at_256(self):
        vdp = make_vdp()
        vdp._cycle = 512
        assert vdp.port_read(0x7F) == 0   # (512 >> 1) & 0xFF = 256 & 0xFF = 0

    # -----------------------------------------------------------------------
    # Unknown port returns 0xFF
    # -----------------------------------------------------------------------

    def test_unknown_port_read_returns_ff(self):
        vdp = make_vdp()
        assert vdp.port_read(0x00) == 0xFF
        assert vdp.port_read(0xC0) == 0xFF
