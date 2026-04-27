"""VDP test suite.

Structure
---------
TestVDPPorts    — control/data port I/O, VRAM, CRAM, registers, address latch,
                  auto-increment, status clear-on-read, V/H counters.
TestTileDecoder — 4bpp planar tile decode, hflip, vflip.

More test classes will be added in subsequent tasks (background renderer,
timing/interrupts).
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pygear.vdp.vdp   import VDP, VRAM_SIZE, CRAM_SIZE, NUM_REGS
from pygear.vdp.tiles import decode_tile, TILE_BYTES


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


# ---------------------------------------------------------------------------
# Helper — build a 32-byte tile in a fresh bytearray
# ---------------------------------------------------------------------------

def make_vram(tile_data: dict = None) -> bytearray:
    """Return a 16 KB VRAM; *tile_data* maps (tile_num, row) → (b0,b1,b2,b3)."""
    vram = bytearray(VRAM_SIZE)
    if tile_data:
        for (tile_num, row), (b0, b1, b2, b3) in tile_data.items():
            off = tile_num * TILE_BYTES + row * 4
            vram[off], vram[off+1], vram[off+2], vram[off+3] = b0, b1, b2, b3
    return vram


class TestTileDecoder:
    # -----------------------------------------------------------------------
    # Output shape
    # -----------------------------------------------------------------------

    def test_returns_8_rows(self):
        vram = make_vram()
        tile = decode_tile(vram, 0)
        assert len(tile) == 8

    def test_each_row_has_8_pixels(self):
        vram = make_vram()
        tile = decode_tile(vram, 0)
        for row in tile:
            assert len(row) == 8

    # -----------------------------------------------------------------------
    # Colour index decoding
    # -----------------------------------------------------------------------

    def test_blank_tile_all_color_0(self):
        vram = make_vram()
        tile = decode_tile(vram, 0)
        assert all(px == 0 for row in tile for px in row)

    def test_plane0_only_gives_color_1(self):
        # All bits of plane 0 set in row 0 → all pixels have colour index 1
        vram = make_vram({(0, 0): (0xFF, 0x00, 0x00, 0x00)})
        tile = decode_tile(vram, 0)
        assert tile[0] == [1] * 8
        assert tile[1] == [0] * 8   # other rows untouched

    def test_plane1_only_gives_color_2(self):
        vram = make_vram({(0, 0): (0x00, 0xFF, 0x00, 0x00)})
        tile = decode_tile(vram, 0)
        assert tile[0] == [2] * 8

    def test_plane2_only_gives_color_4(self):
        vram = make_vram({(0, 0): (0x00, 0x00, 0xFF, 0x00)})
        tile = decode_tile(vram, 0)
        assert tile[0] == [4] * 8

    def test_plane3_only_gives_color_8(self):
        vram = make_vram({(0, 0): (0x00, 0x00, 0x00, 0xFF)})
        tile = decode_tile(vram, 0)
        assert tile[0] == [8] * 8

    def test_all_planes_set_gives_color_15(self):
        vram = make_vram({(0, 0): (0xFF, 0xFF, 0xFF, 0xFF)})
        tile = decode_tile(vram, 0)
        assert tile[0] == [15] * 8

    def test_combined_planes_give_correct_index(self):
        # planes 0+2 set → colour index 5  (0b0101)
        vram = make_vram({(0, 0): (0xFF, 0x00, 0xFF, 0x00)})
        tile = decode_tile(vram, 0)
        assert tile[0] == [5] * 8

    # -----------------------------------------------------------------------
    # Bit/pixel ordering
    # -----------------------------------------------------------------------

    def test_bit7_maps_to_column_0(self):
        # Plane 0 byte 0x80 (only bit 7 set) → only pixel 0 has colour 1
        vram = make_vram({(0, 0): (0x80, 0x00, 0x00, 0x00)})
        tile = decode_tile(vram, 0)
        assert tile[0][0] == 1
        assert tile[0][1:] == [0] * 7

    def test_bit0_maps_to_column_7(self):
        # Plane 0 byte 0x01 (only bit 0 set) → only pixel 7 has colour 1
        vram = make_vram({(0, 0): (0x01, 0x00, 0x00, 0x00)})
        tile = decode_tile(vram, 0)
        assert tile[0][7] == 1
        assert tile[0][:7] == [0] * 7

    def test_each_bit_position_selects_correct_column(self):
        # Set only plane 0; one bit at a time — confirms all 8 column positions
        for bit in range(8):
            col = 7 - bit
            byte = 1 << bit
            vram = make_vram({(0, 0): (byte, 0x00, 0x00, 0x00)})
            tile = decode_tile(vram, 0)
            assert tile[0][col] == 1, f"bit {bit} should light column {col}"
            others = [tile[0][c] for c in range(8) if c != col]
            assert others == [0] * 7, f"only column {col} should be lit"

    def test_independent_planes_per_pixel(self):
        # One pixel per plane, each at a different column
        # plane0 bit7 → col0=1; plane1 bit6 → col1=2;
        # plane2 bit5 → col2=4; plane3 bit4 → col3=8
        vram = make_vram({(0, 0): (0x80, 0x40, 0x20, 0x10)})
        tile = decode_tile(vram, 0)
        assert tile[0][:4] == [1, 2, 4, 8]
        assert tile[0][4:] == [0, 0, 0, 0]

    # -----------------------------------------------------------------------
    # Multiple rows
    # -----------------------------------------------------------------------

    def test_each_row_decoded_independently(self):
        # Row 0: all planes 0xFF (colour 15); rows 1-7: blank
        vram = make_vram({(0, 0): (0xFF, 0xFF, 0xFF, 0xFF)})
        tile = decode_tile(vram, 0)
        assert tile[0] == [15] * 8
        for r in range(1, 8):
            assert tile[r] == [0] * 8

    def test_all_rows_decodable(self):
        # Write a distinct value in plane 0 of every row of tile 0
        td = {}
        for row in range(8):
            td[(0, row)] = (1 << row, 0x00, 0x00, 0x00)
        vram = make_vram(td)
        tile = decode_tile(vram, 0)
        for row in range(8):
            # Only the pixel at column (7 - row) should be lit
            lit_col = 7 - row
            assert tile[row][lit_col] == 1
            assert sum(tile[row]) == 1

    # -----------------------------------------------------------------------
    # Tile number → VRAM offset
    # -----------------------------------------------------------------------

    def test_tile_0_starts_at_byte_0(self):
        vram = bytearray(VRAM_SIZE)
        vram[0] = 0xFF   # plane 0, row 0 of tile 0
        tile = decode_tile(vram, 0)
        assert tile[0] == [1] * 8

    def test_tile_1_starts_at_byte_32(self):
        vram = bytearray(VRAM_SIZE)
        vram[32] = 0xFF  # plane 0, row 0 of tile 1
        tile0 = decode_tile(vram, 0)
        tile1 = decode_tile(vram, 1)
        assert tile0[0] == [0] * 8   # tile 0 unaffected
        assert tile1[0] == [1] * 8

    def test_tile_n_starts_at_byte_n_times_32(self):
        vram = bytearray(VRAM_SIZE)
        for tile_num in (0, 1, 7, 64, 255, 511):
            off = tile_num * TILE_BYTES
            vram[off] = 0xFF
        for tile_num in (0, 1, 7, 64, 255, 511):
            tile = decode_tile(vram, tile_num)
            assert tile[0] == [1] * 8, f"tile {tile_num}"

    # -----------------------------------------------------------------------
    # Horizontal flip
    # -----------------------------------------------------------------------

    def test_hflip_reverses_pixel_order(self):
        # Pixel 0 lit before flip → pixel 7 lit after flip
        vram = make_vram({(0, 0): (0x80, 0x00, 0x00, 0x00)})   # bit7 → col0
        tile = decode_tile(vram, 0, hflip=True)
        assert tile[0][7] == 1
        assert tile[0][:7] == [0] * 7

    def test_hflip_all_rows_mirrored(self):
        # Ascending pattern across a row: col0=1, col7=0 → after hflip col0=0, col7=1
        vram = make_vram({(0, 3): (0xFE, 0x00, 0x00, 0x00)})   # pixels 0-6 lit, 7 dark
        tile_normal = decode_tile(vram, 0)
        tile_hflip  = decode_tile(vram, 0, hflip=True)
        assert tile_normal[3] == tile_hflip[3][::-1]

    def test_hflip_does_not_affect_row_order(self):
        # Row 0 pattern, rows 1-7 blank; hflip only affects columns
        vram = make_vram({(0, 0): (0xFF, 0xFF, 0xFF, 0xFF)})
        tile = decode_tile(vram, 0, hflip=True)
        assert tile[0] == [15] * 8
        for r in range(1, 8):
            assert tile[r] == [0] * 8

    # -----------------------------------------------------------------------
    # Vertical flip
    # -----------------------------------------------------------------------

    def test_vflip_reverses_row_order(self):
        # Only row 0 has data; after vflip it should appear at row 7
        vram = make_vram({(0, 0): (0xFF, 0x00, 0x00, 0x00)})
        tile = decode_tile(vram, 0, vflip=True)
        assert tile[7] == [1] * 8
        for r in range(7):
            assert tile[r] == [0] * 8

    def test_vflip_all_rows_mirrored(self):
        td = {(0, r): (1 << r, 0, 0, 0) for r in range(8)}
        vram = make_vram(td)
        normal = decode_tile(vram, 0)
        flipped = decode_tile(vram, 0, vflip=True)
        assert flipped == normal[::-1]

    def test_vflip_does_not_affect_pixel_order(self):
        # Pixel 0 of row 0 = colour 1; after vflip pixel 0 of row 7 = colour 1
        vram = make_vram({(0, 0): (0x80, 0x00, 0x00, 0x00)})
        tile = decode_tile(vram, 0, vflip=True)
        assert tile[7][0] == 1   # column unchanged
        assert tile[7][1:] == [0] * 7

    # -----------------------------------------------------------------------
    # Both flips combined
    # -----------------------------------------------------------------------

    def test_hflip_vflip_combined(self):
        # Original: row 0 only, pixel 0 only (colour 1)
        # hflip+vflip → should appear at row 7, pixel 7
        vram = make_vram({(0, 0): (0x80, 0x00, 0x00, 0x00)})
        tile = decode_tile(vram, 0, hflip=True, vflip=True)
        assert tile[7][7] == 1
        assert sum(px for row in tile for px in row) == 1

    def test_hflip_vflip_is_180_rotation(self):
        # hflip + vflip = 180° rotation of the tile
        td = {(0, r): (1 << r, 0, 0, 0) for r in range(8)}
        vram = make_vram(td)
        normal   = decode_tile(vram, 0)
        rotated  = decode_tile(vram, 0, hflip=True, vflip=True)
        expected = [row[::-1] for row in normal[::-1]]
        assert rotated == expected


# ---------------------------------------------------------------------------
# Background scanline renderer helpers
# ---------------------------------------------------------------------------

_NAME_TABLE_BASE = 0x3800   # R2 = 0xFF  →  (0xFF & 0x0E) << 10 = 0x3800


def make_bg_vdp() -> VDP:
    """VDP with name table at 0x3800, no scroll, no scroll-lock."""
    vdp = VDP()
    vdp.reset()
    vdp.regs[2] = 0xFF   # name table at 0x3800
    vdp.regs[8] = 0      # H-scroll = 0
    vdp.regs[9] = 0      # V-scroll = 0
    return vdp


def write_tile_row(vdp: VDP, tile_num: int, row: int,
                   b0: int, b1: int, b2: int, b3: int) -> None:
    """Write four plane bytes for one row of a tile into VRAM directly."""
    off = tile_num * 32 + row * 4
    vdp.vram[off], vdp.vram[off+1], vdp.vram[off+2], vdp.vram[off+3] = b0, b1, b2, b3


def write_solid_tile(vdp: VDP, tile_num: int, color: int) -> None:
    """Fill all 8 rows of a tile with a single solid colour (0–15)."""
    b0 = 0xFF if (color & 1) else 0
    b1 = 0xFF if (color & 2) else 0
    b2 = 0xFF if (color & 4) else 0
    b3 = 0xFF if (color & 8) else 0
    for row in range(8):
        write_tile_row(vdp, tile_num, row, b0, b1, b2, b3)


def write_name_entry(vdp: VDP, tile_row: int, tile_col: int, tile_num: int,
                     hflip: bool = False, vflip: bool = False,
                     palette: int = 0, priority: bool = False) -> None:
    """Write a 2-byte little-endian name-table entry into VRAM."""
    off = _NAME_TABLE_BASE + (tile_row * 32 + tile_col) * 2
    lo = tile_num & 0xFF
    hi = (
        ((tile_num >> 8) & 1)
        | (int(hflip)    << 1)
        | (int(vflip)    << 2)
        | (palette       << 3)
        | (int(priority) << 4)
    )
    vdp.vram[off]     = lo
    vdp.vram[off + 1] = hi


class TestBackground:
    # -----------------------------------------------------------------------
    # Output shape
    # -----------------------------------------------------------------------

    def test_render_line_returns_256_elements(self):
        vdp = make_bg_vdp()
        line = vdp.render_line(0)
        assert len(line) == 256

    def test_render_line_elements_are_tuples(self):
        vdp = make_bg_vdp()
        line = vdp.render_line(0)
        assert all(isinstance(px, tuple) and len(px) == 2 for px in line)

    # -----------------------------------------------------------------------
    # Blank / default rendering
    # -----------------------------------------------------------------------

    def test_blank_vram_renders_color_0_no_priority(self):
        vdp = make_bg_vdp()
        line = vdp.render_line(0)
        assert all(px == (0, False) for px in line)

    # -----------------------------------------------------------------------
    # Solid tile colours
    # -----------------------------------------------------------------------

    def test_solid_tile_at_column_0(self):
        vdp = make_bg_vdp()
        write_solid_tile(vdp, 1, color=5)
        write_name_entry(vdp, 0, 0, tile_num=1)
        line = vdp.render_line(0)
        assert all(px[0] == 5 for px in line[:8])   # first tile-column on screen
        assert all(px[0] == 0 for px in line[8:])   # rest untouched

    def test_adjacent_tiles_rendered_side_by_side(self):
        vdp = make_bg_vdp()
        write_solid_tile(vdp, 1, color=3)
        write_solid_tile(vdp, 2, color=7)
        write_name_entry(vdp, 0, 0, tile_num=1)
        write_name_entry(vdp, 0, 1, tile_num=2)
        line = vdp.render_line(0)
        assert all(px[0] == 3 for px in line[0:8])
        assert all(px[0] == 7 for px in line[8:16])
        assert all(px[0] == 0 for px in line[16:])

    def test_tile_rows_correct_for_each_scanline(self):
        # Tile rows 0-7 each hold a distinct colour (1-8)
        vdp = make_bg_vdp()
        for row in range(8):
            write_tile_row(vdp, 1, row, 0xFF if (row+1)&1 else 0,
                                        0xFF if (row+1)&2 else 0,
                                        0xFF if (row+1)&4 else 0,
                                        0xFF if (row+1)&8 else 0)
        write_name_entry(vdp, 0, 0, tile_num=1)
        for screen_line in range(8):
            expected_color = screen_line + 1
            assert vdp.render_line(screen_line)[0][0] == expected_color

    # -----------------------------------------------------------------------
    # Name-table base address from R2
    # -----------------------------------------------------------------------

    def test_name_table_base_r2_default(self):
        # R2=0xFF → base 0x3800; entry at 0x3800 should be read
        vdp = make_bg_vdp()
        write_solid_tile(vdp, 1, color=9)
        write_name_entry(vdp, 0, 0, tile_num=1)   # writes to 0x3800
        assert vdp.render_line(0)[0][0] == 9

    def test_name_table_base_r2_alternate(self):
        # R2=0x06 → (0x06 & 0x0E) << 10 = 6 << 10 = 0x1800
        vdp = make_bg_vdp()
        vdp.regs[2] = 0x06
        write_solid_tile(vdp, 2, color=11)
        # Write name entry at 0x1800 directly
        vdp.vram[0x1800] = 2   # tile 2, lo byte
        vdp.vram[0x1801] = 0   # hi byte
        assert vdp.render_line(0)[0][0] == 11

    # -----------------------------------------------------------------------
    # Horizontal scroll (R8)
    # -----------------------------------------------------------------------

    def test_h_scroll_zero_no_shift(self):
        vdp = make_bg_vdp()
        write_solid_tile(vdp, 1, color=4)
        write_name_entry(vdp, 0, 0, tile_num=1)
        vdp.regs[8] = 0
        assert vdp.render_line(0)[0][0] == 4

    def test_h_scroll_shifts_background_left(self):
        # H-scroll=8 → screen x=0 shows bg column 8 → tile column 1
        vdp = make_bg_vdp()
        write_solid_tile(vdp, 1, color=6)   # tile in column 1 of bg
        write_name_entry(vdp, 0, 1, tile_num=1)
        vdp.regs[8] = 8
        line = vdp.render_line(0)
        assert all(px[0] == 6 for px in line[0:8])

    def test_h_scroll_wraps_at_256(self):
        # H-scroll=248 → screen x=0 shows bg column 248 → tile col 31
        vdp = make_bg_vdp()
        write_solid_tile(vdp, 5, color=2)
        write_name_entry(vdp, 0, 31, tile_num=5)
        vdp.regs[8] = 248
        line = vdp.render_line(0)
        assert all(px[0] == 2 for px in line[0:8])

    def test_h_scroll_fine_pixel_offset(self):
        # Tile 0 (col 0) = color 1, tile 1 (col 1) = color 2; scroll by 4
        # Screen x 0-3 → bg x 4-7 → tile col 0, pixel col 4-7 → color 1
        # Screen x 4-11 → bg x 8-15 → tile col 1 → color 2
        vdp = make_bg_vdp()
        write_solid_tile(vdp, 1, color=1)
        write_solid_tile(vdp, 2, color=2)
        write_name_entry(vdp, 0, 0, tile_num=1)
        write_name_entry(vdp, 0, 1, tile_num=2)
        vdp.regs[8] = 4
        line = vdp.render_line(0)
        assert all(px[0] == 1 for px in line[0:4])
        assert all(px[0] == 2 for px in line[4:12])

    # -----------------------------------------------------------------------
    # Vertical scroll (R9)
    # -----------------------------------------------------------------------

    def test_v_scroll_zero_no_shift(self):
        vdp = make_bg_vdp()
        write_solid_tile(vdp, 1, color=3)
        write_name_entry(vdp, 0, 0, tile_num=1)
        vdp.regs[9] = 0
        assert vdp.render_line(0)[0][0] == 3

    def test_v_scroll_shifts_to_next_tile_row(self):
        # V-scroll=8 → screen line 0 shows bg_y=8 → tile row 1
        vdp = make_bg_vdp()
        write_solid_tile(vdp, 2, color=7)
        write_name_entry(vdp, 1, 0, tile_num=2)   # bg tile row 1
        vdp.regs[9] = 8
        assert vdp.render_line(0)[0][0] == 7

    def test_v_scroll_fine_row_within_tile(self):
        # V-scroll=3 → screen line 0 shows tile row 0, pixel_row=3
        vdp = make_bg_vdp()
        # Row 3 of tile 1 = color 5, all other rows = 0
        write_tile_row(vdp, 1, 3, 0xFF, 0x00, 0xFF, 0x00)  # color 5
        write_name_entry(vdp, 0, 0, tile_num=1)
        vdp.regs[9] = 3
        assert vdp.render_line(0)[0][0] == 5

    def test_v_scroll_wraps_at_224(self):
        # V-scroll=224 → bg_y = (0+224)%224 = 0 → tile row 0
        vdp = make_bg_vdp()
        write_solid_tile(vdp, 1, color=10)
        write_name_entry(vdp, 0, 0, tile_num=1)
        vdp.regs[9] = 224   # full wrap, same as 0
        assert vdp.render_line(0)[0][0] == 10

    # -----------------------------------------------------------------------
    # Palette selection
    # -----------------------------------------------------------------------

    def test_palette_0_uses_cram_0_to_15(self):
        vdp = make_bg_vdp()
        write_solid_tile(vdp, 1, color=7)
        write_name_entry(vdp, 0, 0, tile_num=1, palette=0)
        assert vdp.render_line(0)[0][0] == 7     # CRAM index 7

    def test_palette_1_offsets_by_16(self):
        vdp = make_bg_vdp()
        write_solid_tile(vdp, 1, color=7)
        write_name_entry(vdp, 0, 0, tile_num=1, palette=1)
        assert vdp.render_line(0)[0][0] == 7 + 16   # CRAM index 23

    def test_palette_0_color_0_gives_cram_0(self):
        vdp = make_bg_vdp()
        write_name_entry(vdp, 0, 0, tile_num=0, palette=0)
        assert vdp.render_line(0)[0][0] == 0

    def test_palette_1_color_0_gives_cram_16(self):
        vdp = make_bg_vdp()
        write_name_entry(vdp, 0, 0, tile_num=0, palette=1)
        assert vdp.render_line(0)[0][0] == 16

    # -----------------------------------------------------------------------
    # Priority bit
    # -----------------------------------------------------------------------

    def test_priority_false_by_default(self):
        vdp = make_bg_vdp()
        write_name_entry(vdp, 0, 0, tile_num=0, priority=False)
        assert vdp.render_line(0)[0][1] is False

    def test_priority_true_when_set(self):
        vdp = make_bg_vdp()
        write_name_entry(vdp, 0, 0, tile_num=0, priority=True)
        assert vdp.render_line(0)[0][1] is True

    def test_priority_per_tile(self):
        # Tile at column 0 has priority, column 1 does not
        vdp = make_bg_vdp()
        write_solid_tile(vdp, 1, color=1)
        write_solid_tile(vdp, 2, color=2)
        write_name_entry(vdp, 0, 0, tile_num=1, priority=True)
        write_name_entry(vdp, 0, 1, tile_num=2, priority=False)
        line = vdp.render_line(0)
        assert all(px[1] is True  for px in line[0:8])
        assert all(px[1] is False for px in line[8:16])

    # -----------------------------------------------------------------------
    # Horizontal flip
    # -----------------------------------------------------------------------

    def test_hflip_reverses_pixel_columns(self):
        # Pixel 0 of tile = color 1; after hflip it appears at pixel 7 on screen
        vdp = make_bg_vdp()
        write_tile_row(vdp, 1, 0, 0x80, 0x00, 0x00, 0x00)  # only pixel 0 lit
        write_name_entry(vdp, 0, 0, tile_num=1, hflip=True)
        line = vdp.render_line(0)
        assert line[0][0] == 0    # pixel 0 of screen → was pixel 7 of tile → 0
        assert line[7][0] == 1    # pixel 7 of screen → was pixel 0 of tile → 1

    def test_hflip_false_gives_normal_order(self):
        vdp = make_bg_vdp()
        write_tile_row(vdp, 1, 0, 0x80, 0x00, 0x00, 0x00)
        write_name_entry(vdp, 0, 0, tile_num=1, hflip=False)
        line = vdp.render_line(0)
        assert line[0][0] == 1
        assert line[7][0] == 0

    # -----------------------------------------------------------------------
    # Vertical flip
    # -----------------------------------------------------------------------

    def test_vflip_reverses_pixel_rows(self):
        # Row 0 of tile = color 5; after vflip, scanline 0 shows row 7 (blank)
        vdp = make_bg_vdp()
        write_tile_row(vdp, 1, 0, 0xFF, 0x00, 0xFF, 0x00)  # row 0 = color 5
        write_name_entry(vdp, 0, 0, tile_num=1, vflip=True)
        assert vdp.render_line(0)[0][0] == 0    # row 7 is blank
        assert vdp.render_line(7)[0][0] == 5    # row 0 appears at scanline 7

    def test_vflip_false_gives_normal_row_order(self):
        vdp = make_bg_vdp()
        write_tile_row(vdp, 1, 0, 0xFF, 0x00, 0xFF, 0x00)
        write_name_entry(vdp, 0, 0, tile_num=1, vflip=False)
        assert vdp.render_line(0)[0][0] == 5
        assert vdp.render_line(7)[0][0] == 0

    # -----------------------------------------------------------------------
    # Scroll locks (R0 bits 6 and 7)
    # -----------------------------------------------------------------------

    def test_hscroll_lock_top_lines_ignore_hscroll(self):
        # R0 bit 6 set, H-scroll=8; lines 0-15 should not scroll
        # Tile at bg col 0 = color 3; after scroll normally screen x=0 → bg col 8
        # With lock on line 0: screen x=0 → bg col 0 → color 3
        vdp = make_bg_vdp()
        write_solid_tile(vdp, 1, color=3)
        write_name_entry(vdp, 0, 0, tile_num=1)   # bg column 0
        vdp.regs[8] = 8
        vdp.regs[0] = 0x40    # H-scroll lock
        assert vdp.render_line(0)[0][0] == 3      # locked: shows bg col 0

    def test_hscroll_lock_applies_only_below_line_16(self):
        # Line 16 is NOT locked → should use H-scroll=8 → screen x=0 → bg col 8 → color 0
        vdp = make_bg_vdp()
        write_solid_tile(vdp, 1, color=3)
        write_name_entry(vdp, 2, 0, tile_num=1)   # bg tile row 2 (lines 16-23)
        vdp.regs[8] = 8
        vdp.regs[0] = 0x40
        # bg col 0 is blank; bg col 1 also blank (only col 0 has tile 1, but
        # that's at tile-row 2, col 0). With H-scroll=8 col 1 shows at x=0.
        # Since tile 1 is only at (row2,col0), x=0 at line 16 shows (row2,col1)=0.
        assert vdp.render_line(16)[0][0] == 0     # scrolled: bg col 1 = blank

    def test_hscroll_lock_line_15_is_locked(self):
        vdp = make_bg_vdp()
        write_solid_tile(vdp, 1, color=6)
        write_name_entry(vdp, 1, 0, tile_num=1)   # bg tile row 1 (lines 8-15)
        vdp.regs[8] = 8
        vdp.regs[0] = 0x40
        assert vdp.render_line(15)[0][0] == 6     # locked: shows bg col 0

    def test_vscroll_lock_right_columns_ignore_vscroll(self):
        # R0 bit 7 set, V-scroll=8; screen x>=192 shows line 0 → bg_y=0 → tile row 0
        # Without lock line 0 + V-scroll 8 → bg_y=8 → tile row 1
        vdp = make_bg_vdp()
        write_solid_tile(vdp, 1, color=4)
        write_name_entry(vdp, 0, 24, tile_num=1)  # bg tile row 0, col 24 (screen x 192)
        vdp.regs[9] = 8
        vdp.regs[0] = 0x80    # V-scroll lock
        line = vdp.render_line(0)
        # screen x=192 → bg_x=192 (no H-scroll) → tile col 24, row 0 (locked) → tile 1
        assert line[192][0] == 4

    def test_vscroll_lock_does_not_affect_left_columns(self):
        # Left columns (x < 192) still use V-scroll
        vdp = make_bg_vdp()
        write_solid_tile(vdp, 1, color=4)
        write_name_entry(vdp, 1, 0, tile_num=1)   # bg tile row 1 (V-scrolled into view)
        vdp.regs[9] = 8
        vdp.regs[0] = 0x80
        # Screen x=0, line 0 → bg_y = 0+8 = 8 → tile row 1, col 0 → tile 1 → color 4
        assert vdp.render_line(0)[0][0] == 4
