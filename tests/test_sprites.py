"""Tests for pygear.vdp.sprites and VDP sprite compositing."""
import numpy as np
import pytest
from pygear.vdp.sprites import sat_base, parse_sat, sprites_on_line, render_sprite_line
from pygear.vdp.vdp import VDP, CYCLES_PER_LINE, ACTIVE_LINES, ScanlineView

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_vram():
    return bytearray(0x4000)


def make_regs(**kw):
    """Return an 11-byte register file with named registers set, e.g. r1=0x02."""
    regs = bytearray(11)
    for name, val in kw.items():
        regs[int(name[1:])] = val
    return regs


def sat_write(vram, regs, n, y, x, tile_num):
    """Write sprite slot *n* into the SAT."""
    base = sat_base(regs)
    vram[(base + n) & 0x3FFF]               = y
    vram[(base + 128 + n * 2)     & 0x3FFF] = x
    vram[(base + 128 + n * 2 + 1) & 0x3FFF] = tile_num


def sat_terminate(vram, regs, n):
    """Write the Y=0xD0 terminator at slot *n*."""
    base = sat_base(regs)
    vram[(base + n) & 0x3FFF] = 0xD0


def make_solid_sprite(x: int, tile_num: int = 0):
    """Return (vram, regs) with one opaque 8×8 sprite (all pixels color 1) at X=x, Y=0."""
    vram = make_vram()
    regs = make_regs(r5=0x7E)
    # Write solid tile 0: all pixels = color index 1 (b0=0xFF, others=0)
    for row in range(8):
        vram[tile_num * 32 + row * 4] = 0xFF   # bitplane 0 all-1 → color index 1
    sat_write(vram, regs, 0, 0, x, tile_num)   # sprite at Y=0, visible on line 1
    sat_terminate(vram, regs, 1)
    return vram, regs


# ---------------------------------------------------------------------------
# TestSATParser
# ---------------------------------------------------------------------------

class TestSATParser:

    # --- sat_base -----------------------------------------------------------

    def test_sat_base_r5_zero(self):
        assert sat_base(make_regs()) == 0x0000

    def test_sat_base_r5_02(self):
        assert sat_base(make_regs(r5=0x02)) == 0x0100

    def test_sat_base_r5_40(self):
        assert sat_base(make_regs(r5=0x40)) == 0x2000

    def test_sat_base_r5_7e(self):
        assert sat_base(make_regs(r5=0x7E)) == 0x3F00

    def test_sat_base_low_bit_of_r5_ignored(self):
        # R5 bit 0 is not used; 0x7F and 0x7E must give the same base
        assert sat_base(make_regs(r5=0x7F)) == sat_base(make_regs(r5=0x7E))

    def test_sat_base_odd_r5_same_as_even(self):
        assert sat_base(make_regs(r5=0x03)) == sat_base(make_regs(r5=0x02))

    # --- parse_sat: terminator behaviour ------------------------------------

    def test_parse_sat_terminator_at_slot_0(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        sat_terminate(vram, regs, 0)
        assert parse_sat(vram, regs) == []

    def test_parse_sat_terminator_after_first_sprite(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        sat_write(vram, regs, 0, 10, 20, 5)
        sat_terminate(vram, regs, 1)
        result = parse_sat(vram, regs)
        assert len(result) == 1

    def test_parse_sat_terminator_mid_list(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        for n in range(3):
            sat_write(vram, regs, n, n, n * 8, n)
        sat_terminate(vram, regs, 3)
        assert len(parse_sat(vram, regs)) == 3

    # --- parse_sat: value fidelity ------------------------------------------

    def test_parse_sat_correct_y_x_tile(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        sat_write(vram, regs, 0, 42, 100, 7)
        sat_terminate(vram, regs, 1)
        y, x, tile_num = parse_sat(vram, regs)[0]
        assert y == 42
        assert x == 100
        assert tile_num == 7

    def test_parse_sat_multiple_sprites_values(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        data = [(10, 20, 1), (30, 40, 2), (50, 60, 3)]
        for n, (y, x, t) in enumerate(data):
            sat_write(vram, regs, n, y, x, t)
        sat_terminate(vram, regs, 3)
        assert parse_sat(vram, regs) == data

    def test_parse_sat_all_64_without_terminator(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        for n in range(64):
            sat_write(vram, regs, n, n, 0, 0)
        # No terminator — should return all 64
        assert len(parse_sat(vram, regs)) == 64

    def test_parse_sat_reads_from_correct_sat_base(self):
        # With R5=0x02 the SAT base is 0x0100; sprites written there
        # should be found; the default base (0x0000) area is unrelated.
        vram = make_vram()
        regs = make_regs(r5=0x02)
        sat_write(vram, regs, 0, 55, 77, 9)
        sat_terminate(vram, regs, 1)
        y, x, tile_num = parse_sat(vram, regs)[0]
        assert (y, x, tile_num) == (55, 77, 9)

    # --- parse_sat: tall mode tile masking ----------------------------------

    def test_parse_sat_tall_mode_clears_tile_bit0_for_odd_tile(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E, r1=0x02)   # tall mode
        sat_write(vram, regs, 0, 0, 0, 7)    # odd tile 7 → should become 6
        sat_terminate(vram, regs, 1)
        _, _, tile_num = parse_sat(vram, regs)[0]
        assert tile_num == 6

    def test_parse_sat_tall_mode_leaves_even_tile_unchanged(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E, r1=0x02)
        sat_write(vram, regs, 0, 0, 0, 6)    # even tile 6 → stays 6
        sat_terminate(vram, regs, 1)
        _, _, tile_num = parse_sat(vram, regs)[0]
        assert tile_num == 6

    def test_parse_sat_tall_mode_applies_to_all_entries(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E, r1=0x02)
        for n in range(4):
            sat_write(vram, regs, n, n, 0, n * 2 + 1)  # tiles 1,3,5,7
        sat_terminate(vram, regs, 4)
        for _, _, tile_num in parse_sat(vram, regs):
            assert tile_num % 2 == 0, f"tile_num {tile_num} has bit 0 set"

    def test_parse_sat_non_tall_mode_preserves_odd_tile(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)             # no tall mode
        sat_write(vram, regs, 0, 0, 0, 7)
        sat_terminate(vram, regs, 1)
        _, _, tile_num = parse_sat(vram, regs)[0]
        assert tile_num == 7                  # odd tile unchanged without tall mode


# ---------------------------------------------------------------------------
# TestSpritesOnLine
# ---------------------------------------------------------------------------

class TestSpritesOnLine:

    # --- visibility window (normal 8px height) ------------------------------

    def test_sprite_first_visible_line(self):
        # Y=10 → first visible on line 11 (dy=0)
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        sat_write(vram, regs, 0, 10, 0, 0)
        sat_terminate(vram, regs, 1)
        visible, _ = sprites_on_line(vram, regs, 11)
        assert len(visible) == 1

    def test_sprite_not_visible_on_line_before_start(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        sat_write(vram, regs, 0, 10, 0, 0)
        sat_terminate(vram, regs, 1)
        visible, _ = sprites_on_line(vram, regs, 10)
        assert visible == []

    def test_sprite_last_visible_line(self):
        # Y=10, height=8 → last visible on line 18 (dy=7)
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        sat_write(vram, regs, 0, 10, 0, 0)
        sat_terminate(vram, regs, 1)
        visible, _ = sprites_on_line(vram, regs, 18)
        assert len(visible) == 1

    def test_sprite_not_visible_after_last_line(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        sat_write(vram, regs, 0, 10, 0, 0)
        sat_terminate(vram, regs, 1)
        visible, _ = sprites_on_line(vram, regs, 19)
        assert visible == []

    # --- dy value in result -------------------------------------------------

    def test_dy_is_zero_on_first_visible_line(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        sat_write(vram, regs, 0, 10, 0, 0)
        sat_terminate(vram, regs, 1)
        (_, _, dy), = sprites_on_line(vram, regs, 11)[0]
        assert dy == 0

    def test_dy_is_correct_mid_sprite(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        sat_write(vram, regs, 0, 10, 0, 0)
        sat_terminate(vram, regs, 1)
        (_, _, dy), = sprites_on_line(vram, regs, 15)[0]
        assert dy == 4    # (15 - 11) = 4

    def test_dy_is_7_on_last_visible_line(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        sat_write(vram, regs, 0, 10, 0, 0)
        sat_terminate(vram, regs, 1)
        (_, _, dy), = sprites_on_line(vram, regs, 18)[0]
        assert dy == 7

    # --- x and tile_num in result -------------------------------------------

    def test_x_and_tile_num_preserved(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        sat_write(vram, regs, 0, 10, 88, 42)
        sat_terminate(vram, regs, 1)
        (x, tile_num, _), = sprites_on_line(vram, regs, 11)[0]
        assert x == 88
        assert tile_num == 42

    # --- tall mode (height = 16) --------------------------------------------

    def test_tall_last_visible_line(self):
        # R1 bit 1 → height 16; Y=10 → last visible on line 26 (dy=15)
        vram = make_vram()
        regs = make_regs(r5=0x7E, r1=0x02)
        sat_write(vram, regs, 0, 10, 0, 0)
        sat_terminate(vram, regs, 1)
        visible, _ = sprites_on_line(vram, regs, 26)
        assert len(visible) == 1

    def test_tall_not_visible_after_last_line(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E, r1=0x02)
        sat_write(vram, regs, 0, 10, 0, 0)
        sat_terminate(vram, regs, 1)
        visible, _ = sprites_on_line(vram, regs, 27)
        assert visible == []

    # --- zoom mode (height = 16) --------------------------------------------

    def test_zoom_last_visible_line(self):
        # R1 bit 0 → zoom; height doubles to 16
        vram = make_vram()
        regs = make_regs(r5=0x7E, r1=0x01)
        sat_write(vram, regs, 0, 10, 0, 0)
        sat_terminate(vram, regs, 1)
        visible, _ = sprites_on_line(vram, regs, 26)
        assert len(visible) == 1

    def test_zoom_not_visible_after_last_line(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E, r1=0x01)
        sat_write(vram, regs, 0, 10, 0, 0)
        sat_terminate(vram, regs, 1)
        visible, _ = sprites_on_line(vram, regs, 27)
        assert visible == []

    # --- tall + zoom (height = 32) ------------------------------------------

    def test_tall_zoom_last_visible_line(self):
        # R1 bits 0+1 → tall + zoom; height = 32
        vram = make_vram()
        regs = make_regs(r5=0x7E, r1=0x03)
        sat_write(vram, regs, 0, 10, 0, 0)
        sat_terminate(vram, regs, 1)
        visible, _ = sprites_on_line(vram, regs, 42)
        assert len(visible) == 1

    def test_tall_zoom_not_visible_after_last_line(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E, r1=0x03)
        sat_write(vram, regs, 0, 10, 0, 0)
        sat_terminate(vram, regs, 1)
        visible, _ = sprites_on_line(vram, regs, 43)
        assert visible == []

    # --- 8-sprite per-line limit and overflow -------------------------------

    def test_exactly_8_sprites_no_overflow(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        for n in range(8):
            sat_write(vram, regs, n, 10, n * 8, n)
        sat_terminate(vram, regs, 8)
        visible, overflow = sprites_on_line(vram, regs, 11)
        assert len(visible) == 8
        assert overflow is False

    def test_9th_sprite_triggers_overflow(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        for n in range(9):
            sat_write(vram, regs, n, 10, n * 8, n)
        sat_terminate(vram, regs, 9)
        visible, overflow = sprites_on_line(vram, regs, 11)
        assert len(visible) == 8
        assert overflow is True

    def test_overflow_visible_list_capped_at_8(self):
        # Even with 12 sprites on the line, visible stays at 8
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        for n in range(12):
            sat_write(vram, regs, n, 10, 0, 0)
        sat_terminate(vram, regs, 12)
        visible, overflow = sprites_on_line(vram, regs, 11)
        assert len(visible) == 8
        assert overflow is True

    # --- terminator stops before limit --------------------------------------

    def test_terminator_stops_before_overflow(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        for n in range(3):
            sat_write(vram, regs, n, 10, n * 8, n)
        sat_terminate(vram, regs, 3)
        visible, overflow = sprites_on_line(vram, regs, 11)
        assert len(visible) == 3
        assert overflow is False

    # --- Y wrap-around ------------------------------------------------------

    def test_y_0xff_visible_on_line_0(self):
        # Y=0xFF → dy = (0 - 0) & 0xFF = 0 → visible on line 0
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        sat_write(vram, regs, 0, 0xFF, 5, 2)
        sat_terminate(vram, regs, 1)
        visible, _ = sprites_on_line(vram, regs, 0)
        assert len(visible) == 1

    def test_y_0xff_not_visible_on_line_8(self):
        # dy = (8 - 0) & 0xFF = 8 ≥ height(8) → not visible
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        sat_write(vram, regs, 0, 0xFF, 0, 0)
        sat_terminate(vram, regs, 1)
        visible, _ = sprites_on_line(vram, regs, 8)
        assert visible == []


# ---------------------------------------------------------------------------
# TestSpriteLineRenderer
# ---------------------------------------------------------------------------

def write_tile_row(vram, tile_num, row, b0=0, b1=0, b2=0, b3=0, tile_base=0):
    """Write the four bitplane bytes for one row of a tile."""
    addr = tile_base + tile_num * 32 + row * 4
    vram[ addr      & 0x3FFF] = b0
    vram[(addr + 1) & 0x3FFF] = b1
    vram[(addr + 2) & 0x3FFF] = b2
    vram[(addr + 3) & 0x3FFF] = b3


def setup_sprite(vram, regs, n, y, x, tile_num):
    """Convenience wrapper: write sprite slot *n* and a terminator at n+1."""
    sat_write(vram, regs, n, y, x, tile_num)
    sat_terminate(vram, regs, n + 1)


class TestSpriteLineRenderer:

    # --- all-transparent defaults ------------------------------------------

    def test_no_sprites_all_transparent(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        sat_terminate(vram, regs, 0)
        sp_cram, sp_has, ov, col = render_sprite_line(vram, regs, 0)
        assert len(sp_has) == 256
        assert not sp_has.any()
        assert ov is False
        assert col is False

    def test_color_0_tile_is_all_transparent(self):
        # Tile 0 is all zeros → every pixel is color index 0 → transparent
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        setup_sprite(vram, regs, 0, 0, 0, 0)    # Y=0 → line 1
        sp_cram, sp_has, _, _ = render_sprite_line(vram, regs, 1)
        assert not sp_has.any()

    # --- palette 1 cram index mapping --------------------------------------

    def test_color_1_maps_to_cram_17(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        write_tile_row(vram, 0, 0, b0=0xFF)    # plane 0 all set → color 1
        setup_sprite(vram, regs, 0, 0, 0, 0)
        sp_cram, sp_has, _, _ = render_sprite_line(vram, regs, 1)
        assert sp_cram[0] == 17 and sp_has[0]

    def test_color_15_maps_to_cram_31(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        write_tile_row(vram, 0, 0, b0=0xFF, b1=0xFF, b2=0xFF, b3=0xFF)
        setup_sprite(vram, regs, 0, 0, 0, 0)
        sp_cram, sp_has, _, _ = render_sprite_line(vram, regs, 1)
        assert sp_cram[0] == 31 and sp_has[0]

    def test_color_2_maps_to_cram_18(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        write_tile_row(vram, 0, 0, b1=0xFF)    # plane 1 only → color 2
        setup_sprite(vram, regs, 0, 0, 0, 0)
        sp_cram, sp_has, _, _ = render_sprite_line(vram, regs, 1)
        assert sp_cram[0] == 18 and sp_has[0]

    # --- X positioning and pixel pattern -----------------------------------

    def test_sprite_at_x10_covers_pixels_10_to_17(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        write_tile_row(vram, 0, 0, b0=0xFF)
        setup_sprite(vram, regs, 0, 0, 10, 0)
        sp_cram, sp_has, _, _ = render_sprite_line(vram, regs, 1)
        assert not sp_has[9]
        assert sp_has[10:18].all()
        assert not sp_has[18]

    def test_sprite_at_x0_covers_pixels_0_to_7(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        write_tile_row(vram, 0, 0, b0=0xFF)
        setup_sprite(vram, regs, 0, 0, 0, 0)
        sp_cram, sp_has, _, _ = render_sprite_line(vram, regs, 1)
        assert sp_has[0:8].all()
        assert not sp_has[8]

    def test_checkerboard_0xAA_odd_cols_transparent(self):
        # 0xAA = 10101010: bit7,5,3,1 set → cols 0,2,4,6 have color 1
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        write_tile_row(vram, 0, 0, b0=0xAA)
        setup_sprite(vram, regs, 0, 0, 0, 0)
        sp_cram, sp_has, _, _ = render_sprite_line(vram, regs, 1)
        for col in range(8):
            if col % 2 == 0:
                assert sp_has[col], f"col {col} should have pixel"
            else:
                assert not sp_has[col], f"col {col} should be transparent"

    # --- tile row selection ------------------------------------------------

    def test_correct_tile_row_used_for_dy(self):
        # dy=3 → tile row 3; only row 3 has data, others are zero
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        write_tile_row(vram, 0, 3, b0=0xFF)    # row 3 has pixels
        setup_sprite(vram, regs, 0, 0, 0, 0)   # Y=0 → line 1+dy
        sp_cram, sp_has, _, _ = render_sprite_line(vram, regs, 4)   # line 4 → dy=3
        assert sp_has[0]

    def test_different_dy_uses_different_row(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        write_tile_row(vram, 0, 2, b0=0xFF)    # only row 2 has data
        setup_sprite(vram, regs, 0, 0, 0, 0)
        _, has_row1, _, _ = render_sprite_line(vram, regs, 2)   # dy=1 → row 1
        _, has_row2, _, _ = render_sprite_line(vram, regs, 3)   # dy=2 → row 2
        assert not has_row1[0]
        assert has_row2[0]

    # --- tile pattern base (R6) --------------------------------------------

    def test_tile_base_r6_0_reads_from_0x0000(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E, r6=0x00)
        write_tile_row(vram, 0, 0, b0=0xFF, tile_base=0x0000)
        setup_sprite(vram, regs, 0, 0, 0, 0)
        sp_cram, sp_has, _, _ = render_sprite_line(vram, regs, 1)
        assert sp_has[0]

    def test_tile_base_r6_4_reads_from_0x2000(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E, r6=0x04)
        write_tile_row(vram, 0, 0, b0=0xFF, tile_base=0x2000)
        setup_sprite(vram, regs, 0, 0, 0, 0)
        sp_cram, sp_has, _, _ = render_sprite_line(vram, regs, 1)
        assert sp_has[0]

    def test_tile_base_r6_4_ignores_0x0000(self):
        # Data at 0x0000 should NOT be used when R6 selects 0x2000
        vram = make_vram()
        regs = make_regs(r5=0x7E, r6=0x04)
        write_tile_row(vram, 0, 0, b0=0xFF, tile_base=0x0000)  # wrong base
        setup_sprite(vram, regs, 0, 0, 0, 0)
        sp_cram, sp_has, _, _ = render_sprite_line(vram, regs, 1)
        assert not sp_has[0]

    # --- tall mode tile selection -------------------------------------------

    def test_tall_upper_tile_used_for_dy_lt_8(self):
        # Tall: dy=0 → actual_row=0 < 8 → use tile 2 (even), row 0
        vram = make_vram()
        regs = make_regs(r5=0x7E, r1=0x02)
        write_tile_row(vram, 2, 0, b0=0xFF)   # upper tile has data
        write_tile_row(vram, 3, 0, b0=0x00)   # lower tile is transparent
        setup_sprite(vram, regs, 0, 0, 0, 2)  # tile 2 (even)
        sp_cram, sp_has, _, _ = render_sprite_line(vram, regs, 1)   # dy=0
        assert sp_has[0]

    def test_tall_lower_tile_used_for_dy_ge_8(self):
        # Tall: dy=8 → actual_row=8 ≥ 8 → use tile 3 (2|1), row 0
        vram = make_vram()
        regs = make_regs(r5=0x7E, r1=0x02)
        write_tile_row(vram, 2, 0, b0=0x00)   # upper transparent
        write_tile_row(vram, 3, 0, b0=0xFF)   # lower has data
        setup_sprite(vram, regs, 0, 0, 0, 2)
        sp_cram, sp_has, _, _ = render_sprite_line(vram, regs, 9)   # dy=8
        assert sp_has[0]

    def test_tall_lower_tile_row_7_for_dy_15(self):
        # dy=15 → actual_row=15 ≥ 8 → lower tile, row 15-8=7
        vram = make_vram()
        regs = make_regs(r5=0x7E, r1=0x02)
        write_tile_row(vram, 3, 7, b0=0xFF)   # lower tile row 7
        setup_sprite(vram, regs, 0, 0, 0, 2)
        sp_cram, sp_has, _, _ = render_sprite_line(vram, regs, 16)  # dy=15
        assert sp_has[0]

    # --- zoom mode ---------------------------------------------------------

    def test_zoom_doubles_each_pixel_in_x(self):
        # col 0 only → screen pixels x and x+1 both set; x+2 transparent
        vram = make_vram()
        regs = make_regs(r5=0x7E, r1=0x01)
        write_tile_row(vram, 0, 0, b0=0x80)   # only bit 7 (col 0) set
        setup_sprite(vram, regs, 0, 0, 10, 0)
        sp_cram, sp_has, _, _ = render_sprite_line(vram, regs, 1)
        assert sp_has[10]
        assert sp_has[11]
        assert not sp_has[12]

    def test_zoom_dy_1_uses_same_row_as_dy_0(self):
        # zoom: actual_row = dy>>1; dy=0 and dy=1 both map to row 0
        vram = make_vram()
        regs = make_regs(r5=0x7E, r1=0x01)
        write_tile_row(vram, 0, 0, b0=0xFF)   # row 0 has data
        write_tile_row(vram, 0, 1, b0=0x00)   # row 1 empty
        setup_sprite(vram, regs, 0, 0, 0, 0)
        _, has_dy0, _, _ = render_sprite_line(vram, regs, 1)  # dy=0
        _, has_dy1, _, _ = render_sprite_line(vram, regs, 2)  # dy=1
        assert has_dy0[0]
        assert has_dy1[0]   # same row 0, still has pixel

    def test_zoom_dy_2_uses_row_1(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E, r1=0x01)
        write_tile_row(vram, 0, 0, b0=0xFF)   # row 0 has data
        write_tile_row(vram, 0, 1, b0=0x00)   # row 1 empty
        setup_sprite(vram, regs, 0, 0, 0, 0)
        _, has_dy2, _, _ = render_sprite_line(vram, regs, 3)  # dy=2 → row 1
        assert not has_dy2[0]

    # --- right-edge clipping -----------------------------------------------

    def test_right_edge_clipping_x252(self):
        # X=252 → cols 0-3 map to screen 252-255 (in range);
        # cols 4-7 map to 260-263 (out of range) → clipped
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        write_tile_row(vram, 0, 0, b0=0xFF)
        setup_sprite(vram, regs, 0, 0, 252, 0)
        sp_cram, sp_has, _, _ = render_sprite_line(vram, regs, 1)
        assert sp_has[252:256].all()
        assert int(sp_has.sum()) == 4

    # --- sprite priority ---------------------------------------------------

    def test_earlier_sprite_wins_at_shared_position(self):
        # Sprite 0 (cram 17) and sprite 1 (cram 18) both at X=0;
        # sprite 0 is drawn first and should occupy position 0
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        write_tile_row(vram, 0, 0, b0=0xFF)          # tile 0 → color 1 → cram 17
        write_tile_row(vram, 1, 0, b1=0xFF)          # tile 1 → color 2 → cram 18
        sat_write(vram, regs, 0, 0, 0, 0)
        sat_write(vram, regs, 1, 0, 0, 1)
        sat_terminate(vram, regs, 2)
        sp_cram, sp_has, _, collision = render_sprite_line(vram, regs, 1)
        assert sp_cram[0] == 17
        assert collision is True

    # --- collision detection -----------------------------------------------

    def test_collision_two_overlapping_sprites(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        write_tile_row(vram, 0, 0, b0=0xFF)
        sat_write(vram, regs, 0, 0, 0, 0)   # X=0, covers 0-7
        sat_write(vram, regs, 1, 0, 4, 0)   # X=4, covers 4-11 → overlap 4-7
        sat_terminate(vram, regs, 2)
        _, _, _, collision = render_sprite_line(vram, regs, 1)
        assert collision is True

    def test_no_collision_non_overlapping_sprites(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        write_tile_row(vram, 0, 0, b0=0xFF)
        sat_write(vram, regs, 0, 0,  0, 0)  # X=0,  pixels 0-7
        sat_write(vram, regs, 1, 0, 16, 0)  # X=16, pixels 16-23
        sat_terminate(vram, regs, 2)
        _, _, _, collision = render_sprite_line(vram, regs, 1)
        assert collision is False

    def test_no_collision_when_overlap_is_transparent(self):
        # Two sprites at same X, but all pixels are color 0 → transparent
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        # tile 0 all zeros = transparent
        sat_write(vram, regs, 0, 0, 0, 0)
        sat_write(vram, regs, 1, 0, 0, 0)
        sat_terminate(vram, regs, 2)
        _, _, _, collision = render_sprite_line(vram, regs, 1)
        assert collision is False

    # --- overflow forwarded ------------------------------------------------

    def test_overflow_forwarded_from_sprites_on_line(self):
        vram = make_vram()
        regs = make_regs(r5=0x7E)
        for n in range(9):
            sat_write(vram, regs, n, 0, n * 8, 0)
        sat_terminate(vram, regs, 9)
        _, _, overflow, _ = render_sprite_line(vram, regs, 1)
        assert overflow is True

    # --- EC bit (R0 bit 3): 8-pixel left shift ----------------------------

    def test_ec_bit_clear_sprite_at_x0_occupies_cols_0_to_7(self):
        vram, regs = make_solid_sprite(x=0)
        # R0 bit 3 = 0 (EC off) — sprite at X=0 fills columns 0–7
        sp_cram, sp_has, _, _ = render_sprite_line(vram, regs, 1)
        assert sp_has[0:8].all()
        assert not sp_has[8]

    def test_ec_bit_set_shifts_sprite_left_8(self):
        vram, regs = make_solid_sprite(x=0)
        regs[0] |= 0x08   # set EC bit
        sp_cram, sp_has, _, _ = render_sprite_line(vram, regs, 1)
        # All 8 columns shifted off the left edge — nothing visible
        assert not sp_has.any()

    def test_ec_bit_set_sprite_at_x8_occupies_cols_0_to_7(self):
        vram, regs = make_solid_sprite(x=8)
        regs[0] |= 0x08
        sp_cram, sp_has, _, _ = render_sprite_line(vram, regs, 1)
        assert sp_has[0:8].all()
        assert not sp_has[8]

    def test_ec_bit_set_partial_clip_at_left_edge(self):
        # Sprite at X=4 with EC: effective X = -4, so cols 0–3 visible, 4–7 clipped
        vram, regs = make_solid_sprite(x=4)
        regs[0] |= 0x08
        sp_cram, sp_has, _, _ = render_sprite_line(vram, regs, 1)
        assert sp_has[0:4].all()
        assert not sp_has[4:8].any()

    def test_ec_bit_does_not_affect_collision_detection(self):
        # Two sprites that overlap after the EC shift should still flag collision
        vram, regs = make_solid_sprite(x=8)   # effective X=0 with EC
        regs[0] |= 0x08
        # Add a second sprite also landing at effective X=0
        sat_write(vram, regs, 1, 0, 8, 0)    # same tile, same effective position
        sat_terminate(vram, regs, 2)
        _, _, _, collision = render_sprite_line(vram, regs, 1)
        assert collision is True


# ---------------------------------------------------------------------------
# TestComposition
# ---------------------------------------------------------------------------

# Helpers for _compose_line (operates on any equal-length lists)
def bg_px(cram_idx, priority=False):
    return (cram_idx, priority)

def sp_px(cram_idx=0, has_pixel=False):
    return (cram_idx, has_pixel)

def to_sv(bg_list):
    """Convert list of (cram_idx, priority) tuples to ScanlineView."""
    return ScanlineView(
        np.array([t[0] for t in bg_list], dtype=np.uint8),
        np.array([t[1] for t in bg_list], dtype=np.bool_),
    )

def to_sp(sp_list):
    """Convert list of (cram_idx, has_pixel) tuples to (sp_cram, sp_has) arrays."""
    return (
        np.array([t[0] for t in sp_list], dtype=np.uint8),
        np.array([t[1] for t in sp_list], dtype=np.bool_),
    )

# VDP helpers for status-bit tests
_COMP_SAT = 0x3F00   # R5=0x7E → (0x7E & 0x7E) << 7 = 0x3F00

def make_comp_vdp():
    vdp = VDP()
    vdp.regs[1] = 0x40   # display enable
    vdp.regs[5] = 0x7E
    return vdp

def vdp_add_sprite(vdp, n, y, x, tile_num):
    vdp.vram[_COMP_SAT + n]               = y
    vdp.vram[_COMP_SAT + 128 + n * 2]     = x
    vdp.vram[_COMP_SAT + 128 + n * 2 + 1] = tile_num

def vdp_terminate(vdp, n):
    vdp.vram[_COMP_SAT + n] = 0xD0

def step_lines(vdp, count):
    for _ in range(count):
        vdp.step(CYCLES_PER_LINE)


class TestComposition:

    # --- _compose_line pixel rules -----------------------------------------

    def test_sprite_wins_over_non_priority_bg(self):
        vdp = VDP()
        bg = to_sv([bg_px(1, False)] * 256)
        sp_cram, sp_has = to_sp([sp_px(17, True)] * 256)
        result = vdp._compose_line(bg, sp_cram, sp_has)
        assert result[0] == 17

    def test_priority_bg_opaque_hides_sprite(self):
        vdp = VDP()
        # bg_cram=1, priority=True, 1%16=1 (opaque) → bg wins
        bg = to_sv([bg_px(1, True)] * 256)
        sp_cram, sp_has = to_sp([sp_px(17, True)] * 256)
        result = vdp._compose_line(bg, sp_cram, sp_has)
        assert result[0] == 1

    def test_priority_bg_palette0_color0_shows_sprite(self):
        # bg_cram=0 → 0%16=0 (transparent) → sprite wins despite priority
        vdp = VDP()
        bg = to_sv([bg_px(0, True)] * 256)
        sp_cram, sp_has = to_sp([sp_px(17, True)] * 256)
        result = vdp._compose_line(bg, sp_cram, sp_has)
        assert result[0] == 17

    def test_priority_bg_palette1_color0_shows_sprite(self):
        # bg_cram=16 → 16%16=0 (palette 1 color 0) → still transparent → sprite wins
        vdp = VDP()
        bg = to_sv([bg_px(16, True)] * 256)
        sp_cram, sp_has = to_sp([sp_px(17, True)] * 256)
        result = vdp._compose_line(bg, sp_cram, sp_has)
        assert result[0] == 17

    def test_no_sprite_pixel_shows_bg(self):
        vdp = VDP()
        bg = to_sv([bg_px(5, False)] * 256)
        sp_cram, sp_has = to_sp([sp_px(0, False)] * 256)
        result = vdp._compose_line(bg, sp_cram, sp_has)
        assert result[0] == 5

    def test_no_sprite_pixel_shows_bg_even_with_priority(self):
        vdp = VDP()
        bg = to_sv([bg_px(3, True)] * 256)
        sp_cram, sp_has = to_sp([sp_px(17, False)] * 256)
        result = vdp._compose_line(bg, sp_cram, sp_has)
        assert result[0] == 3

    def test_output_length_is_256(self):
        vdp = VDP()
        bg = to_sv([bg_px(1, False)] * 256)
        sp_cram, sp_has = to_sp([sp_px(17, True)] * 256)
        assert len(vdp._compose_line(bg, sp_cram, sp_has)) == 256

    def test_mixed_all_four_cases(self):
        # Per-pixel: sprite/no-prio-bg, sprite/prio-bg/transparent, bg/prio-bg/opaque, bg/no-sprite
        vdp = VDP()
        bg = to_sv([bg_px(1, False), bg_px(0, True), bg_px(2, True), bg_px(4, False)] + [bg_px(0, False)] * 252)
        sp_cram, sp_has = to_sp([sp_px(17, True), sp_px(18, True), sp_px(19, True), sp_px(0, False)] + [sp_px(0, False)] * 252)
        result = vdp._compose_line(bg, sp_cram, sp_has)
        assert result[0] == 17   # sprite over non-priority bg
        assert result[1] == 18   # sprite wins: priority bg but transparent (0%16=0)
        assert result[2] == 2    # bg wins: priority + opaque (2%16=2)
        assert result[3] == 4    # bg wins: no sprite pixel

    def test_bg_cram_values_preserved_in_output(self):
        vdp = VDP()
        bg = to_sv([bg_px(n, False) for n in range(256)])
        sp_cram, sp_has = to_sp([sp_px(0, False)] * 256)
        result = vdp._compose_line(bg, sp_cram, sp_has)
        assert list(result) == list(range(256))

    def test_sprite_cram_values_preserved_in_output(self):
        vdp = VDP()
        bg = to_sv([bg_px(0, False)] * 256)
        sp_cram, sp_has = to_sp([sp_px(n % 16 + 16, True) for n in range(256)])
        result = vdp._compose_line(bg, sp_cram, sp_has)
        assert list(result) == [n % 16 + 16 for n in range(256)]

    # --- VDP status bits via step() ----------------------------------------

    def test_overflow_bit_set_in_status(self):
        # 9 sprites on line 11 → overflow → status bit 6 set
        vdp = make_comp_vdp()
        for n in range(9):
            vdp_add_sprite(vdp, n, 10, n * 8, 0)
        vdp_terminate(vdp, 9)
        step_lines(vdp, 12)           # renders lines 0-11
        assert vdp.status & 0x40

    def test_overflow_bit_not_set_with_8_sprites(self):
        vdp = make_comp_vdp()
        for n in range(8):
            vdp_add_sprite(vdp, n, 10, n * 8, 0)
        vdp_terminate(vdp, 8)
        step_lines(vdp, 12)
        assert not (vdp.status & 0x40)

    def test_collision_bit_set_in_status(self):
        # Two overlapping sprites with non-transparent pixels → status bit 5
        vdp = make_comp_vdp()
        vdp.vram[0] = 0xFF                     # tile 0 row 0 plane 0: all color 1
        vdp_add_sprite(vdp, 0, 10, 0, 0)       # X=0,  pixels 0-7
        vdp_add_sprite(vdp, 1, 10, 4, 0)       # X=4,  pixels 4-11 → overlap 4-7
        vdp_terminate(vdp, 2)
        step_lines(vdp, 12)
        assert vdp.status & 0x20

    def test_collision_bit_not_set_without_overlap(self):
        vdp = make_comp_vdp()
        vdp.vram[0] = 0xFF
        vdp_add_sprite(vdp, 0, 10,  0, 0)      # pixels 0-7
        vdp_add_sprite(vdp, 1, 10, 16, 0)      # pixels 16-23 — no overlap
        vdp_terminate(vdp, 2)
        step_lines(vdp, 12)
        assert not (vdp.status & 0x20)

    def test_both_status_bits_set_simultaneously(self):
        # 9 sprites (overflow) and two of them overlap (collision) on same line
        vdp = make_comp_vdp()
        vdp.vram[0] = 0xFF                     # tile 0: all color 1
        for n in range(9):
            vdp_add_sprite(vdp, n, 10, 0, 0)   # all at X=0 → collision on any two
        vdp_terminate(vdp, 9)
        step_lines(vdp, 12)
        assert vdp.status & 0x40               # overflow
        assert vdp.status & 0x20               # collision

    def test_status_bits_accumulate_across_lines(self):
        # Overflow on lines 11-18 (9 sprites at Y=10), then collision on line 20
        # (2 sprites at Y=19, appearing after the overflow sprites finish).
        # Placing collision sprites after the overflow window ensures they aren't
        # shadowed by the 8-sprite limit on lines 11-18.
        vdp = make_comp_vdp()
        vdp.vram[0] = 0xFF                     # tile 0 row 0 plane 0: all color 1
        for n in range(9):
            vdp_add_sprite(vdp, n, 10, n * 8, 0)   # Y=10, visible lines 11-18
        # Slots 9 and 10: colliding sprites at Y=19 → visible lines 20-27
        vdp.vram[_COMP_SAT + 9]                = 19
        vdp.vram[_COMP_SAT + 128 + 9 * 2]      = 0
        vdp.vram[_COMP_SAT + 128 + 9 * 2 + 1]  = 0
        vdp.vram[_COMP_SAT + 10]               = 19
        vdp.vram[_COMP_SAT + 128 + 10 * 2]     = 4
        vdp.vram[_COMP_SAT + 128 + 10 * 2 + 1] = 0
        vdp_terminate(vdp, 11)
        step_lines(vdp, 21)        # renders through line 20
        assert vdp.status & 0x40   # overflow from lines 11-18
        assert vdp.status & 0x20   # collision from line 20

