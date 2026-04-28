"""Tests for pygear.vdp.sprites and VDP sprite compositing."""
import pytest
from pygear.vdp.sprites import sat_base, parse_sat, sprites_on_line, render_sprite_line
from pygear.vdp.vdp import VDP, CYCLES_PER_LINE, ACTIVE_LINES

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
