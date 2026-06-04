"""Game Gear / SMS VDP sprite subsystem — SAT parsing, line collection, rendering.

Sprite Attribute Table (SAT)
-----------------------------
Base address: (R5 & 0x7E) << 7  — must be on a 128-byte boundary.

Layout
  Bytes   0 –  63  Y coordinates for sprites 0–63 (one byte each).
  Bytes 128 – 255  For each sprite N: X at 128+N*2, tile# at 128+N*2+1.

Y terminator
  If the Y byte equals 0xD0 (208), sprite processing stops.  No further
  sprites in the table are evaluated.

Y coordinate convention
  A sprite with Y value *y* is first visible on scanline y+1.
  The unsigned distance  dy = (line − (y+1)) & 0xFF  is the row offset
  within the sprite; the sprite is visible when dy < effective_height.

Heights and zoom
  R1 bit 1 (tall)  — 0 → 8-pixel-tall sprites; 1 → 16-pixel-tall sprites.
  R1 bit 0 (zoom)  — doubles both dimensions, so height becomes 16 or 32.
  In tall mode the tile number has bit 0 forced to 0 (upper tile); the
  lower half uses tile_num | 1.

Scanline limit
  At most 8 sprites per scanline are returned; if a 9th would match, the
  overflow flag is set and collection stops.

Sprite tile pattern base
  (R6 & 0x04) << 11  → 0x0000 or 0x2000.

Palette
  Sprites always use palette 1 (CRAM entries 16–31).
  Color index 0 is transparent.

Collision
  Set when two non-transparent sprite pixels share the same screen X.
  Detected across all sprites on the line regardless of draw order.
"""

import numpy as np

SPRITE_LIMIT = 8

# Pre-computed bit-shift positions for vectorised bitplane extraction
_SHIFTS = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=np.uint8)


def sat_base(regs: bytearray) -> int:
    """Return the VRAM byte address of the SAT from register R5."""
    return (regs[5] & 0x7E) << 7


def parse_sat(vram: bytearray, regs: bytearray) -> list:
    """Read the full SAT and return a list of (y, x, tile_num) tuples.

    Stops early when Y == 0xD0 (terminator).  In tall mode (R1 bit 1)
    bit 0 of tile_num is cleared so the pair always starts on an even tile.
    Returns at most 64 entries.
    """
    base = sat_base(regs)
    tall = bool(regs[1] & 0x02)
    result = []
    for n in range(64):
        y = vram[(base + n) & 0x3FFF]
        if y == 0xD0:
            break
        xaddr    = base + 128 + n * 2
        x        = vram[ xaddr      & 0x3FFF]
        tile_num = vram[(xaddr + 1) & 0x3FFF]
        if tall:
            tile_num &= 0xFE          # force bit 0 clear for tall sprites
        result.append((y, x, tile_num))
    return result


def sprites_on_line(vram: bytearray, regs: bytearray, line: int,
                    sat: list | None = None) -> tuple:
    """Return (visible, overflow) for *line*.

    *visible* is a list of up to SPRITE_LIMIT (8) tuples
    (x, tile_num, dy) where dy is the zero-based row offset within the
    sprite's effective height (already accounts for zoom).
    *overflow* is True when more than 8 sprites would have matched.

    Pass *sat* (a pre-parsed result from parse_sat()) to avoid re-parsing
    the SAT on every scanline; the caller is responsible for cache validity.

    Height calculation
      base_h = 16 if tall (R1 bit 1) else 8
      height = base_h * 2 if zoom (R1 bit 0) else base_h
    """
    tall   = bool(regs[1] & 0x02)
    zoom   = bool(regs[1] & 0x01)
    base_h = 16 if tall else 8
    height = base_h * (2 if zoom else 1)

    if sat is None:
        sat = parse_sat(vram, regs)

    visible  = []
    overflow = False

    for y, x, tile_num in sat:
        dy = (line - (y + 1)) & 0xFF
        if dy >= height:
            continue
        if len(visible) == SPRITE_LIMIT:
            overflow = True
            break
        visible.append((x, tile_num, dy))

    return visible, overflow


def render_sprite_line(vram: bytearray, regs: bytearray, line: int,
                       sat: list | None = None) -> tuple:
    """Render all sprites visible on *line*.

    Returns (sp_cram, sp_has, overflow, collision).

    sp_cram
      numpy uint8 array of length 256.  Each element is the CRAM index
      (16–31, palette 1) for that screen X position, or 0 if no sprite.

    sp_has
      numpy bool_ array of length 256.  True where a non-transparent
      sprite pixel exists.

    overflow
      True when more than 8 sprites were visible on this line.

    collision
      True when two non-transparent sprite pixels share the same screen X.

    Tile row selection (tall + zoom)
      actual_row = dy >> 1  if zoom else dy          (range 0–7 or 0–15)
      In tall mode: actual_row < 8 → upper tile (tile_num & ~1, tile_row = actual_row)
                    actual_row ≥ 8 → lower tile (tile_num | 1,  tile_row = actual_row − 8)

    Zoom X
      Each of the 8 tile pixel columns occupies 2 consecutive screen columns.
    """
    tall      = bool(regs[1] & 0x02)
    zoom      = bool(regs[1] & 0x01)
    tile_base = (regs[6] & 0x04) << 11
    ec_shift  = -8 if (regs[0] & 0x08) else 0

    visible, overflow = sprites_on_line(vram, regs, line, sat=sat)

    sp_cram   = np.zeros(256, dtype=np.uint8)
    sp_has    = np.zeros(256, dtype=np.bool_)
    collision = False

    for x, tile_num, dy in visible:
        # Which row within the tile(s) are we on?
        actual_row = dy >> 1 if zoom else dy
        if tall and actual_row >= 8:
            tile_num = tile_num | 1       # lower tile of the pair
            tile_row = actual_row - 8
        else:
            tile_row = actual_row

        # Read the four bitplane bytes for this tile row
        addr = (tile_base + tile_num * 32 + tile_row * 4) & 0x3FFF
        b0 = vram[addr]
        b1 = vram[(addr + 1) & 0x3FFF]
        b2 = vram[(addr + 2) & 0x3FFF]
        b3 = vram[(addr + 3) & 0x3FFF]

        # Extract all 8 pixel colour indices at once using numpy
        raw = np.array([b0, b1, b2, b3], dtype=np.uint8)   # (4,)
        color_idx = (
            ((raw[0] >> _SHIFTS) & 1)
            | (((raw[1] >> _SHIFTS) & 1) << 1)
            | (((raw[2] >> _SHIFTS) & 1) << 2)
            | (((raw[3] >> _SHIFTS) & 1) << 3)
        ).astype(np.uint8)                                  # (8,)

        if not zoom:
            sx = x + ec_shift + np.arange(8, dtype=np.int32)   # (8,)
            plot = (color_idx != 0) & (sx >= 0) & (sx <= 255)
            sx_p = sx[plot]
            already = sp_has[sx_p]
            if already.any():
                collision = True
            # Only draw pixels not yet occupied (first sprite wins)
            free = ~already
            free_px = sx_p[free]
            sp_cram[free_px] = color_idx[plot][free] + 16
            sp_has[free_px]  = True
        else:
            # Zoom: each of the 8 source pixels covers 2 screen columns
            for col in range(8):
                if color_idx[col] == 0:
                    continue
                cram_idx = int(color_idx[col]) + 16
                sx0 = x + ec_shift + col * 2
                for sx in range(sx0, sx0 + 2):
                    if sx < 0 or sx > 255:
                        continue
                    if sp_has[sx]:
                        collision = True
                    else:
                        sp_cram[sx] = cram_idx
                        sp_has[sx]  = True

    return sp_cram, sp_has, overflow, collision
