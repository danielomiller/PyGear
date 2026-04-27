"""Game Gear / SMS VDP sprite subsystem — SAT parsing and per-line collection.

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
"""

SPRITE_LIMIT = 8


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


def sprites_on_line(vram: bytearray, regs: bytearray, line: int) -> tuple:
    """Return (visible, overflow) for *line*.

    *visible* is a list of up to SPRITE_LIMIT (8) tuples
    (x, tile_num, dy) where dy is the zero-based row offset within the
    sprite's effective height (already accounts for zoom).
    *overflow* is True when more than 8 sprites would have matched.

    Height calculation
      base_h = 16 if tall (R1 bit 1) else 8
      height = base_h * 2 if zoom (R1 bit 0) else base_h
    """
    tall   = bool(regs[1] & 0x02)
    zoom   = bool(regs[1] & 0x01)
    base_h = 16 if tall else 8
    height = base_h * (2 if zoom else 1)

    visible  = []
    overflow = False

    for y, x, tile_num in parse_sat(vram, regs):
        dy = (line - (y + 1)) & 0xFF
        if dy >= height:
            continue
        if len(visible) == SPRITE_LIMIT:
            overflow = True
            break
        visible.append((x, tile_num, dy))

    return visible, overflow
