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


def render_sprite_line(vram: bytearray, regs: bytearray, line: int) -> tuple:
    """Render all sprites visible on *line*.

    Returns (pixels, overflow, collision).

    pixels
      List of 256 (cram_index, has_pixel) tuples.  has_pixel is False for
      transparent positions (color index 0 or no sprite).  cram_index is
      in the range 16–31 (palette 1) when has_pixel is True, 0 otherwise.

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

    visible, overflow = sprites_on_line(vram, regs, line)

    pixels    = [(0, False)] * 256
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

        # Decode and plot each of the 8 tile columns
        for col in range(8):
            shift = 7 - col
            color_idx = (
                 ((b0 >> shift) & 1)
                | ((b1 >> shift) & 1) << 1
                | ((b2 >> shift) & 1) << 2
                | ((b3 >> shift) & 1) << 3
            )
            if color_idx == 0:
                continue          # transparent

            cram_idx = color_idx + 16     # always palette 1

            # Each column covers 1 screen pixel (or 2 with zoom)
            sx0 = x + col * (2 if zoom else 1)
            for sx in range(sx0, sx0 + (2 if zoom else 1)):
                if sx < 0 or sx > 255:
                    continue
                if pixels[sx][1]:         # another sprite already drew here
                    collision = True
                else:
                    pixels[sx] = (cram_idx, True)

    return pixels, overflow, collision
