"""VDP tile decoder — 4bpp planar SMS/GG tile format.

Tile layout in VRAM
-------------------
Each tile is 8×8 pixels stored as 32 consecutive bytes (4 bytes × 8 rows).
One byte per bit-plane per row:

  offset  content
  4r+0    plane 0  (bit 0 of each pixel's colour index)
  4r+1    plane 1  (bit 1)
  4r+2    plane 2  (bit 2)
  4r+3    plane 3  (bit 3)

Within each plane byte bit 7 is the leftmost pixel (column 0) and bit 0 is
the rightmost pixel (column 7).

Colour index for pixel (row r, column c):
  shift = 7 - c
  index = ((plane0 >> shift) & 1)
        | ((plane1 >> shift) & 1) << 1
        | ((plane2 >> shift) & 1) << 2
        | ((plane3 >> shift) & 1) << 3
"""

TILE_BYTES = 32   # 8 rows × 4 bytes/row


def decode_tile(
    vram: bytearray,
    tile_num: int,
    hflip: bool = False,
    vflip: bool = False,
) -> list:
    """Decode one 8×8 tile from *vram*.

    Returns a list of 8 rows, each a list of 8 four-bit colour indices (0–15).
    *hflip* reverses pixel order within every row; *vflip* reverses row order.
    Tile 0 starts at VRAM byte 0; tile N starts at byte N*32.
    """
    base = tile_num * TILE_BYTES
    rows = []
    for row in range(8):
        off = base + row * 4
        b0, b1, b2, b3 = vram[off], vram[off + 1], vram[off + 2], vram[off + 3]
        pixels = []
        for col in range(8):
            shift = 7 - col
            c = (
                 ((b0 >> shift) & 1)
                | ((b1 >> shift) & 1) << 1
                | ((b2 >> shift) & 1) << 2
                | ((b3 >> shift) & 1) << 3
            )
            pixels.append(c)
        if hflip:
            pixels = pixels[::-1]
        rows.append(pixels)
    if vflip:
        rows = rows[::-1]
    return rows
