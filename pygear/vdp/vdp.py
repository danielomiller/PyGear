"""Sega Game Gear VDP — core: VRAM, CRAM, register file, port I/O, and compositing.

Port map
--------
0x7E  V-counter  (read)
0x7F  H-counter  (read)
0xBE  Data port  (read / write)
0xBF  Control port: write = command word; read = status (clears on read)

Status register bits
--------------------
bit 7  Frame interrupt pending (VBlank)
bit 6  Sprite overflow
bit 5  Sprite collision

Command word (two bytes written to 0xBF)
-----------------------------------------
First byte  A7–A0
Second byte CD1 CD0 A13 A12 A11 A10 A9 A8
  CD = 00  VRAM read  (prefetches one byte, increments address)
  CD = 01  VRAM write
  CD = 10  Register write  (bits 3–0 = register number, first byte = value)
  CD = 11  CRAM write

V-counter (NTSC)
----------------
Lines 0–218 (0x00–0xDA) count linearly.
Line 219 jumps to 0xD5 and continues to 0xFF at line 261.

Timing
------
228 T-states per scanline, 262 scanlines per frame (NTSC).
Active display: lines 0–191.  VBlank: lines 192–261.
"""

import numpy as np

from .sprites import render_sprite_line

VRAM_SIZE = 0x4000   # 16 KB
CRAM_SIZE = 64       # 32 colours × 2 bytes
NUM_REGS  = 11       # R0–R10


class ScanlineView:
    """Read-only view of a rendered background scanline.

    Wraps two numpy arrays (cram_indices and priority flags) and exposes a
    sequence interface compatible with the test suite:

      line[i]        → (int(cram_idx), bool(priority))
      line[a:b]      → list of (int, bool) tuples
      len(line)      → 256
      for px in line → yields (int, bool) tuples

    Internally used by _compose_line() which extracts the raw arrays.
    """
    __slots__ = ("_cram", "_pri")

    def __init__(self, cram: np.ndarray, pri: np.ndarray):
        self._cram = cram   # uint8[256]
        self._pri  = pri    # bool_[256]

    def __len__(self):
        return 256

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(256))
            return [(int(self._cram[i]), bool(self._pri[i])) for i in indices]
        return (int(self._cram[idx]), bool(self._pri[idx]))

    def __iter__(self):
        for i in range(256):
            yield (int(self._cram[i]), bool(self._pri[i]))

# NTSC timing
CYCLES_PER_LINE = 228
TOTAL_LINES     = 262
ACTIVE_LINES    = 192

# Game Gear display window cropped from 256×192 internal render
SCREEN_W = 160
SCREEN_H = 144
CROP_X   = 48    # (256 - 160) // 2
CROP_Y   = 24    # (192 - 144) // 2

# NTSC V-counter non-linear mapping
_VCTR_LINEAR_END = 0xDA   # last line with identity mapping
_VCTR_JUMP_TO    = 0xD5   # V-counter value at line (_VCTR_LINEAR_END + 1)


class VDP:
    def __init__(self):
        self.vram   = bytearray(VRAM_SIZE)
        self.cram   = bytearray(CRAM_SIZE)
        self.regs   = bytearray(NUM_REGS)
        self.status = 0

        # Address / latch state
        self._addr     = 0      # 14-bit address register
        self._code     = 0      # 2-bit code (00–11)
        self._latch    = False  # False = expecting 1st byte, True = expecting 2nd
        self._latch_lo = 0      # buffered first command byte
        self._read_buf = 0      # VRAM read-ahead buffer

        # Timing state
        self._line     = 0      # current scanline 0–261
        self._cycle    = 0      # T-states into current scanline
        self._line_irq = 0      # line interrupt down-counter

        # Frame output
        self._line_buffer = [None] * ACTIVE_LINES   # rendered scanlines 0–191
        self.frame        = None    # assembled 144×160 list of (R,G,B) tuples
        self.frame_ready  = False   # True after each VBlank; caller clears it

        self._cpu = None        # set by attach_cpu()

    # ------------------------------------------------------------------
    def attach_cpu(self, cpu) -> None:
        """Bind the CPU so the VDP can raise interrupts."""
        self._cpu = cpu

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.vram[:] = bytes(VRAM_SIZE)
        self.cram[:] = bytes(CRAM_SIZE)
        self.regs[:] = bytes(NUM_REGS)
        self.status       = 0
        self._addr        = 0
        self._code        = 0
        self._latch       = False
        self._latch_lo    = 0
        self._read_buf    = 0
        self._line        = 0
        self._cycle       = 0
        self._line_irq    = 0
        self._line_buffer = [None] * ACTIVE_LINES
        self.frame        = None
        self.frame_ready  = False

    def get_state(self) -> dict:
        return {
            'vram':         bytes(self.vram),
            'cram':         bytes(self.cram),
            'regs':         bytes(self.regs),
            'status':       self.status,
            '_addr':        self._addr,
            '_code':        self._code,
            '_latch':       self._latch,
            '_latch_lo':    self._latch_lo,
            '_read_buf':    self._read_buf,
            '_line':        self._line,
            '_cycle':       self._cycle,
            '_line_irq':    self._line_irq,
            '_line_buffer': [b.tobytes() if b is not None else None
                             for b in self._line_buffer],
            'frame_ready':  self.frame_ready,
        }

    def set_state(self, s: dict) -> None:
        self.vram[:]  = s['vram']
        self.cram[:]  = s['cram']
        self.regs[:]  = s['regs']
        self.status   = s['status']
        self._addr    = s['_addr']
        self._code    = s['_code']
        self._latch   = s['_latch']
        self._latch_lo = s['_latch_lo']
        self._read_buf = s['_read_buf']
        self._line    = s['_line']
        self._cycle   = s['_cycle']
        self._line_irq = s['_line_irq']
        self._line_buffer = [
            np.frombuffer(b, dtype=np.uint8).copy() if b is not None else None
            for b in s['_line_buffer']
        ]
        self.frame_ready = s['frame_ready']
        # Rebuild frame from last complete line buffer if available
        if any(b is not None for b in self._line_buffer):
            self._assemble_frame()

    # ------------------------------------------------------------------
    # Port interface
    # ------------------------------------------------------------------
    def port_read(self, port: int) -> int:
        port &= 0xFF
        if port == 0x7E:
            return self._vcounter()
        if port == 0x7F:
            return self._hcounter()
        if port == 0xBE:
            return self._data_read()
        if port == 0xBF:
            return self._control_read()
        return 0xFF

    def port_write(self, port: int, value: int) -> None:
        port  &= 0xFF
        value &= 0xFF
        if port == 0xBE:
            self._data_write(value)
        elif port == 0xBF:
            self._control_write(value)

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------
    def _vcounter(self) -> int:
        if self._line <= _VCTR_LINEAR_END:
            return self._line
        return (_VCTR_JUMP_TO + (self._line - _VCTR_LINEAR_END - 1)) & 0xFF

    def _hcounter(self) -> int:
        # Increments every 2 T-states across the scanline
        return (self._cycle >> 1) & 0xFF

    # ------------------------------------------------------------------
    # Control port
    # ------------------------------------------------------------------
    def _control_read(self) -> int:
        s = self.status
        self.status = 0      # cleared on read
        self._latch = False  # reset address latch
        return s

    def _control_write(self, value: int) -> None:
        if not self._latch:
            self._latch_lo = value
            self._latch    = True
            return

        # Second byte — decode command word
        self._latch = False
        self._code  = (value >> 6) & 0x03
        self._addr  = ((value & 0x3F) << 8) | self._latch_lo

        if self._code == 0b10:              # Register write
            reg = value & 0x0F
            if reg < NUM_REGS:
                self.regs[reg] = self._latch_lo

        elif self._code == 0b00:            # VRAM read — prefetch first byte
            self._read_buf = self.vram[self._addr & 0x3FFF]
            self._addr     = (self._addr + 1) & 0x3FFF

    # ------------------------------------------------------------------
    # Data port
    # ------------------------------------------------------------------
    def _data_read(self) -> int:
        self._latch = False          # any data access resets the latch
        v = self._read_buf
        self._read_buf = self.vram[self._addr & 0x3FFF]
        self._addr     = (self._addr + 1) & 0x3FFF
        return v

    def _data_write(self, value: int) -> None:
        self._latch = False          # any data access resets the latch
        if self._code == 0b11:       # CRAM write
            self.cram[self._addr & 0x3F] = value
        else:                        # VRAM write (codes 00, 01)
            self.vram[self._addr & 0x3FFF] = value
        self._addr = (self._addr + 1) & 0x3FFF

    # ------------------------------------------------------------------
    # Background scanline renderer
    # ------------------------------------------------------------------
    def render_line(self, line: int) -> "ScanlineView":
        """Render 256 pixels of the background layer for *line*.

        Returns a ScanlineView wrapping two numpy arrays:
          cram_indices: uint8[256]  — CRAM index 0–31 per pixel
          priority:     bool_[256] — True when the name-table entry has bit 12 set

        The ScanlineView supports sequence indexing so callers can still do
        ``line[i]`` → ``(cram_index, priority)`` tuples as before.

        Name table
        ----------
        Base address = (R2 & 0x0E) << 10.  32 columns × 28 rows, 2 bytes per
        entry (little-endian):
          bits  8–0  tile number (0–511)
          bit   9    horizontal flip
          bit  10    vertical flip
          bit  11    palette (0 → CRAM 0–15, 1 → CRAM 16–31)
          bit  12    priority

        Scroll
        ------
        H-scroll (R8): bg_x = (screen_x + R8) & 0xFF
        V-scroll (R9): bg_y = (line    + R9) % 224  (wraps at 28 tile rows)

        Scroll locks (R0)
        -----------------
        bit 6  H-scroll lock: top 16 lines (line < 16) ignore R8
        bit 7  V-scroll lock: right 8 columns (screen_x >= 248) ignore R9
        """
        r0        = self.regs[0]
        h_scroll  = int(self.regs[8])
        v_scroll  = int(self.regs[9])
        name_base = (self.regs[2] & 0x0E) << 10

        hscroll_lock = bool(r0 & 0x40)
        vscroll_lock = bool(r0 & 0x80)

        eff_h = 0 if (hscroll_lock and line < 16) else h_scroll

        # Scrolled vertical position (wraps at 224 = 28 × 8)
        bg_y      = (line + v_scroll) % 224
        tile_row  = bg_y >> 3
        pixel_row = bg_y & 7

        # Unscrolled vertical position (used by V-scroll-locked columns)
        ly_lock    = line % 224
        tile_row_l = ly_lock >> 3
        prow_l     = ly_lock & 7

        # Zero-copy view of VRAM as a read-only numpy array
        vram_np = np.frombuffer(self.vram, dtype=np.uint8)

        # Per-pixel screen x coordinates
        screen_x = np.arange(256, dtype=np.int32)

        # Apply V-scroll lock: right 8 columns (screen_x >= 248) ignore V-scroll
        if vscroll_lock:
            tr = np.where(screen_x >= 248, tile_row_l, tile_row).astype(np.int32)
            pr = np.where(screen_x >= 248, prow_l,     pixel_row).astype(np.int32)
        else:
            tr = np.full(256, tile_row,  dtype=np.int32)
            pr = np.full(256, pixel_row, dtype=np.int32)

        # Horizontal scroll
        bg_x      = (screen_x + eff_h) & 0xFF
        tile_col  = (bg_x >> 3).astype(np.int32)
        pixel_col = (bg_x & 7).astype(np.int32)

        # Name-table entry addresses
        nt_off = (name_base + (tr * 32 + tile_col) * 2).astype(np.int32)
        lo     = vram_np[ nt_off      & 0x3FFF].astype(np.uint16)
        hi     = vram_np[(nt_off + 1) & 0x3FFF].astype(np.uint16)
        entry  = lo | (hi << 8)

        # Decode entry fields
        tile_num = (entry & 0x1FF).astype(np.int32)
        hflip    = ((entry >> 9)  & 1).astype(np.bool_)
        vflip    = ((entry >> 10) & 1).astype(np.bool_)
        palette  = ((entry >> 11) & 1).astype(np.int32)
        priority = ((entry >> 12) & 1).astype(np.bool_)

        # Apply flips to row/column within tile
        row = np.where(vflip, 7 - pr,        pr).astype(np.int32)
        col = np.where(hflip, 7 - pixel_col, pixel_col).astype(np.int32)

        shift = (7 - col).astype(np.int32)

        # Base address of the 4-byte bitplane row in VRAM
        base = (tile_num * 32 + row * 4).astype(np.int32)

        # Read the four bitplanes
        p0 = vram_np[ base      & 0x3FFF].astype(np.uint8)
        p1 = vram_np[(base + 1) & 0x3FFF].astype(np.uint8)
        p2 = vram_np[(base + 2) & 0x3FFF].astype(np.uint8)
        p3 = vram_np[(base + 3) & 0x3FFF].astype(np.uint8)

        # Extract colour bit from each plane and combine into 4-bit index
        color_idx = (
            ((p0 >> shift) & 1)
            | (((p1 >> shift) & 1) << 1)
            | (((p2 >> shift) & 1) << 2)
            | (((p3 >> shift) & 1) << 3)
        ).astype(np.uint8)

        # Apply palette offset (0 or 16)
        cram_idx = (color_idx + (palette * 16).astype(np.uint8)).astype(np.uint8)

        return ScanlineView(cram_idx, priority)

    # ------------------------------------------------------------------
    # Sprite + background compositing
    # ------------------------------------------------------------------
    def _compose_line(self, bg: "ScanlineView",
                      sp_cram: np.ndarray, sp_has: np.ndarray) -> np.ndarray:
        """Merge background and sprite layers into a 256-element cram_index array.

        Compositing rule per pixel:
          Sprite wins when it has a non-transparent pixel AND the background
          tile either has no priority flag OR its own color index % 16 == 0
          (i.e. the background pixel is itself transparent / palette color 0).
          Otherwise the background pixel is used.

        Parameters
        ----------
        bg      : ScanlineView returned by render_line()
        sp_cram : uint8[256] sprite CRAM indices from render_sprite_line()
        sp_has  : bool_[256] sprite pixel presence mask
        """
        bg_cram = bg._cram   # uint8[256]
        bg_pri  = bg._pri    # bool_[256]
        use_sprite = sp_has & (~bg_pri | (bg_cram % 16 == 0))
        return np.where(use_sprite, sp_cram, bg_cram).astype(np.uint8)

    # ------------------------------------------------------------------
    # CRAM colour decode
    # ------------------------------------------------------------------
    def cram_color(self, index: int) -> tuple:
        """Decode CRAM entry *index* (0–31) to an (R, G, B) tuple (0–255 each).

        Game Gear CRAM format — 2 bytes per entry, little-endian:
          bits  3–0   Red   (4-bit)
          bits  7–4   Green (4-bit)
          bits 11–8   Blue  (4-bit)
          bits 15–12  unused

        Each 4-bit channel is scaled to 8-bit by multiplying by 17
        (0x0→0, 0xF→255, linear).
        """
        lo   = self.cram[index * 2]
        hi   = self.cram[index * 2 + 1]
        word = lo | (hi << 8)
        r = ((word >> 0) & 0xF) * 17
        g = ((word >> 4) & 0xF) * 17
        b = ((word >> 8) & 0xF) * 17
        return (r, g, b)

    # ------------------------------------------------------------------
    # Frame stepper
    # ------------------------------------------------------------------
    def step(self, cycles: int) -> None:
        """Advance the VDP by *cycles* T-states.

        For every completed scanline (228 T-states) _end_of_line() is called:
        active lines are rendered and interrupts are evaluated; VBlank fires
        when the line counter transitions to ACTIVE_LINES (192).
        """
        self._cycle += cycles
        while self._cycle >= CYCLES_PER_LINE:
            self._cycle -= CYCLES_PER_LINE
            self._end_of_line()

    def _end_of_line(self) -> None:
        line = self._line

        if line < ACTIVE_LINES:
            if self.regs[1] & 0x40:  # display enable (R1 bit 6)
                bg                             = self.render_line(line)
                sp_cram, sp_has, ov, collision = render_sprite_line(self.vram, self.regs, line)
                self._line_buffer[line]        = self._compose_line(bg, sp_cram, sp_has)
                if ov:        self.status |= 0x40   # sprite overflow
                if collision: self.status |= 0x20   # sprite collision
            else:
                # Display blanked: all pixels show backdrop color (sprite palette, R7[3:0])
                backdrop = 16 + (self.regs[7] & 0x0F)
                self._line_buffer[line] = np.full(256, backdrop, dtype=np.uint8)
            # Line interrupt counter (R0 bit 4 enables, R10 is reload value).
            # Decrement first, fire when it goes negative, then reload.
            self._line_irq -= 1
            if self._line_irq < 0:
                self._line_irq = self.regs[10]
                if (self.regs[0] & 0x10) and self._cpu:
                    self._cpu.request_interrupt()
        else:
            # VBlank lines: reload counter on last VBlank line only so it is
            # fresh (= regs[10]) when active display begins the next frame.
            if line == TOTAL_LINES - 1:
                self._line_irq = self.regs[10]

        self._line = (line + 1) % TOTAL_LINES

        # VBlank event: fires as _line transitions to ACTIVE_LINES
        if self._line == ACTIVE_LINES:
            self.status |= 0x80                      # frame interrupt flag
            if (self.regs[1] & 0x20) and self._cpu:  # R1 bit 5 enables VBlank IRQ
                self._cpu.request_interrupt()
            self._assemble_frame()

    def _assemble_frame(self) -> None:
        """Crop the 256×192 line buffer to the 160×144 Game Gear display."""
        # Build a CRAM lookup table from current CRAM contents.
        # CRAM is 64 bytes: 32 colours × 2 bytes (little-endian 12-bit BGR).
        cram_np = np.frombuffer(self.cram, dtype=np.uint8)
        lo   = cram_np[0::2].astype(np.uint16)   # 32 low bytes
        hi   = cram_np[1::2].astype(np.uint16)   # 32 high bytes
        word = lo | (hi << 8)
        r = ((word >> 0) & 0xF) * 17
        g = ((word >> 4) & 0xF) * 17
        b = ((word >> 8) & 0xF) * 17
        cram_table = np.stack([r, g, b], axis=1).astype(np.uint8)  # (32, 3)

        # Stack line buffers: each element is uint8[256]; result is (192, 256)
        buf = np.stack(self._line_buffer)   # (192, 256)

        # Crop to (144, 160) Game Gear window
        cropped = buf[CROP_Y:CROP_Y + SCREEN_H, CROP_X:CROP_X + SCREEN_W]  # (144, 160)

        # CRAM index → RGB lookup: (144, 160) → (144, 160, 3)
        frame_arr = cram_table[cropped]  # numpy fancy indexing

        self.frame       = frame_arr
        self.frame_ready = True
