"""Sega Game Gear VDP — core: VRAM, CRAM, register file, and port I/O.

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

VRAM_SIZE = 0x4000   # 16 KB
CRAM_SIZE = 64       # 32 colours × 2 bytes
NUM_REGS  = 11       # R0–R10

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
        for i in range(VRAM_SIZE): self.vram[i] = 0
        for i in range(CRAM_SIZE): self.cram[i] = 0
        for i in range(NUM_REGS):  self.regs[i] = 0
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
    def render_line(self, line: int) -> list:
        """Render 256 pixels of the background layer for *line*.

        Returns a list of 256 (cram_index, priority) tuples.
        cram_index is 0–31 (colour index 0–15 plus palette offset 0 or 16).
        priority is True when the name-table entry has bit 12 set.

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
        bit 7  V-scroll lock: right 8 columns (screen_x >= 192) ignore R9
        """
        r0        = self.regs[0]
        h_scroll  = self.regs[8]
        v_scroll  = self.regs[9]
        name_base = (self.regs[2] & 0x0E) << 10

        hscroll_lock = bool(r0 & 0x40)
        vscroll_lock = bool(r0 & 0x80)

        eff_h = 0 if (hscroll_lock and line < 16) else h_scroll

        # Scrolled vertical position (wraps at 224 = 28 × 8)
        bg_y      = (line + v_scroll) % 224
        tile_row  = bg_y >> 3
        pixel_row = bg_y & 7

        # Unscrolled vertical position (used by V-scroll-locked columns)
        ly_lock     = line % 224
        tile_row_l  = ly_lock >> 3
        prow_l      = ly_lock & 7

        vram   = self.vram
        result = []

        for screen_x in range(256):
            if vscroll_lock and screen_x >= 192:
                tr = tile_row_l
                pr = prow_l
            else:
                tr = tile_row
                pr = pixel_row

            bg_x      = (screen_x + eff_h) & 0xFF
            tile_col  = bg_x >> 3
            pixel_col = bg_x & 7

            # Read 2-byte name-table entry (little-endian)
            nt_off = name_base + (tr * 32 + tile_col) * 2
            lo     = vram[nt_off       & 0x3FFF]
            hi     = vram[(nt_off + 1) & 0x3FFF]
            entry  = lo | (hi << 8)

            tile_num = entry & 0x1FF
            hflip    = bool(entry & 0x0200)
            vflip    = bool(entry & 0x0400)
            palette  = (entry >> 11) & 1
            priority = bool(entry & 0x1000)

            # Pixel row and column inside tile (apply flips)
            row   = (7 - pr)        if vflip else pr
            col   = (7 - pixel_col) if hflip else pixel_col
            shift = 7 - col
            base  = tile_num * 32 + row * 4

            color_idx = (
                 ((vram[base    ] >> shift) & 1)
                | ((vram[base + 1] >> shift) & 1) << 1
                | ((vram[base + 2] >> shift) & 1) << 2
                | ((vram[base + 3] >> shift) & 1) << 3
            )

            result.append((color_idx + palette * 16, priority))

        return result

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
            # Render scanline into buffer
            self._line_buffer[line] = self.render_line(line)
            # Line interrupt counter (R0 bit 4 enables, R10 is reload value)
            if self._line_irq == 0:
                self._line_irq = self.regs[10]
                if (self.regs[0] & 0x10) and self._cpu:
                    self._cpu.request_interrupt()
            else:
                self._line_irq -= 1
        else:
            # VBlank lines: reload counter from R10 each line (no fire)
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
        frame = []
        for row in range(SCREEN_H):
            src_row = self._line_buffer[row + CROP_Y]
            frame_row = []
            for col in range(SCREEN_W):
                cram_idx, _ = src_row[col + CROP_X]
                frame_row.append(self.cram_color(cram_idx))
            frame.append(frame_row)
        self.frame       = frame
        self.frame_ready = True
