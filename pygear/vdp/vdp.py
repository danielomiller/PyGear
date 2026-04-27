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
"""

VRAM_SIZE = 0x4000   # 16 KB
CRAM_SIZE = 64       # 32 colours × 2 bytes
NUM_REGS  = 11       # R0–R10

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

        # Timing state (updated by step() in Task 5)
        self._line  = 0         # current scanline 0–261
        self._cycle = 0         # T-states into current scanline

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
        self.status    = 0
        self._addr     = 0
        self._code     = 0
        self._latch    = False
        self._latch_lo = 0
        self._read_buf = 0
        self._line     = 0
        self._cycle    = 0

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
