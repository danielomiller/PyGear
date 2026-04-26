"""Memory bus — dispatches reads/writes across the Game Gear address space.

Address map
-----------
$0000–$03FF  ROM bank 0 low 1 KB (always fixed)
$0400–$3FFF  ROM slot 0
$4000–$7FFF  ROM slot 1
$8000–$BFFF  ROM slot 2  (or cartridge RAM if mapper enables it)
$C000–$DFFF  8 KB main RAM
$E000–$FFFF  RAM mirror  ($E000–$FFFF mirrors $C000–$DFFF except mapper regs)
$FFFC–$FFFF  Sega mapper control registers (writes only)
"""

from .ram import RAM
from .mapper import SegaMapper


class MemoryBus:
    def __init__(self, cart):
        self.ram = RAM()
        self.mapper = SegaMapper(cart)

    # ------------------------------------------------------------------
    def reset(self):
        self.ram.reset()
        self.mapper.reset()

    # ------------------------------------------------------------------
    def read(self, addr: int) -> int:
        addr &= 0xFFFF

        if addr < 0x8000:
            return self.mapper.read(addr)

        if addr < 0xC000:
            return self.mapper.read_slot2(addr)

        # RAM region ($C000–$FFFF)
        return self.ram.read(addr & 0x1FFF)

    # ------------------------------------------------------------------
    def write(self, addr: int, value: int):
        addr &= 0xFFFF
        value &= 0xFF

        if addr < 0x8000:
            # ROM — only mapper regs inside first 1KB area matter; normally ignored
            return

        if addr < 0xC000:
            # Slot 2: write to cartridge RAM if enabled
            self.mapper.write_slot2(addr, value)
            return

        # RAM mirror ($C000–$FFFF)
        self.ram.write(addr & 0x1FFF, value)

        # Mapper control registers live in the top of RAM
        if addr >= 0xFFFC:
            self.mapper.write_register(addr & 0x03, value)
