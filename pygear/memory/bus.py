"""Memory bus — dispatches reads/writes across the Game Gear address space.

Address map
-----------
$0000–$03FF  ROM bank 0 low 1 KB (always fixed, Sega mapper only)
$0400–$3FFF  ROM slot 0
$4000–$7FFF  ROM slot 1
$8000–$BFFF  ROM slot 2  (or cartridge RAM if Sega mapper enables it)
$C000–$DFFF  8 KB main RAM
$E000–$FFFF  RAM mirror  ($E000–$FFFF mirrors $C000–$DFFF)
$FFFC–$FFFF  Sega mapper control registers (writes only; not used by Codemasters)

Mapper selection is automatic: Codemasters ROMs are identified by their
checksum header; all others use the standard Sega mapper.
"""

from .ram import RAM
from .mapper import SegaMapper, CodemastersMapper


class MemoryBus:
    def __init__(self, cart):
        self.ram = RAM()
        self.mapper = (
            CodemastersMapper(cart) if cart.is_codemasters else SegaMapper(cart)
        )

    # ------------------------------------------------------------------
    def reset(self):
        self.ram.reset()
        self.mapper.reset()

    def load_sav(self, path: str) -> bool:
        return self.mapper.load_sav(path)

    def save_sav(self, path: str) -> bool:
        return self.mapper.save_sav(path)

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
            # Codemasters bank registers live here; Sega mapper ignores these writes.
            self.mapper.write_rom_area(addr, value)
            return

        if addr < 0xC000:
            # Slot 2: Codemasters bank register at $8000; Sega cart RAM writes.
            self.mapper.write_slot2(addr, value)
            return

        # RAM mirror ($C000–$FFFF)
        self.ram.write(addr & 0x1FFF, value)

        # Mapper control registers live in the top of RAM (Sega mapper only).
        if addr >= 0xFFFC:
            self.mapper.write_register(addr & 0x03, value)
