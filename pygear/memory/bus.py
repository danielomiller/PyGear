"""Memory bus — dispatches reads/writes across the Game Gear address space.

Address map
-----------
$0000–$03FF  ROM bank 0 low 1 KB (always fixed, Sega mapper only)
             — overlaid by the internal BIOS ROM while _bios_active is True
$0400–$3FFF  ROM slot 0
$4000–$7FFF  ROM slot 1
$8000–$BFFF  ROM slot 2  (or cartridge RAM if Sega mapper enables it)
$C000–$DFFF  8 KB main RAM
$E000–$FFFF  RAM mirror  ($E000–$FFFF mirrors $C000–$DFFF)
$FFFC–$FFFF  Sega mapper control registers (writes only; not used by Codemasters)

Mapper selection is automatic: Codemasters ROMs are identified by their
checksum header; all others use the standard Sega mapper.

BIOS ROM overlay
----------------
When a BIOS ROM is loaded via load_bios(), reads to $0000–$03FF return BIOS
data until port $3E is written with bit 4 set (BIOS disable).  The BIOS
writes that value itself once it has verified the cartridge header and is
ready to hand off to the game.  After the disable write, the cartridge's
bank-0 first 1 KB is exposed at $0000–$03FF as normal.
"""

from .ram import RAM
from .mapper import SegaMapper, CodemastersMapper

_BIOS_SIZE = 0x400   # Game Gear BIOS is at most 1 KB


class MemoryBus:
    def __init__(self, cart):
        self.ram    = RAM()
        self.mapper = (
            CodemastersMapper(cart) if cart.is_codemasters else SegaMapper(cart)
        )
        self._bios: bytes | None = None   # raw BIOS ROM bytes (immutable after load)
        self._bios_active = False          # True while BIOS overlays $0000–$03FF

    # ------------------------------------------------------------------
    def reset(self):
        self.ram.reset()
        self.mapper.reset()
        # Re-enable BIOS overlay on hard reset (power-cycle semantics).
        if self._bios is not None:
            self._bios_active = True

    def load_bios(self, data: bytes) -> None:
        """Install a BIOS ROM image.  Activates the overlay immediately."""
        self._bios = bytes(data[:_BIOS_SIZE])
        self._bios_active = True

    def set_mem_ctrl(self, value: int) -> None:
        """Handle a write to port $3E (memory control register).

        Bit 4 (0x10) = 1 disables the BIOS ROM overlay, exposing the
        cartridge's bank-0 first 1 KB at $0000–$03FF instead.
        """
        if value & 0x10:
            self._bios_active = False

    def load_sav(self, path: str) -> bool:
        return self.mapper.load_sav(path)

    def save_sav(self, path: str) -> bool:
        return self.mapper.save_sav(path)

    def get_state(self) -> dict:
        return {
            'ram':          self.ram.get_state(),
            'mapper':       self.mapper.get_state(),
            'bios_active':  self._bios_active,
        }

    def set_state(self, s: dict) -> None:
        self.ram.set_state(s['ram'])
        self.mapper.set_state(s['mapper'])
        # Only restore bios_active if a BIOS is loaded; fall back to False.
        self._bios_active = s.get('bios_active', False) and self._bios is not None

    # ------------------------------------------------------------------
    def read(self, addr: int) -> int:
        addr &= 0xFFFF

        # BIOS ROM overlay: $0000–$03FF
        if self._bios_active and addr < _BIOS_SIZE:
            return self._bios[addr]

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
