"""Sega and Codemasters bank mappers.

Sega mapper
-----------
Writes to $FFFC–$FFFF control ROM bank selection and cartridge-RAM banking.

  $FFFC  — mapper control register (cart-RAM bank + enable bits)
  $FFFD  — slot 0 bank select  ($0400–$3FFF)
  $FFFE  — slot 1 bank select  ($4000–$7FFF)
  $FFFF  — slot 2 bank select  ($8000–$BFFF)

The first 1 KB of ROM ($0000–$03FF) is always the first 1 KB of bank 0
regardless of mapper state.

Codemasters mapper
------------------
Bank registers are triggered by writes to ROM addresses:

  $0000  — slot 0 bank select ($0000–$3FFF)
  $4000  — slot 1 bank select ($4000–$7FFF)
  $8000  — slot 2 bank select + SRAM control
             bits 5-0: ROM bank for slot 2 ($A000–$BFFF)
             bit 7:    1 = enable 8 KB on-chip SRAM at $8000–$9FFF

All three slots span the full 16 KB (no fixed-1KB region).
When SRAM is enabled, $8000–$9FFF is overlaid by 8 KB of battery-backed
SRAM; $A000–$BFFF continues to read the banked ROM.  The bank-register
write to exactly $8000 does not propagate to SRAM offset 0.
"""

_CART_RAM_SIZE = 0x8000   # 32 KB Sega cartridge RAM
_CM_SRAM_SIZE  = 0x2000   # 8 KB Codemasters on-chip SRAM


class SegaMapper:
    def __init__(self, cart):
        self.cart = cart
        self._slots = [0, 1, 2]  # bank indices for slots 0/1/2
        self._ctrl = 0x00
        self._cart_ram = bytearray(_CART_RAM_SIZE)
        self._cart_ram_bank = 0
        self._cart_ram_enabled = False
        self._cart_ram_dirty = False   # True once any byte has been written

    # ------------------------------------------------------------------
    def reset(self):
        self._slots = [0, 1, 2]
        self._ctrl = 0x00
        self._cart_ram_bank = 0
        self._cart_ram_enabled = False
        # _cart_ram and _cart_ram_dirty intentionally not cleared —
        # battery-backed SRAM survives a console reset.

    # ------------------------------------------------------------------
    def write_rom_area(self, addr: int, value: int):
        """Called for writes to $0000–$7FFF. No-op for the Sega mapper."""

    def write_register(self, reg: int, value: int):
        """Called when CPU writes to $FFFC–$FFFF."""
        reg &= 0x03
        if reg == 0:  # $FFFC — mapper control
            self._ctrl = value
            self._cart_ram_enabled = bool(value & 0x08)
            self._cart_ram_bank = (value >> 2) & 0x01
        elif reg == 1:  # $FFFD — slot 0
            self._slots[0] = value % self.cart.bank_count
        elif reg == 2:  # $FFFE — slot 1
            self._slots[1] = value % self.cart.bank_count
        elif reg == 3:  # $FFFF — slot 2
            self._slots[2] = value % self.cart.bank_count

    # ------------------------------------------------------------------
    def read(self, addr: int) -> int:
        if addr < 0x0400:
            # First 1 KB always from bank 0
            return self.cart.read(0, addr)

        slot = addr >> 14  # 0, 1, or 2
        offset = addr & 0x3FFF
        bank = self._slots[slot]
        return self.cart.read(bank, offset)

    def read_slot2(self, addr: int) -> int:
        """Read from slot 2 area; may be cart RAM if enabled."""
        offset = addr & 0x3FFF
        if self._cart_ram_enabled and addr >= 0x8000:
            ram_addr = self._cart_ram_bank * 0x4000 + offset
            return self._cart_ram[ram_addr % _CART_RAM_SIZE]
        bank = self._slots[2]
        return self.cart.read(bank, offset)

    def write_slot2(self, addr: int, value: int):
        """Write to slot 2 area (only affects cart RAM if enabled)."""
        if self._cart_ram_enabled and addr >= 0x8000:
            offset = addr & 0x3FFF
            ram_addr = self._cart_ram_bank * 0x4000 + offset
            self._cart_ram[ram_addr % _CART_RAM_SIZE] = value & 0xFF
            self._cart_ram_dirty = True

    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        return {
            '_slots':            list(self._slots),
            '_ctrl':             self._ctrl,
            '_cart_ram':         bytes(self._cart_ram),
            '_cart_ram_bank':    self._cart_ram_bank,
            '_cart_ram_enabled': self._cart_ram_enabled,
            '_cart_ram_dirty':   self._cart_ram_dirty,
        }

    def set_state(self, s: dict) -> None:
        self._slots            = list(s['_slots'])
        self._ctrl             = s['_ctrl']
        self._cart_ram[:]      = s['_cart_ram']
        self._cart_ram_bank    = s['_cart_ram_bank']
        self._cart_ram_enabled = s['_cart_ram_enabled']
        self._cart_ram_dirty   = s['_cart_ram_dirty']

    def load_sav(self, path: str) -> bool:
        """Load cart RAM from *path*. Returns True on success, False if not found."""
        try:
            with open(path, "rb") as f:
                data = f.read(_CART_RAM_SIZE)
            self._cart_ram[:len(data)] = data
            return True
        except FileNotFoundError:
            return False

    def save_sav(self, path: str) -> bool:
        """Write cart RAM to *path*. Returns True if written, False if nothing to save."""
        if not self._cart_ram_dirty:
            return False
        with open(path, "wb") as f:
            f.write(self._cart_ram)
        return True

    @property
    def slot_banks(self):
        return tuple(self._slots)


# ---------------------------------------------------------------------------

class CodemastersMapper:
    """Codemasters bank mapper with optional on-chip SRAM.

    Bank registers are written to ROM addresses $0000, $4000, $8000.
    All three 16 KB slots are fully switchable (no fixed-1KB region).
    Writing to $8000: bit 7 enables 8 KB SRAM at $8000–$9FFF;
                      bits 5-0 set the ROM bank for $A000–$BFFF.
    """

    def __init__(self, cart):
        self.cart = cart
        self._slots       = [0, 1, 2]
        self._sram        = bytearray(_CM_SRAM_SIZE)
        self._sram_enabled = False
        self._sram_dirty  = False

    # ------------------------------------------------------------------
    def reset(self):
        self._slots        = [0, 1, 2]
        self._sram_enabled = False
        # _sram and _sram_dirty preserved — battery-backed SRAM survives reset.

    # ------------------------------------------------------------------
    def write_rom_area(self, addr: int, value: int):
        """Handle bank register writes at $0000 and $4000."""
        if addr == 0x0000:
            self._slots[0] = value % self.cart.bank_count
        elif addr == 0x4000:
            self._slots[1] = value % self.cart.bank_count

    def write_register(self, reg: int, value: int):
        """No mapper control registers in RAM for the Codemasters mapper."""

    # ------------------------------------------------------------------
    def read(self, addr: int) -> int:
        slot   = addr >> 14   # 0 or 1
        offset = addr & 0x3FFF
        return self.cart.read(self._slots[slot], offset)

    def read_slot2(self, addr: int) -> int:
        # $8000–$9FFF → SRAM when enabled; $A000–$BFFF always → banked ROM.
        if self._sram_enabled and (addr & 0xE000) == 0x8000:
            return self._sram[addr & 0x1FFF]
        return self.cart.read(self._slots[2], addr & 0x3FFF)

    def write_slot2(self, addr: int, value: int):
        if addr == 0x8000:
            # Bank register: bit 7 = SRAM enable, bits 5-0 = ROM bank.
            self._sram_enabled = bool(value & 0x80)
            self._slots[2]     = (value & 0x3F) % self.cart.bank_count
            return   # bank-register write does not propagate into SRAM[0]
        if self._sram_enabled and (addr & 0xE000) == 0x8000:
            self._sram[addr & 0x1FFF] = value & 0xFF
            self._sram_dirty = True

    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        return {
            '_slots':        list(self._slots),
            '_sram':         bytes(self._sram),
            '_sram_enabled': self._sram_enabled,
            '_sram_dirty':   self._sram_dirty,
        }

    def set_state(self, s: dict) -> None:
        self._slots        = list(s['_slots'])
        self._sram[:]      = s['_sram']
        self._sram_enabled = s['_sram_enabled']
        self._sram_dirty   = s['_sram_dirty']

    def load_sav(self, path: str) -> bool:
        """Load on-chip SRAM from *path*. Returns True on success."""
        try:
            with open(path, "rb") as f:
                data = f.read(_CM_SRAM_SIZE)
            self._sram[:len(data)] = data
            return True
        except FileNotFoundError:
            return False

    def save_sav(self, path: str) -> bool:
        """Write on-chip SRAM to *path*. Returns True if file was written."""
        if not self._sram_dirty:
            return False
        with open(path, "wb") as f:
            f.write(self._sram)
        return True

    @property
    def slot_banks(self):
        return tuple(self._slots)
