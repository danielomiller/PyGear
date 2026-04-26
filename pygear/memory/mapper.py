"""Sega bank mapper.

Writes to $FFFC–$FFFF control ROM bank selection and cartridge-RAM banking.

  $FFFC  — mapper control register (cart-RAM bank + enable bits)
  $FFFD  — slot 0 bank select  ($0400–$3FFF)
  $FFFE  — slot 1 bank select  ($4000–$7FFF)
  $FFFF  — slot 2 bank select  ($8000–$BFFF)

The first 1 KB of ROM ($0000–$03FF) is always the first 1 KB of bank 0
regardless of mapper state.
"""

_CART_RAM_SIZE = 0x8000  # 32 KB cartridge RAM


class SegaMapper:
    def __init__(self, cart):
        self.cart = cart
        self._slots = [0, 1, 2]  # bank indices for slots 0/1/2
        self._ctrl = 0x00
        self._cart_ram = bytearray(_CART_RAM_SIZE)
        self._cart_ram_bank = 0
        self._cart_ram_enabled = False

    # ------------------------------------------------------------------
    def reset(self):
        self._slots = [0, 1, 2]
        self._ctrl = 0x00
        self._cart_ram_bank = 0
        self._cart_ram_enabled = False

    # ------------------------------------------------------------------
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

    @property
    def slot_banks(self):
        return tuple(self._slots)
