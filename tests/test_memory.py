"""Tests for the memory subsystem — bus, mapper, RAM."""

import sys
import os
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pygear.cartridge import Cartridge
from pygear.memory.ram import RAM
from pygear.memory.mapper import SegaMapper
from pygear.memory.bus import MemoryBus


# ---------------------------------------------------------------------------
# Helpers

def _make_cart(size_kb: int = 64) -> Cartridge:
    """Create an in-memory fake cartridge of the given size (in KB)."""
    size = size_kb * 1024
    data = bytearray(size)
    # Fill each 16 KB bank with its bank index so reads are identifiable
    for bank in range(size // 0x4000):
        for offset in range(0x4000):
            data[bank * 0x4000 + offset] = bank & 0xFF
    data[0x0000] = 0xAA
    data[0x0200] = 0xBB
    tmp = tempfile.NamedTemporaryFile(suffix=".gg", delete=False)
    tmp.write(data)
    tmp.close()
    cart = Cartridge(tmp.name)
    os.unlink(tmp.name)
    return cart


def _cart_from_bytes(data: bytes) -> Cartridge:
    """Load a Cartridge from raw bytes."""
    tmp = tempfile.NamedTemporaryFile(suffix=".gg", delete=False)
    tmp.write(data)
    tmp.close()
    cart = Cartridge(tmp.name)
    os.unlink(tmp.name)
    return cart


_SEGA_MAGIC = b"TMR SEGA"


def _rom_with_header(size: int, offset: int,
                     product_code: int = 0, version: int = 0,
                     region: int = 0, rom_size_byte: int = 0) -> Cartridge:
    """Build a ROM of *size* bytes with a valid Sega header at *offset*."""
    data = bytearray(size)
    data[offset : offset + 8] = _SEGA_MAGIC
    data[offset + 12] = product_code & 0xFF
    data[offset + 13] = (product_code >> 8) & 0xFF
    data[offset + 14] = ((product_code >> 12) & 0xF0) | (version & 0x0F)
    data[offset + 15] = ((region & 0x0F) << 4) | (rom_size_byte & 0x0F)
    return _cart_from_bytes(bytes(data))


# ---------------------------------------------------------------------------
# TestRAM

class TestRAM:
    def test_read_after_write(self):
        ram = RAM()
        ram.write(0x0010, 0xAB)
        assert ram.read(0x0010) == 0xAB

    def test_8kb_wrap(self):
        ram = RAM()
        ram.write(0x0000, 0x42)
        assert ram.read(0x2000) == 0x42

    def test_byte_masking(self):
        ram = RAM()
        ram.write(0x0005, 0x1FF)
        assert ram.read(0x0005) == 0xFF

    def test_reset(self):
        ram = RAM()
        ram.write(0x0100, 0x77)
        ram.reset()
        assert ram.read(0x0100) == 0x00


# ---------------------------------------------------------------------------
# TestCartridge

class TestCartridge:

    # --- basic read / bank_count (original tests) --------------------------

    def test_bank_count_64kb(self):
        assert _make_cart(64).bank_count == 4

    def test_read_bank_data(self):
        cart = _make_cart(64)
        assert cart.read(0, 0x1000) == 0x00
        assert cart.read(1, 0x1000) == 0x01
        assert cart.read(2, 0x1000) == 0x02

    def test_read_wraps_past_rom_end(self):
        cart = _make_cart(32)
        assert cart.read(4, 0x1000) == cart.read(0, 0x1000)

    def test_fixed_1kb_marker(self):
        cart = _make_cart(64)
        assert cart.read_raw(0x0000) == 0xAA
        assert cart.read_raw(0x0200) == 0xBB

    # --- bank_count for various sizes --------------------------------------

    def test_bank_count_16kb(self):
        assert _make_cart(16).bank_count == 1

    def test_bank_count_32kb(self):
        assert _make_cart(32).bank_count == 2

    def test_bank_count_512kb(self):
        assert _make_cart(512).bank_count == 32

    def test_bank_count_minimum_1_for_tiny_rom(self):
        cart = _cart_from_bytes(bytes(8 * 1024))
        assert cart.bank_count == 1

    # --- read_raw wrapping -------------------------------------------------

    def test_read_raw_wraps_at_rom_end(self):
        cart = _make_cart(16)
        assert cart.read_raw(0) == cart.read_raw(16 * 1024)

    # --- header: absent magic ----------------------------------------------

    def test_header_invalid_when_no_magic(self):
        cart = _cart_from_bytes(bytes(32 * 1024))
        assert cart.header_valid is False

    def test_header_invalid_when_rom_too_small_for_any_probe(self):
        # All three probe offsets (0x1FF0, 0x3FF0, 0x7FF0) exceed 4 KB
        cart = _cart_from_bytes(bytes(4 * 1024))
        assert cart.header_valid is False

    def test_header_fields_default_to_zero_when_absent(self):
        cart = _cart_from_bytes(bytes(32 * 1024))
        assert cart.product_code  == 0
        assert cart.version       == 0
        assert cart.region        == 0
        assert cart.rom_size_byte == 0

    # --- header: detection at each probe offset ----------------------------

    def test_header_detected_at_offset_0x1ff0(self):
        cart = _rom_with_header(size=8 * 1024, offset=0x1FF0)
        assert cart.header_valid is True

    def test_header_detected_at_offset_0x3ff0(self):
        cart = _rom_with_header(size=16 * 1024, offset=0x3FF0)
        assert cart.header_valid is True

    def test_header_detected_at_offset_0x7ff0(self):
        cart = _rom_with_header(size=32 * 1024, offset=0x7FF0)
        assert cart.header_valid is True

    def test_first_matching_offset_wins(self):
        # 0x1FF0 and 0x3FF0 both have magic; 0x1FF0 has product_code low=0x11
        data = bytearray(32 * 1024)
        data[0x1FF0 : 0x1FF0 + 8] = _SEGA_MAGIC
        data[0x1FF0 + 12] = 0x11
        data[0x3FF0 : 0x3FF0 + 8] = _SEGA_MAGIC
        data[0x3FF0 + 12] = 0x22
        cart = _cart_from_bytes(bytes(data))
        assert cart.header_valid is True
        assert cart.product_code & 0xFF == 0x11

    # --- header: field extraction ------------------------------------------

    def test_product_code_low_byte(self):
        assert _rom_with_header(8*1024, 0x1FF0, product_code=0x42).product_code == 0x42

    def test_product_code_two_bytes(self):
        assert _rom_with_header(8*1024, 0x1FF0, product_code=0x0142).product_code == 0x0142

    def test_product_code_full_20_bits(self):
        assert _rom_with_header(8*1024, 0x1FF0, product_code=0x30142).product_code == 0x30142

    def test_version_extracted(self):
        assert _rom_with_header(8*1024, 0x1FF0, version=7).version == 7

    def test_version_max(self):
        assert _rom_with_header(8*1024, 0x1FF0, version=0xF).version == 0xF

    def test_region_extracted(self):
        assert _rom_with_header(8*1024, 0x1FF0, region=5).region == 5

    def test_rom_size_byte_extracted(self):
        assert _rom_with_header(8*1024, 0x1FF0, rom_size_byte=0x0E).rom_size_byte == 0x0E

    # --- copier header stripping -------------------------------------------

    def test_copier_header_stripped_size(self):
        padded = bytes(512) + bytes(32 * 1024)
        assert _cart_from_bytes(padded).size == 32 * 1024

    def test_copier_header_bank_count_after_strip(self):
        padded = bytes(512) + bytes(32 * 1024)
        assert _cart_from_bytes(padded).bank_count == 2

    def test_copier_header_data_accessible_after_strip(self):
        rom = bytearray(32 * 1024)
        rom[0x100] = 0xAB
        padded = bytes(512) + bytes(rom)
        assert _cart_from_bytes(padded).read_raw(0x100) == 0xAB

    def test_regular_rom_not_stripped(self):
        rom = bytearray(32 * 1024)
        rom[0] = 0xCC
        cart = _cart_from_bytes(bytes(rom))
        assert cart.size == 32 * 1024
        assert cart.read_raw(0) == 0xCC


# ---------------------------------------------------------------------------
# TestSegaMapper

class TestSegaMapper:
    def test_default_slot_banks(self):
        mapper = SegaMapper(_make_cart(64))
        assert mapper.slot_banks == (0, 1, 2)

    def test_slot0_bank_select(self):
        mapper = SegaMapper(_make_cart(64))
        mapper.write_register(1, 3)
        assert mapper.slot_banks[0] == 3

    def test_slot1_bank_select(self):
        mapper = SegaMapper(_make_cart(64))
        mapper.write_register(2, 2)
        assert mapper.slot_banks[1] == 2

    def test_slot2_bank_select(self):
        mapper = SegaMapper(_make_cart(64))
        mapper.write_register(3, 1)
        assert mapper.slot_banks[2] == 1

    def test_fixed_1kb_unaffected_by_mapper(self):
        mapper = SegaMapper(_make_cart(64))
        mapper.write_register(1, 3)
        assert mapper.read(0x0000) == 0xAA
        assert mapper.read(0x0200) == 0xBB

    def test_slot0_read_reflects_bank(self):
        mapper = SegaMapper(_make_cart(64))
        mapper.write_register(1, 3)
        assert mapper.read(0x0400) == 0x03

    def test_bank_wraps_at_rom_boundary(self):
        mapper = SegaMapper(_make_cart(32))
        mapper.write_register(1, 5)   # 5 % 2 = 1
        assert mapper.slot_banks[0] == 1

    def test_reset(self):
        mapper = SegaMapper(_make_cart(64))
        mapper.write_register(1, 3)
        mapper.write_register(2, 2)
        mapper.write_register(3, 1)
        mapper.reset()
        assert mapper.slot_banks == (0, 1, 2)

    # --- $FFFC control register --------------------------------------------

    def test_fffc_bit3_enables_cart_ram(self):
        mapper = SegaMapper(_make_cart(64))
        mapper.write_register(0, 0x08)   # bit 3 set
        assert mapper._cart_ram_enabled is True

    def test_fffc_bit3_clear_disables_cart_ram(self):
        mapper = SegaMapper(_make_cart(64))
        mapper.write_register(0, 0x08)
        mapper.write_register(0, 0x00)   # clear bit 3
        assert mapper._cart_ram_enabled is False

    def test_fffc_bit2_clear_selects_cart_ram_bank_0(self):
        mapper = SegaMapper(_make_cart(64))
        mapper.write_register(0, 0x08)   # enable, bank bit = 0
        assert mapper._cart_ram_bank == 0

    def test_fffc_bit2_set_selects_cart_ram_bank_1(self):
        mapper = SegaMapper(_make_cart(64))
        mapper.write_register(0, 0x0C)   # bits 3 + 2 set
        assert mapper._cart_ram_bank == 1

    # --- cart RAM write / read when enabled --------------------------------

    def test_write_slot2_stores_in_cart_ram_when_enabled(self):
        mapper = SegaMapper(_make_cart(64))
        mapper.write_register(0, 0x08)   # enable cart RAM
        mapper.write_slot2(0x8010, 0xAB)
        assert mapper.read_slot2(0x8010) == 0xAB

    def test_write_slot2_ignored_when_disabled(self):
        mapper = SegaMapper(_make_cart(64))
        # cart RAM disabled (default)
        mapper.write_slot2(0x8010, 0xAB)
        mapper.write_register(0, 0x08)   # now enable to peek inside
        assert mapper.read_slot2(0x8010) == 0x00   # nothing was written

    def test_read_slot2_returns_rom_when_disabled(self):
        mapper = SegaMapper(_make_cart(64))
        # slot 2 default = bank 2, which is filled with 0x02
        assert mapper.read_slot2(0x8000) == 0x02

    def test_read_slot2_returns_cart_ram_when_enabled(self):
        mapper = SegaMapper(_make_cart(64))
        mapper.write_register(0, 0x08)
        mapper.write_slot2(0x8005, 0x77)
        # now enable is set; reading back should give 0x77, not ROM
        assert mapper.read_slot2(0x8005) == 0x77

    def test_write_slot2_masks_to_byte(self):
        mapper = SegaMapper(_make_cart(64))
        mapper.write_register(0, 0x08)
        mapper.write_slot2(0x8000, 0x1FF)   # only low byte stored
        assert mapper.read_slot2(0x8000) == 0xFF

    # --- cart RAM bank independence ----------------------------------------

    def test_cart_ram_bank0_and_bank1_are_independent(self):
        mapper = SegaMapper(_make_cart(64))
        # Write 0xAA to bank 0, offset 0x0010
        mapper.write_register(0, 0x08)   # bank 0, enabled
        mapper.write_slot2(0x8010, 0xAA)
        # Switch to bank 1 and write 0xBB at same offset
        mapper.write_register(0, 0x0C)   # bank 1, enabled
        mapper.write_slot2(0x8010, 0xBB)
        assert mapper.read_slot2(0x8010) == 0xBB
        # Switch back to bank 0 — should still be 0xAA
        mapper.write_register(0, 0x08)
        assert mapper.read_slot2(0x8010) == 0xAA

    def test_cart_ram_bank1_offset_is_0x4000_into_array(self):
        mapper = SegaMapper(_make_cart(64))
        mapper.write_register(0, 0x0C)   # bank 1, enabled
        mapper.write_slot2(0x8000, 0x55)
        # Verify it landed at position 0x4000 in the raw array
        assert mapper._cart_ram[0x4000] == 0x55

    def test_cart_ram_bank0_offset_is_0x0000_into_array(self):
        mapper = SegaMapper(_make_cart(64))
        mapper.write_register(0, 0x08)   # bank 0, enabled
        mapper.write_slot2(0x8001, 0x33)
        assert mapper._cart_ram[0x0001] == 0x33

    # --- reset clears cart RAM state ---------------------------------------

    def test_reset_disables_cart_ram(self):
        mapper = SegaMapper(_make_cart(64))
        mapper.write_register(0, 0x08)
        mapper.reset()
        assert mapper._cart_ram_enabled is False

    def test_reset_clears_cart_ram_bank_to_0(self):
        mapper = SegaMapper(_make_cart(64))
        mapper.write_register(0, 0x0C)   # bank 1
        mapper.reset()
        assert mapper._cart_ram_bank == 0

    def test_reset_restores_slot_banks_alongside_cart_ram(self):
        mapper = SegaMapper(_make_cart(64))
        mapper.write_register(1, 3)
        mapper.write_register(0, 0x08)
        mapper.reset()
        assert mapper.slot_banks == (0, 1, 2)
        assert mapper._cart_ram_enabled is False


# ---------------------------------------------------------------------------
# TestMemoryBus

class TestMemoryBus:
    def test_rom_read_slot0(self):
        assert MemoryBus(_make_cart(64)).read(0x0400) == 0x00

    def test_rom_read_slot1(self):
        assert MemoryBus(_make_cart(64)).read(0x4000) == 0x01

    def test_rom_read_slot2(self):
        assert MemoryBus(_make_cart(64)).read(0x8000) == 0x02

    def test_ram_read_write(self):
        bus = MemoryBus(_make_cart(64))
        bus.write(0xC000, 0x55)
        assert bus.read(0xC000) == 0x55

    def test_ram_mirror(self):
        bus = MemoryBus(_make_cart(64))
        bus.write(0xC100, 0x33)
        assert bus.read(0xE100) == 0x33

    def test_rom_write_ignored(self):
        bus = MemoryBus(_make_cart(64))
        original = bus.read(0x1000)
        bus.write(0x1000, original ^ 0xFF)
        assert bus.read(0x1000) == original

    def test_mapper_register_write_via_bus(self):
        bus = MemoryBus(_make_cart(64))
        bus.write(0xFFFF, 3)
        assert bus.mapper.slot_banks[2] == 3
        assert bus.read(0x8000) == 0x03

    def test_mapper_slot0_via_fffd(self):
        bus = MemoryBus(_make_cart(64))
        bus.write(0xFFFD, 3)
        assert bus.read(0x0400) == 0x03

    def test_fixed_1kb_immutable(self):
        bus = MemoryBus(_make_cart(64))
        bus.write(0xFFFD, 3)
        assert bus.read(0x0000) == 0xAA
        assert bus.read(0x0200) == 0xBB

    # --- all four mapper registers via bus ---------------------------------

    def test_fffc_via_bus_enables_cart_ram(self):
        bus = MemoryBus(_make_cart(64))
        bus.write(0xFFFC, 0x08)          # bit 3 → cart RAM enable
        assert bus.mapper._cart_ram_enabled is True

    def test_fffe_via_bus_sets_slot1(self):
        bus = MemoryBus(_make_cart(64))
        bus.write(0xFFFE, 3)             # $FFFE → slot 1 = bank 3
        assert bus.mapper.slot_banks[1] == 3
        assert bus.read(0x4000) == 0x03

    def test_ffff_via_bus_sets_slot2(self):
        bus = MemoryBus(_make_cart(64))
        bus.write(0xFFFF, 1)             # $FFFF → slot 2 = bank 1
        assert bus.mapper.slot_banks[2] == 1

    def test_all_four_mapper_registers_written_through_bus(self):
        bus = MemoryBus(_make_cart(64))
        bus.write(0xFFFC, 0x00)          # $FFFC: cart RAM off, bank 0
        bus.write(0xFFFD, 3)             # $FFFD: slot 0 = bank 3
        bus.write(0xFFFE, 2)             # $FFFE: slot 1 = bank 2
        bus.write(0xFFFF, 1)             # $FFFF: slot 2 = bank 1
        assert bus.mapper.slot_banks == (3, 2, 1)

    def test_mapper_register_not_triggered_below_fffc(self):
        # Writes to $FFFB and below land in RAM but do not update mapper
        bus = MemoryBus(_make_cart(64))
        before = bus.mapper.slot_banks
        bus.write(0xFFFB, 3)
        assert bus.mapper.slot_banks == before

    # --- cart RAM through the bus ($8000–$BFFF) ----------------------------

    def test_cart_ram_write_and_read_via_bus(self):
        bus = MemoryBus(_make_cart(64))
        bus.write(0xFFFC, 0x08)          # enable cart RAM (bank 0)
        bus.write(0x8100, 0xAB)
        assert bus.read(0x8100) == 0xAB

    def test_cart_ram_byte_masking_via_bus(self):
        bus = MemoryBus(_make_cart(64))
        bus.write(0xFFFC, 0x08)
        bus.write(0x8000, 0x1FF)         # only low byte stored
        assert bus.read(0x8000) == 0xFF

    def test_write_to_slot2_without_cart_ram_has_no_effect_on_read(self):
        # Cart RAM disabled → write to $8000–$BFFF is silently dropped
        bus = MemoryBus(_make_cart(64))
        original = bus.read(0x8000)      # ROM value (bank 2 = 0x02)
        bus.write(0x8000, original ^ 0xFF)
        assert bus.read(0x8000) == original

    def test_cart_ram_bank1_via_bus(self):
        bus = MemoryBus(_make_cart(64))
        bus.write(0xFFFC, 0x08)          # bank 0, enabled
        bus.write(0x8050, 0x11)
        bus.write(0xFFFC, 0x0C)          # bank 1, enabled
        bus.write(0x8050, 0x22)
        # Back to bank 0 → 0x11 unchanged
        bus.write(0xFFFC, 0x08)
        assert bus.read(0x8050) == 0x11
        # Back to bank 1 → 0x22
        bus.write(0xFFFC, 0x0C)
        assert bus.read(0x8050) == 0x22

    # --- writes to ROM regions are ignored ---------------------------------

    def test_write_to_slot0_ignored(self):
        bus = MemoryBus(_make_cart(64))
        original = bus.read(0x1000)
        bus.write(0x1000, original ^ 0xFF)
        assert bus.read(0x1000) == original

    def test_write_to_slot1_ignored(self):
        bus = MemoryBus(_make_cart(64))
        original = bus.read(0x5000)
        bus.write(0x5000, original ^ 0xFF)
        assert bus.read(0x5000) == original

    def test_write_to_slot0_boundary_7fff_ignored(self):
        bus = MemoryBus(_make_cart(64))
        original = bus.read(0x7FFF)
        bus.write(0x7FFF, original ^ 0xFF)
        assert bus.read(0x7FFF) == original

    # --- RAM region boundaries and mirrors ---------------------------------

    def test_ram_write_at_dfff(self):
        bus = MemoryBus(_make_cart(64))
        bus.write(0xDFFF, 0x99)
        assert bus.read(0xDFFF) == 0x99

    def test_ram_mirror_write_at_e000_readable_at_c000(self):
        bus = MemoryBus(_make_cart(64))
        bus.write(0xE200, 0x44)
        assert bus.read(0xC200) == 0x44  # $E200 mirrors $C200

    def test_ram_mirror_write_at_c000_readable_at_e000(self):
        bus = MemoryBus(_make_cart(64))
        bus.write(0xC300, 0x66)
        assert bus.read(0xE300) == 0x66

    # --- reset -------------------------------------------------------------

    def test_reset_clears_ram_content(self):
        bus = MemoryBus(_make_cart(64))
        bus.write(0xC000, 0x42)
        bus.reset()
        assert bus.read(0xC000) == 0x00

    def test_reset_restores_default_slot_mapping(self):
        bus = MemoryBus(_make_cart(64))
        bus.write(0xFFFD, 3)
        bus.write(0xFFFE, 2)
        bus.write(0xFFFF, 1)
        bus.reset()
        assert bus.mapper.slot_banks == (0, 1, 2)

    def test_reset_disables_cart_ram(self):
        bus = MemoryBus(_make_cart(64))
        bus.write(0xFFFC, 0x08)          # enable cart RAM
        bus.reset()
        assert bus.mapper._cart_ram_enabled is False

    def test_reset_rom_reads_work_after_reset(self):
        bus = MemoryBus(_make_cart(64))
        bus.write(0xFFFF, 3)             # remap slot 2 to bank 3
        bus.reset()
        assert bus.read(0x8000) == 0x02  # back to default bank 2

    # --- address and value masking -----------------------------------------

    def test_16bit_address_masking_on_read(self):
        bus = MemoryBus(_make_cart(64))
        bus.write(0xC000, 0x77)
        assert bus.read(0x1C000) == 0x77   # 0x1C000 & 0xFFFF == 0xC000

    def test_16bit_address_masking_on_write(self):
        bus = MemoryBus(_make_cart(64))
        bus.write(0x1C010, 0x88)            # 0x1C010 & 0xFFFF == 0xC010
        assert bus.read(0xC010) == 0x88

    def test_byte_value_masking_on_write(self):
        bus = MemoryBus(_make_cart(64))
        bus.write(0xC000, 0x1AB)            # only low byte stored
        assert bus.read(0xC000) == 0xAB
