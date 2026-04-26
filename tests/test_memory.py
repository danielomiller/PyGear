"""Tests for the memory subsystem — bus, mapper, RAM."""

import io
import sys
import os
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
    # Write a recognisable byte at offset 0 of bank 0 (first 1 KB area)
    data[0x0000] = 0xAA
    data[0x0200] = 0xBB

    # Persist to a temp file and load via Cartridge
    import tempfile, pathlib
    tmp = tempfile.NamedTemporaryFile(suffix=".gg", delete=False)
    tmp.write(data)
    tmp.close()
    cart = Cartridge(tmp.name)
    os.unlink(tmp.name)
    return cart


# ---------------------------------------------------------------------------
# RAM tests

class TestRAM:
    def test_read_after_write(self):
        ram = RAM()
        ram.write(0x0010, 0xAB)
        assert ram.read(0x0010) == 0xAB

    def test_8kb_wrap(self):
        ram = RAM()
        ram.write(0x0000, 0x42)
        assert ram.read(0x2000) == 0x42  # mirrors at +8 KB

    def test_byte_masking(self):
        ram = RAM()
        ram.write(0x0005, 0x1FF)  # only low 8 bits stored
        assert ram.read(0x0005) == 0xFF

    def test_reset(self):
        ram = RAM()
        ram.write(0x0100, 0x77)
        ram.reset()
        assert ram.read(0x0100) == 0x00


# ---------------------------------------------------------------------------
# Cartridge tests

class TestCartridge:
    def test_bank_count(self):
        cart = _make_cart(64)
        assert cart.bank_count == 4  # 64 KB / 16 KB

    def test_read_bank_data(self):
        cart = _make_cart(64)
        # Each bank is filled with its bank index; use offsets not overlapping markers
        assert cart.read(0, 0x1000) == 0x00
        assert cart.read(1, 0x1000) == 0x01
        assert cart.read(2, 0x1000) == 0x02

    def test_read_wraps_past_rom_end(self):
        cart = _make_cart(32)
        # Reading bank 4 of a 32 KB (2-bank) ROM: 4 % 2 = 0 → same as bank 0
        assert cart.read(4, 0x1000) == cart.read(0, 0x1000)

    def test_fixed_1kb_marker(self):
        cart = _make_cart(64)
        assert cart.read_raw(0x0000) == 0xAA
        assert cart.read_raw(0x0200) == 0xBB


# ---------------------------------------------------------------------------
# SegaMapper tests

class TestSegaMapper:
    def _make(self, size_kb=64):
        return _make_cart(size_kb), None

    def test_default_slot_banks(self):
        cart = _make_cart(64)
        mapper = SegaMapper(cart)
        assert mapper.slot_banks == (0, 1, 2)

    def test_slot0_bank_select(self):
        cart = _make_cart(64)
        mapper = SegaMapper(cart)
        mapper.write_register(1, 3)  # $FFFD = 3 → slot 0 = bank 3
        assert mapper.slot_banks[0] == 3

    def test_slot1_bank_select(self):
        cart = _make_cart(64)
        mapper = SegaMapper(cart)
        mapper.write_register(2, 2)  # $FFFE = 2 → slot 1 = bank 2
        assert mapper.slot_banks[1] == 2

    def test_slot2_bank_select(self):
        cart = _make_cart(64)
        mapper = SegaMapper(cart)
        mapper.write_register(3, 1)  # $FFFF = 1 → slot 2 = bank 1
        assert mapper.slot_banks[2] == 1

    def test_fixed_1kb_unaffected_by_mapper(self):
        cart = _make_cart(64)
        mapper = SegaMapper(cart)
        # Map slot 0 to bank 3
        mapper.write_register(1, 3)
        # $0000–$03FF should still read from bank 0
        assert mapper.read(0x0000) == 0xAA
        assert mapper.read(0x0200) == 0xBB

    def test_slot0_read_reflects_bank(self):
        cart = _make_cart(64)
        mapper = SegaMapper(cart)
        mapper.write_register(1, 3)
        # $0400 onward in slot 0 should now come from bank 3 (value = 0x03)
        assert mapper.read(0x0400) == 0x03

    def test_bank_wraps_at_rom_boundary(self):
        cart = _make_cart(32)  # 2 banks
        mapper = SegaMapper(cart)
        # Writing bank 5 to slot 0 of a 2-bank ROM: 5 % 2 = 1
        mapper.write_register(1, 5)
        assert mapper.slot_banks[0] == 1

    def test_reset(self):
        cart = _make_cart(64)
        mapper = SegaMapper(cart)
        mapper.write_register(1, 3)
        mapper.write_register(2, 2)
        mapper.write_register(3, 1)
        mapper.reset()
        assert mapper.slot_banks == (0, 1, 2)


# ---------------------------------------------------------------------------
# MemoryBus integration tests

class TestMemoryBus:
    def test_rom_read_slot0(self):
        cart = _make_cart(64)
        bus = MemoryBus(cart)
        # Bank 0 fills $0400–$3FFF, value = 0x00
        assert bus.read(0x0400) == 0x00

    def test_rom_read_slot1(self):
        cart = _make_cart(64)
        bus = MemoryBus(cart)
        # Bank 1 maps to $4000–$7FFF, value = 0x01
        assert bus.read(0x4000) == 0x01

    def test_rom_read_slot2(self):
        cart = _make_cart(64)
        bus = MemoryBus(cart)
        # Bank 2 maps to $8000–$BFFF, value = 0x02
        assert bus.read(0x8000) == 0x02

    def test_ram_read_write(self):
        cart = _make_cart(64)
        bus = MemoryBus(cart)
        bus.write(0xC000, 0x55)
        assert bus.read(0xC000) == 0x55

    def test_ram_mirror(self):
        cart = _make_cart(64)
        bus = MemoryBus(cart)
        bus.write(0xC100, 0x33)
        assert bus.read(0xE100) == 0x33  # $E000 mirrors $C000

    def test_rom_write_ignored(self):
        cart = _make_cart(64)
        bus = MemoryBus(cart)
        original = bus.read(0x1000)
        bus.write(0x1000, original ^ 0xFF)
        assert bus.read(0x1000) == original  # write to ROM is a no-op

    def test_mapper_register_write_via_bus(self):
        cart = _make_cart(64)
        bus = MemoryBus(cart)
        # Writing to $FFFF (RAM mirror) also updates slot 2 mapping
        bus.write(0xFFFF, 3)
        assert bus.mapper.slot_banks[2] == 3
        # Reading slot 2 should now reflect bank 3 (value = 0x03)
        assert bus.read(0x8000) == 0x03

    def test_mapper_slot0_via_fffd(self):
        cart = _make_cart(64)
        bus = MemoryBus(cart)
        bus.write(0xFFFD, 3)
        assert bus.read(0x0400) == 0x03  # now reads bank 3

    def test_fixed_1kb_immutable(self):
        cart = _make_cart(64)
        bus = MemoryBus(cart)
        # Even after remapping slot 0, $0000–$03FF is always bank 0
        bus.write(0xFFFD, 3)
        assert bus.read(0x0000) == 0xAA
        assert bus.read(0x0200) == 0xBB
