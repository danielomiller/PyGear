"""Tests for save-state serialization across all hardware components."""

import os
import tempfile
import pytest

from pygear.cartridge import Cartridge
from pygear.console import GameGearConsole
from pygear.cpu.z80 import Z80
from pygear.vdp.vdp import VDP, CYCLES_PER_LINE, ACTIVE_LINES
from pygear.sound.psg import PSG
from pygear.memory.ram import RAM
from pygear.memory.mapper import SegaMapper, CodemastersMapper
from pygear.memory.bus import MemoryBus
from pygear.io.joypad import Joypad, UP, DOWN, BUTTON1


# ---------------------------------------------------------------------------
# Helpers

def _make_cart(size_kb: int = 64) -> Cartridge:
    size = size_kb * 1024
    data = bytearray(size)
    for bank in range(size // 0x4000):
        for offset in range(0x4000):
            data[bank * 0x4000 + offset] = bank & 0xFF
    tmp = tempfile.NamedTemporaryFile(suffix=".gg", delete=False)
    tmp.write(data)
    tmp.close()
    cart = Cartridge(tmp.name)
    os.unlink(tmp.name)
    return cart


def _make_console(size_kb: int = 64) -> GameGearConsole:
    return GameGearConsole(_make_cart(size_kb))


# ---------------------------------------------------------------------------
# Z80

class TestZ80State:

    def _make_z80(self):
        cart = _make_cart()
        bus = MemoryBus(cart)
        from pygear.io.ports import IOPorts
        from pygear.vdp.vdp import VDP
        from pygear.sound.psg import PSG
        from pygear.io.joypad import Joypad
        ports = IOPorts(VDP(), Joypad(), PSG())
        return Z80(bus, ports)

    def test_roundtrip_registers(self):
        cpu = self._make_z80()
        cpu.A = 0x12; cpu.F = 0x34; cpu.B = 0x56; cpu.C = 0x78
        cpu.D = 0x9A; cpu.E = 0xBC; cpu.H = 0xDE; cpu.L = 0xF0
        cpu.IX = 0x1234; cpu.IY = 0x5678; cpu.SP = 0xFFF0; cpu.PC = 0x1000
        cpu.I = 0x3F; cpu.R = 0x7E
        state = cpu.get_state()
        cpu2 = self._make_z80()
        cpu2.set_state(state)
        assert cpu2.A == 0x12 and cpu2.F == 0x34
        assert cpu2.IX == 0x1234 and cpu2.IY == 0x5678
        assert cpu2.SP == 0xFFF0 and cpu2.PC == 0x1000
        assert cpu2.I == 0x3F and cpu2.R == 0x7E

    def test_roundtrip_shadow_registers(self):
        cpu = self._make_z80()
        cpu.A_ = 0xAA; cpu.F_ = 0xBB; cpu.B_ = 0xCC; cpu.C_ = 0xDD
        state = cpu.get_state()
        cpu2 = self._make_z80()
        cpu2.set_state(state)
        assert cpu2.A_ == 0xAA and cpu2.F_ == 0xBB
        assert cpu2.B_ == 0xCC and cpu2.C_ == 0xDD

    def test_roundtrip_interrupt_flags(self):
        cpu = self._make_z80()
        cpu.IFF1 = True; cpu.IFF2 = False; cpu.IM = 2
        cpu.halted = True
        state = cpu.get_state()
        cpu2 = self._make_z80()
        cpu2.set_state(state)
        assert cpu2.IFF1 is True
        assert cpu2.IFF2 is False
        assert cpu2.IM == 2
        assert cpu2.halted is True

    def test_roundtrip_pending_flags(self):
        cpu = self._make_z80()
        cpu._int_pending = True
        cpu._nmi_pending = True
        cpu._ei_delay = True
        state = cpu.get_state()
        cpu2 = self._make_z80()
        cpu2.set_state(state)
        assert cpu2._int_pending is True
        assert cpu2._nmi_pending is True
        assert cpu2._ei_delay is True

    def test_roundtrip_cycles(self):
        cpu = self._make_z80()
        cpu.cycles = 123456
        state = cpu.get_state()
        cpu2 = self._make_z80()
        cpu2.set_state(state)
        assert cpu2.cycles == 123456


# ---------------------------------------------------------------------------
# VDP

class TestVDPState:

    def test_roundtrip_vram(self):
        vdp = VDP(); vdp.reset()
        vdp.vram[0x1234] = 0xAB
        state = vdp.get_state()
        vdp2 = VDP(); vdp2.reset()
        vdp2.set_state(state)
        assert vdp2.vram[0x1234] == 0xAB

    def test_roundtrip_cram(self):
        vdp = VDP(); vdp.reset()
        vdp.cram[0x1E] = 0x7F
        state = vdp.get_state()
        vdp2 = VDP(); vdp2.reset()
        vdp2.set_state(state)
        assert vdp2.cram[0x1E] == 0x7F

    def test_roundtrip_registers(self):
        vdp = VDP(); vdp.reset()
        vdp.regs[2] = 0xFF; vdp.regs[5] = 0x7E
        state = vdp.get_state()
        vdp2 = VDP(); vdp2.reset()
        vdp2.set_state(state)
        assert vdp2.regs[2] == 0xFF
        assert vdp2.regs[5] == 0x7E

    def test_roundtrip_internal_counters(self):
        vdp = VDP(); vdp.reset()
        vdp._line = 42; vdp._cycle = 100; vdp._line_irq = 7
        state = vdp.get_state()
        vdp2 = VDP(); vdp2.reset()
        vdp2.set_state(state)
        assert vdp2._line == 42
        assert vdp2._cycle == 100
        assert vdp2._line_irq == 7

    def test_roundtrip_address_latch(self):
        vdp = VDP(); vdp.reset()
        vdp._addr = 0x3ABC; vdp._latch = True; vdp._latch_lo = 0xBC
        state = vdp.get_state()
        vdp2 = VDP(); vdp2.reset()
        vdp2.set_state(state)
        assert vdp2._addr == 0x3ABC
        assert vdp2._latch is True
        assert vdp2._latch_lo == 0xBC

    def test_roundtrip_line_buffer(self):
        import numpy as np
        vdp = VDP(); vdp.reset()
        vdp.regs[1] = 0x40; vdp.regs[2] = 0xFF
        for _ in range(ACTIVE_LINES):
            vdp.step(CYCLES_PER_LINE)
        assert vdp._line_buffer[0] is not None
        state = vdp.get_state()
        vdp2 = VDP(); vdp2.reset()
        vdp2.set_state(state)
        assert vdp2._line_buffer[0] is not None
        assert (vdp2._line_buffer[0] == vdp._line_buffer[0]).all()

    def test_roundtrip_frame_ready(self):
        vdp = VDP(); vdp.reset()
        vdp.frame_ready = True
        state = vdp.get_state()
        vdp2 = VDP(); vdp2.reset()
        vdp2.set_state(state)
        assert vdp2.frame_ready is True


# ---------------------------------------------------------------------------
# PSG

class TestPSGState:

    def test_roundtrip_tone_periods(self):
        psg = PSG(); psg.reset()
        psg._tone_period = [100, 200, 300]
        state = psg.get_state()
        psg2 = PSG(); psg2.reset()
        psg2.set_state(state)
        assert psg2._tone_period == [100, 200, 300]

    def test_roundtrip_volume(self):
        psg = PSG(); psg.reset()
        psg._volume = [0, 5, 10, 15]
        state = psg.get_state()
        psg2 = PSG(); psg2.reset()
        psg2.set_state(state)
        assert psg2._volume == [0, 5, 10, 15]

    def test_roundtrip_stereo(self):
        psg = PSG(); psg.reset()
        psg._stereo = 0xA5
        state = psg.get_state()
        psg2 = PSG(); psg2.reset()
        psg2.set_state(state)
        assert psg2._stereo == 0xA5

    def test_roundtrip_lfsr(self):
        psg = PSG(); psg.reset()
        psg._lfsr = 0x1234
        state = psg.get_state()
        psg2 = PSG(); psg2.reset()
        psg2.set_state(state)
        assert psg2._lfsr == 0x1234


# ---------------------------------------------------------------------------
# RAM

class TestRAMState:

    def test_roundtrip(self):
        ram = RAM(); ram.reset()
        ram.write(0x0100, 0xDE)
        ram.write(0x1FFF, 0xAD)
        state = ram.get_state()
        ram2 = RAM(); ram2.reset()
        ram2.set_state(state)
        assert ram2.read(0x0100) == 0xDE
        assert ram2.read(0x1FFF) == 0xAD


# ---------------------------------------------------------------------------
# Mappers

class TestMapperState:

    def test_sega_roundtrip_slots(self):
        mapper = SegaMapper(_make_cart(64))
        mapper.write_register(1, 3); mapper.write_register(2, 2); mapper.write_register(3, 1)
        state = mapper.get_state()
        mapper2 = SegaMapper(_make_cart(64))
        mapper2.set_state(state)
        assert mapper2.slot_banks == (3, 2, 1)

    def test_sega_roundtrip_cart_ram(self):
        mapper = SegaMapper(_make_cart(64))
        mapper.write_register(0, 0x08)
        mapper.write_slot2(0x8010, 0xBE)
        state = mapper.get_state()
        mapper2 = SegaMapper(_make_cart(64))
        mapper2.set_state(state)
        mapper2.write_register(0, 0x08)
        assert mapper2.read_slot2(0x8010) == 0xBE

    def test_sega_roundtrip_dirty_flag(self):
        mapper = SegaMapper(_make_cart(64))
        mapper.write_register(0, 0x08)
        mapper.write_slot2(0x8000, 1)
        state = mapper.get_state()
        mapper2 = SegaMapper(_make_cart(64))
        mapper2.set_state(state)
        assert mapper2._cart_ram_dirty is True

    def test_codemasters_roundtrip_slots(self):
        cart = _make_cart(128)
        mapper = CodemastersMapper(cart)
        mapper.write_rom_area(0x0000, 5)
        mapper.write_rom_area(0x4000, 3)
        mapper.write_slot2(0x8000, 7)
        state = mapper.get_state()
        mapper2 = CodemastersMapper(cart)
        mapper2.set_state(state)
        assert mapper2.slot_banks == (5, 3, 7)


# ---------------------------------------------------------------------------
# Joypad

class TestJoypadState:

    def test_roundtrip_pressed_buttons(self):
        jp = Joypad()
        jp.press(UP); jp.press(BUTTON1)
        state = jp.get_state()
        jp2 = Joypad()
        jp2.set_state(state)
        assert UP in jp2._pressed
        assert BUTTON1 in jp2._pressed
        assert DOWN not in jp2._pressed

    def test_roundtrip_no_buttons(self):
        jp = Joypad()
        state = jp.get_state()
        jp2 = Joypad()
        jp2.press(DOWN)
        jp2.set_state(state)
        assert len(jp2._pressed) == 0


# ---------------------------------------------------------------------------
# Console (end-to-end)

class TestConsoleSaveState:

    def test_save_creates_file(self, tmp_path):
        console = _make_console()
        console._sav_path = str(tmp_path / "game.sav")
        path = console.save_state(slot=0)
        assert os.path.exists(path)

    def test_load_returns_false_when_no_file(self, tmp_path):
        console = _make_console()
        console._sav_path = str(tmp_path / "game.sav")
        assert console.load_state(slot=0) is False

    def test_load_returns_true_after_save(self, tmp_path):
        console = _make_console()
        console._sav_path = str(tmp_path / "game.sav")
        console.save_state(slot=0)
        assert console.load_state(slot=0) is True

    def test_state_path_uses_slot_number(self, tmp_path):
        console = _make_console()
        console._sav_path = str(tmp_path / "game.sav")
        p0 = console._state_path(0)
        p1 = console._state_path(1)
        assert p0 != p1
        assert "_s0.state" in p0
        assert "_s1.state" in p1

    def test_cpu_state_survives_roundtrip(self, tmp_path):
        console = _make_console()
        console._sav_path = str(tmp_path / "game.sav")
        console.cpu.PC = 0x1234
        console.cpu.A  = 0xAB
        console.save_state()
        console.cpu.PC = 0x0000
        console.cpu.A  = 0x00
        console.load_state()
        assert console.cpu.PC == 0x1234
        assert console.cpu.A  == 0xAB

    def test_vram_survives_roundtrip(self, tmp_path):
        console = _make_console()
        console._sav_path = str(tmp_path / "game.sav")
        console.vdp.vram[0x2000] = 0x55
        console.save_state()
        console.vdp.vram[0x2000] = 0x00
        console.load_state()
        assert console.vdp.vram[0x2000] == 0x55

    def test_ram_survives_roundtrip(self, tmp_path):
        console = _make_console()
        console._sav_path = str(tmp_path / "game.sav")
        console.bus.write(0xC100, 0x77)
        console.save_state()
        console.bus.write(0xC100, 0x00)
        console.load_state()
        assert console.bus.read(0xC100) == 0x77

    def test_psg_state_survives_roundtrip(self, tmp_path):
        console = _make_console()
        console._sav_path = str(tmp_path / "game.sav")
        console.psg._stereo = 0x3C
        console.psg._volume = [1, 2, 3, 4]
        console.save_state()
        console.psg._stereo = 0xFF
        console.psg._volume = [15, 15, 15, 15]
        console.load_state()
        assert console.psg._stereo == 0x3C
        assert console.psg._volume == [1, 2, 3, 4]

    def test_multiple_slots_are_independent(self, tmp_path):
        console = _make_console()
        console._sav_path = str(tmp_path / "game.sav")
        console.cpu.A = 0x11
        console.save_state(slot=0)
        console.cpu.A = 0x22
        console.save_state(slot=1)
        console.load_state(slot=0)
        assert console.cpu.A == 0x11
        console.load_state(slot=1)
        assert console.cpu.A == 0x22


# ---------------------------------------------------------------------------
# Pause button

class TestPauseButton:

    def test_trigger_pause_sets_nmi_pending(self):
        console = _make_console()
        assert console.cpu._nmi_pending is False
        console.trigger_pause()
        assert console.cpu._nmi_pending is True

    def test_nmi_consumed_and_jumps_to_0x0066(self):
        console = _make_console()
        console.cpu.PC  = 0x1000
        console.cpu.SP  = 0xFFFF
        console.cpu.IFF1 = True
        console.trigger_pause()
        console.cpu.step()
        assert console.cpu._nmi_pending is False
        assert console.cpu.PC == 0x0066

    def test_trigger_pause_survives_save_state(self, tmp_path):
        console = _make_console()
        console._sav_path = str(tmp_path / "game.sav")
        console.trigger_pause()
        console.save_state()
        console.cpu._nmi_pending = False
        console.load_state()
        assert console.cpu._nmi_pending is True
