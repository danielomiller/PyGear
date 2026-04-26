"""Z80 CPU test suite.

Structure
---------
Each test class is self-contained and uses the shared helpers defined below.
Tests verify register values, flag bits, memory side-effects, and T-state
counts returned by step().

Flag bit constants (Z80 F register)
-------------------------------------
S=0x80  Z=0x40  Y=0x20  H=0x10  X=0x08  PV=0x04  N=0x02  C=0x01
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pygear.cpu.z80 import Z80

# ---------------------------------------------------------------------------
# Flag bit constants — mirrors what opcodes.py uses internally
S_FLAG  = 0x80
Z_FLAG  = 0x40
Y_FLAG  = 0x20
H_FLAG  = 0x10
X_FLAG  = 0x08
PV_FLAG = 0x04
N_FLAG  = 0x02
C_FLAG  = 0x01


# ---------------------------------------------------------------------------
# Shared test infrastructure
# ---------------------------------------------------------------------------

class FakeBus:
    """Flat 64 KB RAM — no mapper, no ROM protection."""

    def __init__(self):
        self.mem = bytearray(0x10000)

    def read(self, addr: int) -> int:
        return self.mem[addr & 0xFFFF]

    def write(self, addr: int, value: int):
        self.mem[addr & 0xFFFF] = value & 0xFF


class FakePorts:
    """I/O port space: reads return a configurable value, writes are logged."""

    def __init__(self, default: int = 0xFF):
        self._default = default
        self._values: dict[int, int] = {}   # port → value for next read
        self.writes:  list[tuple[int, int]] = []  # (port, value) log

    def set_port(self, port: int, value: int):
        """Pre-load a value to be returned by the next read from *port*."""
        self._values[port] = value

    def read(self, port: int) -> int:
        return self._values.pop(port, self._default)

    def write(self, port: int, value: int):
        self.writes.append((port, value))


def make_cpu() -> Z80:
    """Return a freshly reset Z80 wired to a FakeBus and FakePorts."""
    cpu = Z80(FakeBus(), FakePorts())
    cpu.reset()
    return cpu


def load_prog(cpu: Z80, prog: bytes | list[int], origin: int = 0x0000):
    """Write *prog* into bus memory starting at *origin* and set PC there."""
    for i, byte in enumerate(prog):
        cpu.bus.mem[(origin + i) & 0xFFFF] = byte & 0xFF
    cpu.PC = origin


def set_flags(cpu: Z80, **flags):
    """Set/clear individual flag bits by keyword (S, Z, H, PV, N, C, Y, X)."""
    _map = {
        'S': S_FLAG, 'Z': Z_FLAG, 'Y': Y_FLAG, 'H': H_FLAG,
        'X': X_FLAG, 'PV': PV_FLAG, 'N': N_FLAG, 'C': C_FLAG,
    }
    f = cpu.F
    for name, val in flags.items():
        bit = _map[name]
        if val:
            f |= bit
        else:
            f &= ~bit
    cpu.F = f


def flag(cpu: Z80, name: str) -> bool:
    """Return the boolean state of a single named flag."""
    _map = {
        'S': S_FLAG, 'Z': Z_FLAG, 'Y': Y_FLAG, 'H': H_FLAG,
        'X': X_FLAG, 'PV': PV_FLAG, 'N': N_FLAG, 'C': C_FLAG,
    }
    return bool(cpu.F & _map[name])


# ---------------------------------------------------------------------------
# Smoke test — verifies the harness itself is wired correctly
# ---------------------------------------------------------------------------

class TestHarness:
    def test_make_cpu_returns_z80(self):
        cpu = make_cpu()
        assert isinstance(cpu, Z80)

    def test_reset_state(self):
        cpu = make_cpu()
        assert cpu.PC == 0x0000
        assert cpu.SP == 0xFFFF
        assert cpu.IFF1 is False
        assert cpu.IFF2 is False
        assert cpu.IM == 1
        assert cpu.halted is False

    def test_load_prog_sets_pc(self):
        cpu = make_cpu()
        load_prog(cpu, [0x00], origin=0x0100)
        assert cpu.PC == 0x0100

    def test_load_prog_writes_bytes(self):
        cpu = make_cpu()
        load_prog(cpu, [0xAA, 0xBB, 0xCC], origin=0x0200)
        assert cpu.bus.mem[0x0200] == 0xAA
        assert cpu.bus.mem[0x0201] == 0xBB
        assert cpu.bus.mem[0x0202] == 0xCC

    def test_set_flags_sets_bits(self):
        cpu = make_cpu()
        cpu.F = 0x00
        set_flags(cpu, S=True, C=True)
        assert cpu.F & S_FLAG
        assert cpu.F & C_FLAG
        assert not (cpu.F & Z_FLAG)

    def test_set_flags_clears_bits(self):
        cpu = make_cpu()
        cpu.F = 0xFF
        set_flags(cpu, Z=False, N=False)
        assert not (cpu.F & Z_FLAG)
        assert not (cpu.F & N_FLAG)
        assert cpu.F & S_FLAG   # untouched

    def test_flag_helper_reads_bit(self):
        cpu = make_cpu()
        cpu.F = Z_FLAG | C_FLAG
        assert flag(cpu, 'Z') is True
        assert flag(cpu, 'C') is True
        assert flag(cpu, 'S') is False

    def test_fake_bus_read_write(self):
        cpu = make_cpu()
        cpu.bus.write(0xC000, 0x42)
        assert cpu.bus.read(0xC000) == 0x42

    def test_fake_bus_wraps(self):
        cpu = make_cpu()
        cpu.bus.write(0x10000, 0x55)    # wraps to 0x0000
        assert cpu.bus.read(0x0000) == 0x55

    def test_fake_ports_read_default(self):
        cpu = make_cpu()
        assert cpu.ports.read(0x3F) == 0xFF

    def test_fake_ports_set_and_read(self):
        cpu = make_cpu()
        cpu.ports.set_port(0x7E, 0x42)
        assert cpu.ports.read(0x7E) == 0x42
        assert cpu.ports.read(0x7E) == 0xFF   # consumed

    def test_fake_ports_write_log(self):
        cpu = make_cpu()
        cpu.ports.write(0x01, 0xAB)
        cpu.ports.write(0x01, 0xCD)
        assert cpu.ports.writes == [(0x01, 0xAB), (0x01, 0xCD)]

    def test_nop_executes_and_advances_pc(self):
        cpu = make_cpu()
        load_prog(cpu, [0x00])   # NOP
        cycles = cpu.step()
        assert cpu.PC == 0x0001
        assert cycles == 4

    def test_step_accumulates_cycles(self):
        cpu = make_cpu()
        load_prog(cpu, [0x00, 0x00, 0x00])  # 3× NOP
        cpu.step(); cpu.step(); cpu.step()
        assert cpu.cycles == 12
