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


# ---------------------------------------------------------------------------
# Task 2 — Load instructions
# ---------------------------------------------------------------------------

class TestLoad:

    # -----------------------------------------------------------------------
    # LD r, n  — immediate byte into register
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("opcode,reg_attr,val", [
        (0x06, 'B', 0x11),
        (0x0E, 'C', 0x22),
        (0x16, 'D', 0x33),
        (0x1E, 'E', 0x44),
        (0x26, 'H', 0x55),
        (0x2E, 'L', 0x66),
        (0x3E, 'A', 0x77),
    ])
    def test_ld_r_n(self, opcode, reg_attr, val):
        cpu = make_cpu()
        load_prog(cpu, [opcode, val])
        cycles = cpu.step()
        assert getattr(cpu, reg_attr) == val
        assert cycles == 7

    # -----------------------------------------------------------------------
    # LD r, r'  — register-to-register transfers (0x40–0x7F, not HALT)
    # -----------------------------------------------------------------------

    def test_ld_b_c(self):
        cpu = make_cpu()
        cpu.C = 0xAB
        load_prog(cpu, [0x41])   # LD B, C
        cycles = cpu.step()
        assert cpu.B == 0xAB
        assert cycles == 4

    def test_ld_c_d(self):
        cpu = make_cpu()
        cpu.D = 0x12
        load_prog(cpu, [0x4A])   # LD C, D
        cpu.step()
        assert cpu.C == 0x12

    def test_ld_d_e(self):
        cpu = make_cpu()
        cpu.E = 0x34
        load_prog(cpu, [0x53])   # LD D, E
        cpu.step()
        assert cpu.D == 0x34

    def test_ld_e_h(self):
        cpu = make_cpu()
        cpu.H = 0x56
        load_prog(cpu, [0x5C])   # LD E, H
        cpu.step()
        assert cpu.E == 0x56

    def test_ld_h_l(self):
        cpu = make_cpu()
        cpu.L = 0x78
        load_prog(cpu, [0x65])   # LD H, L
        cpu.step()
        assert cpu.H == 0x78

    def test_ld_l_a(self):
        cpu = make_cpu()
        cpu.A = 0x9A
        load_prog(cpu, [0x6F])   # LD L, A
        cpu.step()
        assert cpu.L == 0x9A

    def test_ld_a_b(self):
        cpu = make_cpu()
        cpu.B = 0xBC
        load_prog(cpu, [0x78])   # LD A, B
        cpu.step()
        assert cpu.A == 0xBC

    def test_ld_r_r_self(self):
        # LD B, B — value unchanged, cycles = 4
        cpu = make_cpu()
        cpu.B = 0x55
        load_prog(cpu, [0x40])
        cycles = cpu.step()
        assert cpu.B == 0x55
        assert cycles == 4

    def test_ld_r_r_does_not_touch_flags(self):
        cpu = make_cpu()
        cpu.F = 0xFF
        cpu.B = 0x01; cpu.C = 0x02
        load_prog(cpu, [0x41])   # LD B, C
        cpu.step()
        assert cpu.F == 0xFF    # flags untouched

    # -----------------------------------------------------------------------
    # LD r, (HL)  — load register from memory at HL
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("opcode,reg_attr", [
        (0x46, 'B'),
        (0x4E, 'C'),
        (0x56, 'D'),
        (0x5E, 'E'),
        (0x66, 'H'),  # HL itself — value becomes (old HL)
        (0x6E, 'L'),
        (0x7E, 'A'),
    ])
    def test_ld_r_hl_mem(self, opcode, reg_attr):
        cpu = make_cpu()
        cpu.HL = 0xC000
        cpu.bus.mem[0xC000] = 0x42
        load_prog(cpu, [opcode])
        cycles = cpu.step()
        assert getattr(cpu, reg_attr) == 0x42
        assert cycles == 7

    # -----------------------------------------------------------------------
    # LD (HL), r  — store register to memory at HL
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("opcode,reg_attr,val", [
        (0x70, 'B', 0x11),
        (0x71, 'C', 0x22),
        (0x72, 'D', 0x33),
        (0x73, 'E', 0x44),
        (0x77, 'A', 0x66),
    ])
    def test_ld_hl_mem_r(self, opcode, reg_attr, val):
        cpu = make_cpu()
        cpu.HL = 0xC100
        setattr(cpu, reg_attr, val)
        load_prog(cpu, [opcode])
        cycles = cpu.step()
        assert cpu.bus.mem[0xC100] == val
        assert cycles == 7

    def test_ld_hl_mem_h(self):
        # LD (HL), H — stores the high byte of HL itself
        cpu = make_cpu()
        cpu.HL = 0xC200
        load_prog(cpu, [0x74])
        cpu.step()
        assert cpu.bus.mem[0xC200] == 0xC2

    def test_ld_hl_mem_l(self):
        # LD (HL), L — setting L changes HL, so address and value must agree
        cpu = make_cpu()
        # Use an HL where changing L still points into RAM: H=0xC2, L=0x55
        cpu.H = 0xC2
        cpu.L = 0x55
        load_prog(cpu, [0x75])
        cpu.step()
        assert cpu.bus.mem[0xC255] == 0x55

    # -----------------------------------------------------------------------
    # LD (HL), n  — store immediate to memory at HL
    # -----------------------------------------------------------------------

    def test_ld_hl_n(self):
        cpu = make_cpu()
        cpu.HL = 0xC300
        load_prog(cpu, [0x36, 0xAB])
        cycles = cpu.step()
        assert cpu.bus.mem[0xC300] == 0xAB
        assert cycles == 10

    # -----------------------------------------------------------------------
    # LD A, (BC) / LD A, (DE)
    # -----------------------------------------------------------------------

    def test_ld_a_bc(self):
        cpu = make_cpu()
        cpu.BC = 0xC400
        cpu.bus.mem[0xC400] = 0x99
        load_prog(cpu, [0x0A])
        cycles = cpu.step()
        assert cpu.A == 0x99
        assert cycles == 7

    def test_ld_a_de(self):
        cpu = make_cpu()
        cpu.DE = 0xC500
        cpu.bus.mem[0xC500] = 0x88
        load_prog(cpu, [0x1A])
        cycles = cpu.step()
        assert cpu.A == 0x88
        assert cycles == 7

    # -----------------------------------------------------------------------
    # LD (BC), A / LD (DE), A
    # -----------------------------------------------------------------------

    def test_ld_bc_a(self):
        cpu = make_cpu()
        cpu.BC = 0xC600
        cpu.A  = 0x77
        load_prog(cpu, [0x02])
        cycles = cpu.step()
        assert cpu.bus.mem[0xC600] == 0x77
        assert cycles == 7

    def test_ld_de_a(self):
        cpu = make_cpu()
        cpu.DE = 0xC700
        cpu.A  = 0x66
        load_prog(cpu, [0x12])
        cycles = cpu.step()
        assert cpu.bus.mem[0xC700] == 0x66
        assert cycles == 7

    # -----------------------------------------------------------------------
    # LD A, (nn) / LD (nn), A
    # -----------------------------------------------------------------------

    def test_ld_a_nn(self):
        cpu = make_cpu()
        cpu.bus.mem[0xC800] = 0x55
        load_prog(cpu, [0x3A, 0x00, 0xC8])   # LD A, (0xC800)
        cycles = cpu.step()
        assert cpu.A == 0x55
        assert cycles == 13

    def test_ld_nn_a(self):
        cpu = make_cpu()
        cpu.A = 0x44
        load_prog(cpu, [0x32, 0x00, 0xC9])   # LD (0xC900), A
        cycles = cpu.step()
        assert cpu.bus.mem[0xC900] == 0x44
        assert cycles == 13

    # -----------------------------------------------------------------------
    # LD rr, nn  — 16-bit immediate loads
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("opcode,reg_attr,val", [
        (0x01, 'BC', 0x1234),
        (0x11, 'DE', 0x5678),
        (0x21, 'HL', 0x9ABC),
        (0x31, 'SP', 0xDEF0),
    ])
    def test_ld_rr_nn(self, opcode, reg_attr, val):
        cpu = make_cpu()
        lo = val & 0xFF
        hi = (val >> 8) & 0xFF
        load_prog(cpu, [opcode, lo, hi])
        cycles = cpu.step()
        assert getattr(cpu, reg_attr) == val
        assert cycles == 10

    # -----------------------------------------------------------------------
    # LD HL, (nn) / LD (nn), HL
    # -----------------------------------------------------------------------

    def test_ld_hl_from_addr(self):
        cpu = make_cpu()
        cpu.bus.mem[0xCA00] = 0xEF
        cpu.bus.mem[0xCA01] = 0xBE
        load_prog(cpu, [0x2A, 0x00, 0xCA])   # LD HL, (0xCA00)
        cycles = cpu.step()
        assert cpu.HL == 0xBEEF
        assert cycles == 16

    def test_ld_addr_hl(self):
        cpu = make_cpu()
        cpu.HL = 0x1234
        load_prog(cpu, [0x22, 0x00, 0xCB])   # LD (0xCB00), HL
        cycles = cpu.step()
        assert cpu.bus.mem[0xCB00] == 0x34   # lo first
        assert cpu.bus.mem[0xCB01] == 0x12
        assert cycles == 16

    # -----------------------------------------------------------------------
    # LD SP, HL
    # -----------------------------------------------------------------------

    def test_ld_sp_hl(self):
        cpu = make_cpu()
        cpu.HL = 0x8000
        load_prog(cpu, [0xF9])
        cycles = cpu.step()
        assert cpu.SP == 0x8000
        assert cycles == 6

    # -----------------------------------------------------------------------
    # HALT (0x76) — not a load, but lives in the LD r,r block
    # -----------------------------------------------------------------------

    def test_halt_sets_halted(self):
        cpu = make_cpu()
        load_prog(cpu, [0x76])
        cycles = cpu.step()
        assert cpu.halted is True
        assert cycles == 4

    def test_halt_spins_on_pc(self):
        cpu = make_cpu()
        load_prog(cpu, [0x76])
        cpu.step()           # sets halted, PC stays at 0x0000
        cycles = cpu.step()  # halted NOP
        assert cpu.halted is True
        assert cycles == 4
        assert cpu.PC == 0x0000
