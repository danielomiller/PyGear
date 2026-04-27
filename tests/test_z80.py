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


# ---------------------------------------------------------------------------
# Task 3 — 8-bit ALU and flags
# ---------------------------------------------------------------------------

class TestALU8:

    # -----------------------------------------------------------------------
    # ADD A, r  /  ADD A, n
    # -----------------------------------------------------------------------

    def test_add_basic(self):
        cpu = make_cpu()
        cpu.A = 0x10; cpu.B = 0x20
        load_prog(cpu, [0x80])          # ADD A, B
        cycles = cpu.step()
        assert cpu.A == 0x30
        assert not flag(cpu, 'S')
        assert not flag(cpu, 'Z')
        assert not flag(cpu, 'H')
        assert not flag(cpu, 'PV')
        assert not flag(cpu, 'N')
        assert not flag(cpu, 'C')
        assert cycles == 4

    def test_add_carry_and_zero(self):
        # 0xFF + 0x01 → 0x00  (C=1, Z=1, H=1)
        cpu = make_cpu()
        cpu.A = 0xFF; cpu.B = 0x01
        load_prog(cpu, [0x80])
        cpu.step()
        assert cpu.A == 0x00
        assert flag(cpu, 'Z')
        assert flag(cpu, 'C')
        assert flag(cpu, 'H')
        assert not flag(cpu, 'PV')
        assert not flag(cpu, 'N')

    def test_add_half_carry(self):
        # 0x0F + 0x01 → 0x10  (H=1 only)
        cpu = make_cpu()
        cpu.A = 0x0F; cpu.B = 0x01
        load_prog(cpu, [0x80])
        cpu.step()
        assert cpu.A == 0x10
        assert flag(cpu, 'H')
        assert not flag(cpu, 'C')

    def test_add_overflow_positive(self):
        # 0x7F + 0x01 → 0x80  (PV=1, S=1, H=1)
        cpu = make_cpu()
        cpu.A = 0x7F; cpu.B = 0x01
        load_prog(cpu, [0x80])
        cpu.step()
        assert cpu.A == 0x80
        assert flag(cpu, 'S')
        assert flag(cpu, 'PV')
        assert flag(cpu, 'H')
        assert not flag(cpu, 'C')

    def test_add_overflow_negative(self):
        # 0x80 + 0x80 → 0x00  (PV=1, Z=1, C=1)
        cpu = make_cpu()
        cpu.A = 0x80; cpu.B = 0x80
        load_prog(cpu, [0x80])
        cpu.step()
        assert cpu.A == 0x00
        assert flag(cpu, 'Z')
        assert flag(cpu, 'C')
        assert flag(cpu, 'PV')
        assert not flag(cpu, 'S')

    def test_add_clears_n_flag(self):
        cpu = make_cpu()
        cpu.A = 0x01; cpu.B = 0x01
        set_flags(cpu, N=True)
        load_prog(cpu, [0x80])
        cpu.step()
        assert not flag(cpu, 'N')

    def test_add_n_immediate(self):
        cpu = make_cpu()
        cpu.A = 0x05
        load_prog(cpu, [0xC6, 0x03])    # ADD A, 3
        cycles = cpu.step()
        assert cpu.A == 0x08
        assert cycles == 7

    def test_add_hl_mem(self):
        # ADD A, (HL) — 7 cycles, reads from memory
        cpu = make_cpu()
        cpu.A = 0x10; cpu.HL = 0xC000
        cpu.bus.mem[0xC000] = 0x05
        load_prog(cpu, [0x86])
        cycles = cpu.step()
        assert cpu.A == 0x15
        assert cycles == 7

    # -----------------------------------------------------------------------
    # ADC A, r  /  ADC A, n
    # -----------------------------------------------------------------------

    def test_adc_without_carry(self):
        cpu = make_cpu()
        cpu.A = 0x10; cpu.B = 0x05
        set_flags(cpu, C=False)
        load_prog(cpu, [0x88])          # ADC A, B
        cpu.step()
        assert cpu.A == 0x15

    def test_adc_with_carry(self):
        cpu = make_cpu()
        cpu.A = 0x10; cpu.B = 0x05
        set_flags(cpu, C=True)
        load_prog(cpu, [0x88])
        cpu.step()
        assert cpu.A == 0x16

    def test_adc_carry_into_overflow(self):
        # 0x7E + 0x01 + C=1 → 0x80  (PV=1)
        cpu = make_cpu()
        cpu.A = 0x7E; cpu.B = 0x01
        set_flags(cpu, C=True)
        load_prog(cpu, [0x88])
        cpu.step()
        assert cpu.A == 0x80
        assert flag(cpu, 'PV')
        assert flag(cpu, 'S')

    def test_adc_n_immediate(self):
        cpu = make_cpu()
        cpu.A = 0x01
        set_flags(cpu, C=True)
        load_prog(cpu, [0xCE, 0x02])    # ADC A, 2
        cycles = cpu.step()
        assert cpu.A == 0x04
        assert cycles == 7

    # -----------------------------------------------------------------------
    # SUB r  /  SUB n
    # -----------------------------------------------------------------------

    def test_sub_basic(self):
        cpu = make_cpu()
        cpu.A = 0x20; cpu.B = 0x10
        load_prog(cpu, [0x90])          # SUB B
        cycles = cpu.step()
        assert cpu.A == 0x10
        assert flag(cpu, 'N')
        assert not flag(cpu, 'C')
        assert not flag(cpu, 'Z')
        assert cycles == 4

    def test_sub_to_zero(self):
        cpu = make_cpu()
        cpu.A = 0x05; cpu.B = 0x05
        load_prog(cpu, [0x90])
        cpu.step()
        assert cpu.A == 0x00
        assert flag(cpu, 'Z')
        assert flag(cpu, 'N')
        assert not flag(cpu, 'C')

    def test_sub_borrow(self):
        # 0x00 - 0x01 → 0xFF  (C=1, S=1, H=1)
        cpu = make_cpu()
        cpu.A = 0x00; cpu.B = 0x01
        load_prog(cpu, [0x90])
        cpu.step()
        assert cpu.A == 0xFF
        assert flag(cpu, 'C')
        assert flag(cpu, 'S')
        assert flag(cpu, 'H')

    def test_sub_overflow(self):
        # 0x80 - 0x01 → 0x7F  (PV=1: negative − positive = positive)
        cpu = make_cpu()
        cpu.A = 0x80; cpu.B = 0x01
        load_prog(cpu, [0x90])
        cpu.step()
        assert cpu.A == 0x7F
        assert flag(cpu, 'PV')
        assert not flag(cpu, 'C')

    def test_sub_n_immediate(self):
        cpu = make_cpu()
        cpu.A = 0x10
        load_prog(cpu, [0xD6, 0x05])    # SUB 5
        cycles = cpu.step()
        assert cpu.A == 0x0B
        assert cycles == 7

    # -----------------------------------------------------------------------
    # SBC A, r  /  SBC A, n
    # -----------------------------------------------------------------------

    def test_sbc_without_carry(self):
        cpu = make_cpu()
        cpu.A = 0x10; cpu.B = 0x05
        set_flags(cpu, C=False)
        load_prog(cpu, [0x98])          # SBC A, B
        cpu.step()
        assert cpu.A == 0x0B
        assert flag(cpu, 'N')

    def test_sbc_with_carry(self):
        cpu = make_cpu()
        cpu.A = 0x10; cpu.B = 0x05
        set_flags(cpu, C=True)
        load_prog(cpu, [0x98])
        cpu.step()
        assert cpu.A == 0x0A

    def test_sbc_borrow_chain(self):
        # 0x00 - 0x00 - C=1 → 0xFF  (C=1)
        cpu = make_cpu()
        cpu.A = 0x00; cpu.B = 0x00
        set_flags(cpu, C=True)
        load_prog(cpu, [0x98])
        cpu.step()
        assert cpu.A == 0xFF
        assert flag(cpu, 'C')

    def test_sbc_n_immediate(self):
        cpu = make_cpu()
        cpu.A = 0x10
        set_flags(cpu, C=True)
        load_prog(cpu, [0xDE, 0x05])    # SBC A, 5
        cycles = cpu.step()
        assert cpu.A == 0x0A
        assert cycles == 7

    # -----------------------------------------------------------------------
    # AND r  /  AND n
    # -----------------------------------------------------------------------

    def test_and_basic(self):
        cpu = make_cpu()
        cpu.A = 0xF0; cpu.B = 0x0F
        load_prog(cpu, [0xA0])          # AND B
        cycles = cpu.step()
        assert cpu.A == 0x00
        assert flag(cpu, 'Z')
        assert flag(cpu, 'H')
        assert not flag(cpu, 'N')
        assert not flag(cpu, 'C')
        assert cycles == 4

    def test_and_sets_parity(self):
        # 0xFF & 0x03 = 0x03 — two bits set → even parity → PV=1
        cpu = make_cpu()
        cpu.A = 0xFF; cpu.B = 0x03
        load_prog(cpu, [0xA0])
        cpu.step()
        assert cpu.A == 0x03
        assert flag(cpu, 'PV')

    def test_and_clears_carry(self):
        cpu = make_cpu()
        cpu.A = 0xFF; cpu.B = 0xFF
        set_flags(cpu, C=True)
        load_prog(cpu, [0xA0])
        cpu.step()
        assert not flag(cpu, 'C')

    def test_and_n_immediate(self):
        cpu = make_cpu()
        cpu.A = 0xAA
        load_prog(cpu, [0xE6, 0x55])    # AND 0x55
        cycles = cpu.step()
        assert cpu.A == 0x00
        assert flag(cpu, 'Z')
        assert cycles == 7

    # -----------------------------------------------------------------------
    # XOR r  /  XOR n
    # -----------------------------------------------------------------------

    def test_xor_self_zero(self):
        # XOR A — always yields 0 with Z=1, all others clear
        cpu = make_cpu()
        cpu.A = 0x42
        set_flags(cpu, C=True, N=True, H=True)
        load_prog(cpu, [0xAF])          # XOR A
        cycles = cpu.step()
        assert cpu.A == 0x00
        assert flag(cpu, 'Z')
        assert not flag(cpu, 'H')
        assert not flag(cpu, 'N')
        assert not flag(cpu, 'C')
        assert flag(cpu, 'PV')          # zero has even parity
        assert cycles == 4

    def test_xor_basic(self):
        cpu = make_cpu()
        cpu.A = 0xFF; cpu.B = 0x0F
        load_prog(cpu, [0xA8])          # XOR B
        cpu.step()
        assert cpu.A == 0xF0
        assert flag(cpu, 'S')
        assert not flag(cpu, 'H')

    def test_xor_n_immediate(self):
        cpu = make_cpu()
        cpu.A = 0x55
        load_prog(cpu, [0xEE, 0xAA])    # XOR 0xAA
        cycles = cpu.step()
        assert cpu.A == 0xFF
        assert cycles == 7

    # -----------------------------------------------------------------------
    # OR r  /  OR n
    # -----------------------------------------------------------------------

    def test_or_basic(self):
        cpu = make_cpu()
        cpu.A = 0xF0; cpu.B = 0x0F
        load_prog(cpu, [0xB0])          # OR B
        cycles = cpu.step()
        assert cpu.A == 0xFF
        assert flag(cpu, 'S')
        assert not flag(cpu, 'Z')
        assert not flag(cpu, 'H')
        assert not flag(cpu, 'N')
        assert not flag(cpu, 'C')
        assert cycles == 4

    def test_or_zero(self):
        cpu = make_cpu()
        cpu.A = 0x00; cpu.B = 0x00
        load_prog(cpu, [0xB0])
        cpu.step()
        assert cpu.A == 0x00
        assert flag(cpu, 'Z')
        assert flag(cpu, 'PV')          # zero parity

    def test_or_n_immediate(self):
        cpu = make_cpu()
        cpu.A = 0x0F
        load_prog(cpu, [0xF6, 0xF0])    # OR 0xF0
        cycles = cpu.step()
        assert cpu.A == 0xFF
        assert cycles == 7

    # -----------------------------------------------------------------------
    # CP r  /  CP n
    # -----------------------------------------------------------------------

    def test_cp_equal(self):
        # CP where A == operand → Z=1, A unchanged
        cpu = make_cpu()
        cpu.A = 0x42; cpu.B = 0x42
        load_prog(cpu, [0xB8])          # CP B
        cycles = cpu.step()
        assert cpu.A == 0x42            # A must not change
        assert flag(cpu, 'Z')
        assert flag(cpu, 'N')
        assert not flag(cpu, 'C')
        assert cycles == 4

    def test_cp_less_than(self):
        # A < operand → borrow, C=1
        cpu = make_cpu()
        cpu.A = 0x01; cpu.B = 0x02
        load_prog(cpu, [0xB8])
        cpu.step()
        assert cpu.A == 0x01            # A unchanged
        assert flag(cpu, 'C')
        assert not flag(cpu, 'Z')

    def test_cp_yx_from_operand(self):
        # Y/X undocumented flags come from the *operand*, not the result
        cpu = make_cpu()
        cpu.A = 0xFF
        cpu.B = 0b00101000              # bits 5 and 3 set
        load_prog(cpu, [0xB8])
        cpu.step()
        assert cpu.F & Y_FLAG           # bit 5 of operand
        assert cpu.F & X_FLAG           # bit 3 of operand

    def test_cp_n_immediate(self):
        cpu = make_cpu()
        cpu.A = 0x10
        load_prog(cpu, [0xFE, 0x10])    # CP 0x10
        cycles = cpu.step()
        assert cpu.A == 0x10
        assert flag(cpu, 'Z')
        assert cycles == 7

    # -----------------------------------------------------------------------
    # INC r
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("opcode,reg_attr", [
        (0x04, 'B'), (0x0C, 'C'), (0x14, 'D'), (0x1C, 'E'),
        (0x24, 'H'), (0x2C, 'L'), (0x3C, 'A'),
    ])
    def test_inc_r_basic(self, opcode, reg_attr):
        cpu = make_cpu()
        setattr(cpu, reg_attr, 0x41)
        load_prog(cpu, [opcode])
        cycles = cpu.step()
        assert getattr(cpu, reg_attr) == 0x42
        assert not flag(cpu, 'Z')
        assert not flag(cpu, 'S')
        assert not flag(cpu, 'PV')
        assert not flag(cpu, 'N')
        assert cycles == 4

    def test_inc_ff_wraps_to_zero(self):
        # INC B: 0xFF → 0x00, Z=1, H=1
        cpu = make_cpu()
        cpu.B = 0xFF
        set_flags(cpu, C=True)          # carry must be preserved
        load_prog(cpu, [0x04])
        cpu.step()
        assert cpu.B == 0x00
        assert flag(cpu, 'Z')
        assert flag(cpu, 'H')
        assert not flag(cpu, 'S')
        assert not flag(cpu, 'PV')
        assert flag(cpu, 'C')           # C unchanged by INC

    def test_inc_7f_overflow(self):
        # INC B: 0x7F → 0x80, PV=1, S=1, H=1; C is preserved (not changed)
        cpu = make_cpu()
        cpu.B = 0x7F
        set_flags(cpu, C=False)         # explicitly clear so we can assert it stays clear
        load_prog(cpu, [0x04])
        cpu.step()
        assert cpu.B == 0x80
        assert flag(cpu, 'PV')
        assert flag(cpu, 'S')
        assert flag(cpu, 'H')
        assert not flag(cpu, 'Z')
        assert not flag(cpu, 'C')       # C was False before → still False

    def test_inc_does_not_set_n(self):
        cpu = make_cpu()
        cpu.B = 0x00
        set_flags(cpu, N=True)
        load_prog(cpu, [0x04])
        cpu.step()
        assert not flag(cpu, 'N')

    def test_inc_hl_mem(self):
        # INC (HL): 11 cycles, writes back to memory
        cpu = make_cpu()
        cpu.HL = 0xC000
        cpu.bus.mem[0xC000] = 0x09
        load_prog(cpu, [0x34])
        cycles = cpu.step()
        assert cpu.bus.mem[0xC000] == 0x0A
        assert cycles == 11

    # -----------------------------------------------------------------------
    # DEC r
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("opcode,reg_attr", [
        (0x05, 'B'), (0x0D, 'C'), (0x15, 'D'), (0x1D, 'E'),
        (0x25, 'H'), (0x2D, 'L'), (0x3D, 'A'),
    ])
    def test_dec_r_basic(self, opcode, reg_attr):
        cpu = make_cpu()
        setattr(cpu, reg_attr, 0x42)
        load_prog(cpu, [opcode])
        cycles = cpu.step()
        assert getattr(cpu, reg_attr) == 0x41
        assert flag(cpu, 'N')
        assert not flag(cpu, 'Z')
        assert not flag(cpu, 'PV')
        assert cycles == 4

    def test_dec_to_zero(self):
        # DEC B: 0x01 → 0x00, Z=1, N=1
        cpu = make_cpu()
        cpu.B = 0x01
        set_flags(cpu, C=True)
        load_prog(cpu, [0x05])
        cpu.step()
        assert cpu.B == 0x00
        assert flag(cpu, 'Z')
        assert flag(cpu, 'N')
        assert flag(cpu, 'C')           # C unchanged

    def test_dec_80_overflow(self):
        # DEC B: 0x80 → 0x7F, PV=1, N=1, H=1
        cpu = make_cpu()
        cpu.B = 0x80
        load_prog(cpu, [0x05])
        cpu.step()
        assert cpu.B == 0x7F
        assert flag(cpu, 'PV')
        assert flag(cpu, 'N')
        assert flag(cpu, 'H')
        assert not flag(cpu, 'S')

    def test_dec_00_wraps(self):
        # DEC B: 0x00 → 0xFF, S=1, N=1, H=1
        cpu = make_cpu()
        cpu.B = 0x00
        load_prog(cpu, [0x05])
        cpu.step()
        assert cpu.B == 0xFF
        assert flag(cpu, 'S')
        assert flag(cpu, 'H')
        assert not flag(cpu, 'Z')

    def test_dec_preserves_carry(self):
        cpu = make_cpu()
        cpu.B = 0x10
        set_flags(cpu, C=True)
        load_prog(cpu, [0x05])
        cpu.step()
        assert flag(cpu, 'C')

    def test_dec_hl_mem(self):
        # DEC (HL): 11 cycles
        cpu = make_cpu()
        cpu.HL = 0xC000
        cpu.bus.mem[0xC000] = 0x0A
        load_prog(cpu, [0x35])
        cycles = cpu.step()
        assert cpu.bus.mem[0xC000] == 0x09
        assert cycles == 11

    # -----------------------------------------------------------------------
    # 16-bit INC / DEC rr  (no flags affected)
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("opcode,reg_attr,init,expected", [
        (0x03, 'BC', 0x00FF, 0x0100),
        (0x13, 'DE', 0xFFFF, 0x0000),
        (0x23, 'HL', 0x0000, 0x0001),
        (0x33, 'SP', 0x0100, 0x0101),
    ])
    def test_inc_rr(self, opcode, reg_attr, init, expected):
        cpu = make_cpu()
        setattr(cpu, reg_attr, init)
        cpu.F = 0xFF                    # all flags set — none should change
        load_prog(cpu, [opcode])
        cycles = cpu.step()
        assert getattr(cpu, reg_attr) == expected
        assert cpu.F == 0xFF            # flags untouched
        assert cycles == 6

    @pytest.mark.parametrize("opcode,reg_attr,init,expected", [
        (0x0B, 'BC', 0x0100, 0x00FF),
        (0x1B, 'DE', 0x0000, 0xFFFF),
        (0x2B, 'HL', 0x0001, 0x0000),
        (0x3B, 'SP', 0x0101, 0x0100),
    ])
    def test_dec_rr(self, opcode, reg_attr, init, expected):
        cpu = make_cpu()
        setattr(cpu, reg_attr, init)
        cpu.F = 0x00                    # all flags clear — none should change
        load_prog(cpu, [opcode])
        cycles = cpu.step()
        assert getattr(cpu, reg_attr) == expected
        assert cpu.F == 0x00
        assert cycles == 6

    # -----------------------------------------------------------------------
    # CPL  (complement A)
    # -----------------------------------------------------------------------

    def test_cpl(self):
        cpu = make_cpu()
        cpu.A = 0b10110100
        set_flags(cpu, S=True, Z=True, PV=True, C=True)
        load_prog(cpu, [0x2F])
        cycles = cpu.step()
        assert cpu.A == 0b01001011
        assert flag(cpu, 'H')
        assert flag(cpu, 'N')
        assert flag(cpu, 'S')           # S/Z/PV/C preserved
        assert flag(cpu, 'Z')
        assert flag(cpu, 'PV')
        assert flag(cpu, 'C')
        assert cycles == 4

    # -----------------------------------------------------------------------
    # NEG  (ED 44h)
    # -----------------------------------------------------------------------

    def test_neg_normal(self):
        # NEG: A = 0 - A
        cpu = make_cpu()
        cpu.A = 0x05
        load_prog(cpu, [0xED, 0x44])
        cycles = cpu.step()
        assert cpu.A == 0xFB
        assert flag(cpu, 'S')
        assert flag(cpu, 'C')           # result non-zero → borrow
        assert flag(cpu, 'N')
        assert not flag(cpu, 'Z')
        assert not flag(cpu, 'PV')
        assert cycles == 8

    def test_neg_zero(self):
        # NEG 0 → 0, Z=1, C=0
        cpu = make_cpu()
        cpu.A = 0x00
        load_prog(cpu, [0xED, 0x44])
        cpu.step()
        assert cpu.A == 0x00
        assert flag(cpu, 'Z')
        assert not flag(cpu, 'C')

    def test_neg_0x80_overflow(self):
        # NEG 0x80 → 0x80, PV=1 (only case)
        cpu = make_cpu()
        cpu.A = 0x80
        load_prog(cpu, [0xED, 0x44])
        cpu.step()
        assert cpu.A == 0x80
        assert flag(cpu, 'PV')
        assert flag(cpu, 'C')

    def test_neg_0x01(self):
        # NEG 0x01 → 0xFF, C=1, S=1
        cpu = make_cpu()
        cpu.A = 0x01
        load_prog(cpu, [0xED, 0x44])
        cpu.step()
        assert cpu.A == 0xFF
        assert flag(cpu, 'C')
        assert flag(cpu, 'S')


# ---------------------------------------------------------------------------
# Task 4 — Rotates, shifts, and CB prefix
# ---------------------------------------------------------------------------

class TestRotates:
    """RLCA / RRCA / RLA / RRA — accumulator rotates (4 cycles each)."""

    # -----------------------------------------------------------------------
    # RLCA  (0x07) — rotate A left, bit 7 → C and bit 0; S/Z/PV preserved
    # -----------------------------------------------------------------------

    def test_rlca_carry_out(self):
        cpu = make_cpu()
        cpu.A = 0b10110100          # bit 7 = 1
        load_prog(cpu, [0x07])
        cycles = cpu.step()
        assert cpu.A == 0b01101001
        assert flag(cpu, 'C')
        assert cycles == 4

    def test_rlca_no_carry(self):
        cpu = make_cpu()
        cpu.A = 0b01001000
        load_prog(cpu, [0x07])
        cpu.step()
        assert cpu.A == 0b10010000
        assert not flag(cpu, 'C')

    def test_rlca_preserves_szpv(self):
        cpu = make_cpu()
        cpu.A = 0x01
        set_flags(cpu, S=True, Z=True, PV=True)
        load_prog(cpu, [0x07])
        cpu.step()
        assert flag(cpu, 'S')
        assert flag(cpu, 'Z')
        assert flag(cpu, 'PV')

    def test_rlca_clears_h_and_n(self):
        cpu = make_cpu()
        cpu.A = 0x01
        set_flags(cpu, H=True, N=True)
        load_prog(cpu, [0x07])
        cpu.step()
        assert not flag(cpu, 'H')
        assert not flag(cpu, 'N')

    # -----------------------------------------------------------------------
    # RRCA  (0x0F) — rotate A right, bit 0 → C and bit 7; S/Z/PV preserved
    # -----------------------------------------------------------------------

    def test_rrca_carry_out(self):
        cpu = make_cpu()
        cpu.A = 0b10110101          # bit 0 = 1
        load_prog(cpu, [0x0F])
        cpu.step()
        assert cpu.A == 0b11011010
        assert flag(cpu, 'C')

    def test_rrca_no_carry(self):
        cpu = make_cpu()
        cpu.A = 0b10110100
        load_prog(cpu, [0x0F])
        cpu.step()
        assert cpu.A == 0b01011010
        assert not flag(cpu, 'C')

    def test_rrca_preserves_szpv(self):
        cpu = make_cpu()
        cpu.A = 0x02
        set_flags(cpu, S=True, Z=True, PV=True)
        load_prog(cpu, [0x0F])
        cpu.step()
        assert flag(cpu, 'S')
        assert flag(cpu, 'Z')
        assert flag(cpu, 'PV')

    # -----------------------------------------------------------------------
    # RLA  (0x17) — rotate A left through carry; S/Z/PV preserved
    # -----------------------------------------------------------------------

    def test_rla_bit7_into_carry(self):
        cpu = make_cpu()
        cpu.A = 0b10000000
        set_flags(cpu, C=False)
        load_prog(cpu, [0x17])
        cpu.step()
        assert cpu.A == 0b00000000
        assert flag(cpu, 'C')           # old bit 7 → C

    def test_rla_carry_into_bit0(self):
        cpu = make_cpu()
        cpu.A = 0b00000000
        set_flags(cpu, C=True)
        load_prog(cpu, [0x17])
        cpu.step()
        assert cpu.A == 0b00000001      # old C → bit 0
        assert not flag(cpu, 'C')

    def test_rla_both_set(self):
        cpu = make_cpu()
        cpu.A = 0b10110100
        set_flags(cpu, C=True)
        load_prog(cpu, [0x17])
        cpu.step()
        assert cpu.A == 0b01101001
        assert flag(cpu, 'C')

    def test_rla_preserves_szpv(self):
        cpu = make_cpu()
        cpu.A = 0x01
        set_flags(cpu, S=True, Z=True, PV=True, C=False)
        load_prog(cpu, [0x17])
        cpu.step()
        assert flag(cpu, 'S')
        assert flag(cpu, 'Z')
        assert flag(cpu, 'PV')

    # -----------------------------------------------------------------------
    # RRA  (0x1F) — rotate A right through carry; S/Z/PV preserved
    # -----------------------------------------------------------------------

    def test_rra_bit0_into_carry(self):
        cpu = make_cpu()
        cpu.A = 0b00000001
        set_flags(cpu, C=False)
        load_prog(cpu, [0x1F])
        cpu.step()
        assert cpu.A == 0b00000000
        assert flag(cpu, 'C')           # old bit 0 → C

    def test_rra_carry_into_bit7(self):
        cpu = make_cpu()
        cpu.A = 0b00000000
        set_flags(cpu, C=True)
        load_prog(cpu, [0x1F])
        cpu.step()
        assert cpu.A == 0b10000000      # old C → bit 7
        assert not flag(cpu, 'C')

    def test_rra_clears_h_and_n(self):
        cpu = make_cpu()
        cpu.A = 0x10
        set_flags(cpu, H=True, N=True, C=False)
        load_prog(cpu, [0x1F])
        cpu.step()
        assert not flag(cpu, 'H')
        assert not flag(cpu, 'N')


class TestCB:
    """CB-prefix instructions: rotate/shift, BIT, RES, SET."""

    # -----------------------------------------------------------------------
    # CB RLC  (0xCB 0x00–0x07) — sets S/Z/PV from result (unlike RLCA)
    # -----------------------------------------------------------------------

    def test_rlc_b_carry_and_wrap(self):
        cpu = make_cpu()
        cpu.B = 0b10000001
        load_prog(cpu, [0xCB, 0x00])    # RLC B
        cycles = cpu.step()
        assert cpu.B == 0b00000011
        assert flag(cpu, 'C')
        assert cycles == 8

    def test_rlc_sets_szpv_from_result(self):
        # Distinguish from RLCA: S/Z/PV reflect the new value
        cpu = make_cpu()
        cpu.B = 0b11000000              # result = 0b10000001 → S=1
        set_flags(cpu, S=False, Z=True, PV=False)
        load_prog(cpu, [0xCB, 0x00])
        cpu.step()
        assert flag(cpu, 'S')
        assert not flag(cpu, 'Z')

    def test_rlc_zero_result(self):
        cpu = make_cpu()
        cpu.B = 0x00
        load_prog(cpu, [0xCB, 0x00])
        cpu.step()
        assert cpu.B == 0x00
        assert flag(cpu, 'Z')
        assert not flag(cpu, 'C')

    def test_rlc_hl_mem(self):
        # RLC (HL): 15 cycles, writes back to memory
        cpu = make_cpu()
        cpu.HL = 0xC000
        cpu.bus.mem[0xC000] = 0b00010001
        load_prog(cpu, [0xCB, 0x06])    # RLC (HL)
        cycles = cpu.step()
        assert cpu.bus.mem[0xC000] == 0b00100010
        assert not flag(cpu, 'C')
        assert cycles == 15

    # -----------------------------------------------------------------------
    # CB RRC  (0xCB 0x08–0x0F)
    # -----------------------------------------------------------------------

    def test_rrc_a(self):
        cpu = make_cpu()
        cpu.A = 0b00000001
        load_prog(cpu, [0xCB, 0x0F])    # RRC A
        cpu.step()
        assert cpu.A == 0b10000000
        assert flag(cpu, 'C')
        assert flag(cpu, 'S')

    def test_rrc_no_carry(self):
        cpu = make_cpu()
        cpu.C = 0b10000000
        load_prog(cpu, [0xCB, 0x09])    # RRC C
        cpu.step()
        assert cpu.C == 0b01000000
        assert not flag(cpu, 'C')

    # -----------------------------------------------------------------------
    # CB RL  (0xCB 0x10–0x17) — rotate left through carry
    # -----------------------------------------------------------------------

    def test_rl_c_through_carry(self):
        cpu = make_cpu()
        cpu.C = 0b10000000
        set_flags(cpu, C=True)
        load_prog(cpu, [0xCB, 0x11])    # RL C
        cycles = cpu.step()
        assert cpu.C == 0b00000001      # old C in at bit 0
        assert flag(cpu, 'C')           # old bit 7 out
        assert cycles == 8

    def test_rl_clears_carry(self):
        cpu = make_cpu()
        cpu.D = 0b01000000
        set_flags(cpu, C=False)
        load_prog(cpu, [0xCB, 0x12])    # RL D
        cpu.step()
        assert cpu.D == 0b10000000
        assert not flag(cpu, 'C')

    # -----------------------------------------------------------------------
    # CB RR  (0xCB 0x18–0x1F) — rotate right through carry
    # -----------------------------------------------------------------------

    def test_rr_d(self):
        cpu = make_cpu()
        cpu.D = 0b00000001
        set_flags(cpu, C=False)
        load_prog(cpu, [0xCB, 0x1A])    # RR D
        cpu.step()
        assert cpu.D == 0b00000000
        assert flag(cpu, 'C')
        assert flag(cpu, 'Z')

    def test_rr_carry_in(self):
        cpu = make_cpu()
        cpu.E = 0b00000000
        set_flags(cpu, C=True)
        load_prog(cpu, [0xCB, 0x1B])    # RR E
        cpu.step()
        assert cpu.E == 0b10000000
        assert not flag(cpu, 'C')

    # -----------------------------------------------------------------------
    # CB SLA  (0xCB 0x20–0x27) — shift left, bit 0 ← 0
    # -----------------------------------------------------------------------

    def test_sla_e(self):
        cpu = make_cpu()
        cpu.E = 0b10110101
        load_prog(cpu, [0xCB, 0x23])    # SLA E
        cycles = cpu.step()
        assert cpu.E == 0b01101010      # bit 0 = 0
        assert flag(cpu, 'C')           # old bit 7
        assert cycles == 8

    def test_sla_zero_fill(self):
        # SLA always puts 0 in bit 0
        cpu = make_cpu()
        cpu.B = 0b11111111
        load_prog(cpu, [0xCB, 0x20])
        cpu.step()
        assert cpu.B == 0b11111110
        assert flag(cpu, 'C')

    # -----------------------------------------------------------------------
    # CB SRA  (0xCB 0x28–0x2F) — shift right arithmetic, bit 7 preserved
    # -----------------------------------------------------------------------

    def test_sra_preserves_sign(self):
        cpu = make_cpu()
        cpu.H = 0b10000001
        load_prog(cpu, [0xCB, 0x2C])    # SRA H
        cpu.step()
        assert cpu.H == 0b11000000      # bit 7 duplicated
        assert flag(cpu, 'C')           # old bit 0

    def test_sra_positive(self):
        cpu = make_cpu()
        cpu.A = 0b01000010
        load_prog(cpu, [0xCB, 0x2F])    # SRA A
        cpu.step()
        assert cpu.A == 0b00100001
        assert not flag(cpu, 'C')

    # -----------------------------------------------------------------------
    # CB SLL  (0xCB 0x30–0x37) — undocumented: shift left, bit 0 ← 1
    # -----------------------------------------------------------------------

    def test_sll_sets_bit0(self):
        cpu = make_cpu()
        cpu.L = 0b01000000
        load_prog(cpu, [0xCB, 0x35])    # SLL L
        cycles = cpu.step()
        assert cpu.L == 0b10000001      # bit 0 = 1
        assert not flag(cpu, 'C')
        assert cycles == 8

    def test_sll_carry_out(self):
        cpu = make_cpu()
        cpu.B = 0b10000000
        load_prog(cpu, [0xCB, 0x30])    # SLL B
        cpu.step()
        assert cpu.B == 0b00000001
        assert flag(cpu, 'C')

    # -----------------------------------------------------------------------
    # CB SRL  (0xCB 0x38–0x3F) — shift right logical, bit 7 ← 0
    # -----------------------------------------------------------------------

    def test_srl_a(self):
        cpu = make_cpu()
        cpu.A = 0b10000001
        load_prog(cpu, [0xCB, 0x3F])    # SRL A
        cpu.step()
        assert cpu.A == 0b01000000      # bit 7 = 0
        assert flag(cpu, 'C')           # old bit 0

    def test_srl_zero_fill(self):
        cpu = make_cpu()
        cpu.A = 0b10000000
        load_prog(cpu, [0xCB, 0x3F])
        cpu.step()
        assert cpu.A == 0b01000000
        assert not flag(cpu, 'C')
        assert not flag(cpu, 'S')       # logical: bit 7 always 0

    # -----------------------------------------------------------------------
    # CB BIT  (0xCB 0x40–0x7F) — test bit; Z/H/N affected, C preserved
    # -----------------------------------------------------------------------

    def test_bit_zero_sets_z(self):
        cpu = make_cpu()
        cpu.B = 0b11111110              # bit 0 = 0
        load_prog(cpu, [0xCB, 0x40])    # BIT 0, B
        cycles = cpu.step()
        assert flag(cpu, 'Z')
        assert flag(cpu, 'H')
        assert not flag(cpu, 'N')
        assert cycles == 8

    def test_bit_nonzero_clears_z(self):
        cpu = make_cpu()
        cpu.B = 0b00000001              # bit 0 = 1
        load_prog(cpu, [0xCB, 0x40])    # BIT 0, B
        cpu.step()
        assert not flag(cpu, 'Z')
        assert flag(cpu, 'H')

    def test_bit_7_sets_s_when_set(self):
        cpu = make_cpu()
        cpu.A = 0b10000000
        load_prog(cpu, [0xCB, 0x7F])    # BIT 7, A
        cpu.step()
        assert flag(cpu, 'S')
        assert not flag(cpu, 'Z')

    def test_bit_7_clears_s_when_clear(self):
        cpu = make_cpu()
        cpu.A = 0b01111111
        load_prog(cpu, [0xCB, 0x7F])    # BIT 7, A
        cpu.step()
        assert not flag(cpu, 'S')
        assert flag(cpu, 'Z')

    def test_bit_preserves_carry(self):
        cpu = make_cpu()
        cpu.B = 0xFF
        set_flags(cpu, C=True)
        load_prog(cpu, [0xCB, 0x40])    # BIT 0, B
        cpu.step()
        assert flag(cpu, 'C')

    def test_bit_does_not_change_register(self):
        cpu = make_cpu()
        cpu.B = 0b10101010
        load_prog(cpu, [0xCB, 0x40])
        cpu.step()
        assert cpu.B == 0b10101010      # BIT never modifies the register

    def test_bit_hl_mem(self):
        # BIT on (HL): 12 cycles
        cpu = make_cpu()
        cpu.HL = 0xC000
        cpu.bus.mem[0xC000] = 0b00000010
        load_prog(cpu, [0xCB, 0x46])    # BIT 0, (HL)
        cycles = cpu.step()
        assert flag(cpu, 'Z')           # bit 0 = 0
        assert cycles == 12

    def test_bit_hl_mem_nonzero(self):
        cpu = make_cpu()
        cpu.HL = 0xC000
        cpu.bus.mem[0xC000] = 0b00000001
        load_prog(cpu, [0xCB, 0x46])    # BIT 0, (HL)
        cpu.step()
        assert not flag(cpu, 'Z')

    # -----------------------------------------------------------------------
    # CB RES  (0xCB 0x80–0xBF) — clear a bit
    # -----------------------------------------------------------------------

    def test_res_clears_bit(self):
        cpu = make_cpu()
        cpu.B = 0xFF
        load_prog(cpu, [0xCB, 0x80])    # RES 0, B
        cycles = cpu.step()
        assert cpu.B == 0b11111110
        assert cycles == 8

    def test_res_bit3(self):
        cpu = make_cpu()
        cpu.A = 0xFF
        load_prog(cpu, [0xCB, 0xBF])    # RES 7, A
        cpu.step()
        assert cpu.A == 0b01111111

    def test_res_hl_mem(self):
        cpu = make_cpu()
        cpu.HL = 0xC000
        cpu.bus.mem[0xC000] = 0xFF
        load_prog(cpu, [0xCB, 0x86])    # RES 0, (HL)
        cycles = cpu.step()
        assert cpu.bus.mem[0xC000] == 0b11111110
        assert cycles == 15

    def test_res_idempotent(self):
        # Clearing an already-clear bit leaves value unchanged
        cpu = make_cpu()
        cpu.C = 0b11111110
        load_prog(cpu, [0xCB, 0x81])    # RES 0, C
        cpu.step()
        assert cpu.C == 0b11111110

    # -----------------------------------------------------------------------
    # CB SET  (0xCB 0xC0–0xFF) — set a bit
    # -----------------------------------------------------------------------

    def test_set_sets_bit(self):
        cpu = make_cpu()
        cpu.B = 0b00000000
        load_prog(cpu, [0xCB, 0xC0])    # SET 0, B
        cycles = cpu.step()
        assert cpu.B == 0b00000001
        assert cycles == 8

    def test_set_bit7(self):
        cpu = make_cpu()
        cpu.A = 0b00000000
        load_prog(cpu, [0xCB, 0xFF])    # SET 7, A
        cpu.step()
        assert cpu.A == 0b10000000

    def test_set_hl_mem(self):
        cpu = make_cpu()
        cpu.HL = 0xC000
        cpu.bus.mem[0xC000] = 0b00000000
        load_prog(cpu, [0xCB, 0xC6])    # SET 0, (HL)
        cycles = cpu.step()
        assert cpu.bus.mem[0xC000] == 0b00000001
        assert cycles == 15

    def test_set_idempotent(self):
        # Setting an already-set bit leaves value unchanged
        cpu = make_cpu()
        cpu.D = 0b11111111
        load_prog(cpu, [0xCB, 0xC2])    # SET 0, D
        cpu.step()
        assert cpu.D == 0b11111111

    def test_set_does_not_affect_flags(self):
        # SET and RES do not change flags
        cpu = make_cpu()
        cpu.B = 0x00
        cpu.F = 0b01000101              # some flag pattern
        load_prog(cpu, [0xCB, 0xC0])
        cpu.step()
        assert cpu.F == 0b01000101


# ---------------------------------------------------------------------------
# Task 5 — DAA (Decimal Adjust Accumulator)
# ---------------------------------------------------------------------------

class TestDAA:
    """
    DAA (0x27) corrects A after a BCD addition or subtraction.

    After addition (N=0):
      if H=1 or (A & 0xF) > 9 : A += 6   (low-nibble fix)
      if C=1 or A > 0x9F       : A += 0x60, C=1  (high-nibble fix)

    After subtraction (N=1):
      if H=1 : A -= 6
      if C=1 : A -= 0x60, C=1

    Flags after DAA: S Z computed from result; H computed; PV = parity;
    N preserved from before; C as above.
    """

    # shared helpers ---------------------------------------------------------

    def _add_then_daa(self, a: int, b: int):
        """ADD A, B then DAA. Returns cpu after both steps."""
        cpu = make_cpu()
        cpu.A = a
        cpu.B = b
        load_prog(cpu, [0x80, 0x27])    # ADD A,B ; DAA
        cpu.step()                       # ADD
        cpu.step()                       # DAA
        return cpu

    def _sub_then_daa(self, a: int, b: int):
        """SUB B then DAA. Returns cpu after both steps."""
        cpu = make_cpu()
        cpu.A = a
        cpu.B = b
        load_prog(cpu, [0x90, 0x27])    # SUB B ; DAA
        cpu.step()                       # SUB
        cpu.step()                       # DAA
        return cpu

    # -----------------------------------------------------------------------
    # After addition (N=0)
    # -----------------------------------------------------------------------

    def test_daa_add_no_adjust_needed(self):
        # BCD 04 + 03 = 07 — both nibbles valid, no correction
        cpu = self._add_then_daa(0x04, 0x03)
        assert cpu.A == 0x07
        assert not flag(cpu, 'C')
        assert not flag(cpu, 'Z')
        assert not flag(cpu, 'N')

    def test_daa_add_low_nibble_gt9(self):
        # BCD 07 + 05 = 12 — low nibble 0xC > 9, add 6 to fix
        cpu = self._add_then_daa(0x07, 0x05)
        assert cpu.A == 0x12
        assert not flag(cpu, 'C')

    def test_daa_add_half_carry(self):
        # BCD 08 + 08 = 16 — half-carry set after ADD, add 6 to fix
        cpu = self._add_then_daa(0x08, 0x08)
        assert cpu.A == 0x16
        assert not flag(cpu, 'C')

    def test_daa_add_upper_digit_overflow(self):
        # BCD 50 + 60 = 10 (carry) — upper result 0xB0 > 0x9F, add 0x60
        cpu = self._add_then_daa(0x50, 0x60)
        assert cpu.A == 0x10
        assert flag(cpu, 'C')

    def test_daa_add_both_nibbles_overflow(self):
        # BCD 99 + 01 = 00 (carry) — both corrections applied
        cpu = self._add_then_daa(0x99, 0x01)
        assert cpu.A == 0x00
        assert flag(cpu, 'C')
        assert flag(cpu, 'Z')

    def test_daa_add_99_plus_99(self):
        # BCD 99 + 99 = 98 (carry) — maximum BCD addition
        cpu = self._add_then_daa(0x99, 0x99)
        assert cpu.A == 0x98
        assert flag(cpu, 'C')

    def test_daa_add_clears_carry_when_no_overflow(self):
        # BCD 09 + 09 = 18 — H set, but no upper overflow; C must be clear
        cpu = self._add_then_daa(0x09, 0x09)
        assert cpu.A == 0x18
        assert not flag(cpu, 'C')

    def test_daa_add_preserves_existing_carry(self):
        # If C was set before DAA (multi-byte addition) it stays set
        cpu = make_cpu()
        cpu.A = 0x01
        cpu.B = 0x00
        load_prog(cpu, [0x80, 0x27])
        cpu.step()                       # ADD: C=0 after
        # manually force C=1 to simulate multi-word carry-in
        set_flags(cpu, C=True)
        cpu.step()                       # DAA with C forced on
        assert flag(cpu, 'C')            # C stays set: 0x01 with C-in → add 0x60

    def test_daa_add_zero_result_sets_z(self):
        cpu = self._add_then_daa(0x00, 0x00)
        assert cpu.A == 0x00
        assert flag(cpu, 'Z')
        assert not flag(cpu, 'S')

    def test_daa_add_sign_flag(self):
        # BCD result with bit 7 set — but BCD never actually uses > 0x99,
        # so test via forced A: after 0x50+0x50=0xA0 (before DAA, A=0xA0,
        # then +0x60 → 0x00 with C). Result 0 → S=0. Instead test 0x40+0x45:
        # 0x85, no correction → S=1
        cpu = self._add_then_daa(0x40, 0x45)
        assert cpu.A == 0x85
        assert flag(cpu, 'S')

    def test_daa_add_parity_flag(self):
        # PV = parity of result; 0x12 = 0b00010010 → 2 bits set → even → PV=1
        cpu = self._add_then_daa(0x07, 0x05)
        assert cpu.A == 0x12
        assert flag(cpu, 'PV')          # even parity

    def test_daa_add_n_cleared(self):
        # N=0 before DAA (after addition), stays 0 after
        cpu = self._add_then_daa(0x01, 0x01)
        assert not flag(cpu, 'N')

    def test_daa_add_cycles(self):
        cpu = make_cpu()
        cpu.A = 0x05; cpu.B = 0x05
        load_prog(cpu, [0x80, 0x27])
        cpu.step()
        cycles = cpu.step()             # DAA
        assert cycles == 4

    # -----------------------------------------------------------------------
    # After subtraction (N=1)
    # -----------------------------------------------------------------------

    def test_daa_sub_no_adjust_needed(self):
        # BCD 09 - 04 = 05 — no H, no C, result already valid
        cpu = self._sub_then_daa(0x09, 0x04)
        assert cpu.A == 0x05
        assert not flag(cpu, 'C')
        assert flag(cpu, 'N')

    def test_daa_sub_half_borrow(self):
        # BCD 10 - 05 = 05 — borrow from low nibble sets H; subtract 6
        cpu = self._sub_then_daa(0x10, 0x05)
        assert cpu.A == 0x05
        assert not flag(cpu, 'C')
        assert flag(cpu, 'N')

    def test_daa_sub_borrow_across_decades(self):
        # BCD 20 - 08 = 12 — H borrow, subtract 6
        cpu = self._sub_then_daa(0x20, 0x08)
        assert cpu.A == 0x12
        assert not flag(cpu, 'C')

    def test_daa_sub_with_carry_borrow(self):
        # BCD 00 - 01 = 99 (with borrow) — H and C both set after SUB
        cpu = self._sub_then_daa(0x00, 0x01)
        assert cpu.A == 0x99
        assert flag(cpu, 'C')
        assert flag(cpu, 'N')

    def test_daa_sub_50_minus_30(self):
        # BCD 50 - 30 = 20 — upper digits only, no nibble correction needed
        cpu = self._sub_then_daa(0x50, 0x30)
        assert cpu.A == 0x20
        assert not flag(cpu, 'C')

    def test_daa_sub_zero_result(self):
        # BCD 05 - 05 = 00
        cpu = self._sub_then_daa(0x05, 0x05)
        assert cpu.A == 0x00
        assert flag(cpu, 'Z')
        assert flag(cpu, 'N')
        assert not flag(cpu, 'C')

    def test_daa_sub_99_minus_99(self):
        # BCD 99 - 99 = 00
        cpu = self._sub_then_daa(0x99, 0x99)
        assert cpu.A == 0x00
        assert flag(cpu, 'Z')
        assert not flag(cpu, 'C')

    def test_daa_sub_preserves_n(self):
        # N=1 after SUB, must still be 1 after DAA
        cpu = self._sub_then_daa(0x09, 0x04)
        assert flag(cpu, 'N')

    def test_daa_sub_parity(self):
        # 0x05 = 0b00000101 → 2 bits → even parity → PV=1
        cpu = self._sub_then_daa(0x10, 0x05)
        assert cpu.A == 0x05
        assert flag(cpu, 'PV')


# ---------------------------------------------------------------------------
# Task 6 — Control flow, stack, exchange
# ---------------------------------------------------------------------------

class TestFlow:
    """JP / JR / CALL / RET / RST / DJNZ."""

    # -----------------------------------------------------------------------
    # JP nn  (unconditional)
    # -----------------------------------------------------------------------

    def test_jp_nn(self):
        cpu = make_cpu()
        load_prog(cpu, [0xC3, 0x00, 0x50])   # JP 0x5000
        cycles = cpu.step()
        assert cpu.PC == 0x5000
        assert cycles == 10

    def test_jp_nn_always_fetches_operand(self):
        # Even conditional JP that is not taken must advance PC by 3
        cpu = make_cpu()
        set_flags(cpu, Z=True)
        load_prog(cpu, [0xC2, 0x00, 0x50])   # JP NZ, 0x5000 — not taken
        cpu.step()
        assert cpu.PC == 0x0003              # past opcode + 2 operand bytes

    # -----------------------------------------------------------------------
    # JP cc — one taken + one not-taken test per condition pair
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("opcode,flag_name,flag_for_taken,dest", [
        (0xC2, 'Z',  False, 0x1000),   # JP NZ  — taken when Z=0
        (0xCA, 'Z',  True,  0x1001),   # JP Z   — taken when Z=1
        (0xD2, 'C',  False, 0x1002),   # JP NC  — taken when C=0
        (0xDA, 'C',  True,  0x1003),   # JP C   — taken when C=1
        (0xE2, 'PV', False, 0x1004),   # JP PO  — taken when PV=0
        (0xEA, 'PV', True,  0x1005),   # JP PE  — taken when PV=1
        (0xF2, 'S',  False, 0x1006),   # JP P   — taken when S=0
        (0xFA, 'S',  True,  0x1007),   # JP M   — taken when S=1
    ])
    def test_jp_cc_taken(self, opcode, flag_name, flag_for_taken, dest):
        cpu = make_cpu()
        set_flags(cpu, **{flag_name: flag_for_taken})
        lo, hi = dest & 0xFF, (dest >> 8) & 0xFF
        load_prog(cpu, [opcode, lo, hi])
        cpu.step()
        assert cpu.PC == dest

    @pytest.mark.parametrize("opcode,flag_name,flag_for_taken", [
        (0xC2, 'Z',  False),
        (0xCA, 'Z',  True),
        (0xD2, 'C',  False),
        (0xDA, 'C',  True),
        (0xE2, 'PV', False),
        (0xEA, 'PV', True),
        (0xF2, 'S',  False),
        (0xFA, 'S',  True),
    ])
    def test_jp_cc_not_taken(self, opcode, flag_name, flag_for_taken):
        cpu = make_cpu()
        set_flags(cpu, **{flag_name: not flag_for_taken})  # opposite → not taken
        load_prog(cpu, [opcode, 0x00, 0x50])
        cpu.step()
        assert cpu.PC == 0x0003    # past the 3-byte instruction

    # -----------------------------------------------------------------------
    # JP (HL)
    # -----------------------------------------------------------------------

    def test_jp_hl(self):
        cpu = make_cpu()
        cpu.HL = 0x4000
        load_prog(cpu, [0xE9])
        cycles = cpu.step()
        assert cpu.PC == 0x4000
        assert cycles == 4

    # -----------------------------------------------------------------------
    # JR e  (relative jump)
    # -----------------------------------------------------------------------

    def test_jr_forward(self):
        cpu = make_cpu()
        load_prog(cpu, [0x18, 0x05])   # JR +5 → PC = 2 + 5 = 7
        cycles = cpu.step()
        assert cpu.PC == 0x0007
        assert cycles == 12

    def test_jr_backward(self):
        cpu = make_cpu()
        # Place JR at 0x0010, offset -4 → 0x0010 + 2 - 4 = 0x000E
        load_prog(cpu, [0x18, 0xFC], origin=0x0010)   # 0xFC = -4 signed
        cpu.step()
        assert cpu.PC == 0x000E

    def test_jr_zero_offset(self):
        # JR 0 — jumps to next instruction (tight infinite loop)
        cpu = make_cpu()
        load_prog(cpu, [0x18, 0x00])
        cpu.step()
        assert cpu.PC == 0x0002

    @pytest.mark.parametrize("opcode,flag_name,flag_for_taken", [
        (0x20, 'Z', False),   # JR NZ
        (0x28, 'Z', True),    # JR Z
        (0x30, 'C', False),   # JR NC
        (0x38, 'C', True),    # JR C
    ])
    def test_jr_cc_taken(self, opcode, flag_name, flag_for_taken):
        cpu = make_cpu()
        set_flags(cpu, **{flag_name: flag_for_taken})
        load_prog(cpu, [opcode, 0x10])   # +16 → PC = 2 + 16 = 18
        cycles = cpu.step()
        assert cpu.PC == 0x0012
        assert cycles == 12

    @pytest.mark.parametrize("opcode,flag_name,flag_for_taken", [
        (0x20, 'Z', False),
        (0x28, 'Z', True),
        (0x30, 'C', False),
        (0x38, 'C', True),
    ])
    def test_jr_cc_not_taken(self, opcode, flag_name, flag_for_taken):
        cpu = make_cpu()
        set_flags(cpu, **{flag_name: not flag_for_taken})
        load_prog(cpu, [opcode, 0x10])
        cycles = cpu.step()
        assert cpu.PC == 0x0002   # past opcode + offset byte only
        assert cycles == 7

    # -----------------------------------------------------------------------
    # CALL nn  (unconditional)
    # -----------------------------------------------------------------------

    def test_call_nn(self):
        cpu = make_cpu()
        cpu.SP = 0xFFFF
        load_prog(cpu, [0xCD, 0x00, 0x40])   # CALL 0x4000
        cycles = cpu.step()
        assert cpu.PC == 0x4000
        assert cycles == 17
        # Return address (0x0003) on stack
        assert cpu._read16(cpu.SP) == 0x0003

    def test_call_pushes_next_pc(self):
        # Return address is the byte immediately after the CALL operands
        cpu = make_cpu()
        cpu.SP = 0xFFFF
        load_prog(cpu, [0xCD, 0x00, 0x80], origin=0x0100)
        cpu.step()
        assert cpu._read16(cpu.SP) == 0x0103

    # -----------------------------------------------------------------------
    # CALL cc
    # -----------------------------------------------------------------------

    def test_call_z_taken(self):
        cpu = make_cpu()
        cpu.SP = 0xFFFF
        set_flags(cpu, Z=True)
        load_prog(cpu, [0xCC, 0x00, 0x40])   # CALL Z, 0x4000
        cycles = cpu.step()
        assert cpu.PC == 0x4000
        assert cycles == 17

    def test_call_z_not_taken(self):
        cpu = make_cpu()
        cpu.SP = 0xFFFF
        set_flags(cpu, Z=False)
        sp_before = cpu.SP
        load_prog(cpu, [0xCC, 0x00, 0x40])
        cycles = cpu.step()
        assert cpu.PC == 0x0003    # skips the call
        assert cpu.SP == sp_before  # stack unchanged
        assert cycles == 10

    def test_call_nz_taken(self):
        cpu = make_cpu()
        cpu.SP = 0xFFFF
        set_flags(cpu, Z=False)
        load_prog(cpu, [0xC4, 0x00, 0x40])   # CALL NZ, 0x4000
        cycles = cpu.step()
        assert cpu.PC == 0x4000
        assert cycles == 17

    # -----------------------------------------------------------------------
    # RET  (unconditional)
    # -----------------------------------------------------------------------

    def test_ret_unconditional(self):
        cpu = make_cpu()
        cpu.SP = 0xFFFD
        cpu._write16(0xFFFD, 0x1234)
        load_prog(cpu, [0xC9])
        cycles = cpu.step()
        assert cpu.PC == 0x1234
        assert cpu.SP == 0xFFFF
        assert cycles == 10

    def test_call_ret_roundtrip(self):
        # CALL to subroutine, subroutine does RET, PC back at next instruction
        cpu = make_cpu()
        cpu.SP = 0xFFFF
        # 0x0000: CALL 0x0100
        # 0x0100: RET
        load_prog(cpu, [0xCD, 0x00, 0x01])   # CALL 0x0100
        cpu.bus.mem[0x0100] = 0xC9            # RET
        cpu.step()   # CALL
        cpu.step()   # RET
        assert cpu.PC == 0x0003
        assert cpu.SP == 0xFFFF

    # -----------------------------------------------------------------------
    # RET cc
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("opcode,flag_name,flag_for_taken", [
        (0xC0, 'Z',  False),   # RET NZ
        (0xC8, 'Z',  True),    # RET Z
        (0xD0, 'C',  False),   # RET NC
        (0xD8, 'C',  True),    # RET C
        (0xE0, 'PV', False),   # RET PO
        (0xE8, 'PV', True),    # RET PE
        (0xF0, 'S',  False),   # RET P
        (0xF8, 'S',  True),    # RET M
    ])
    def test_ret_cc_taken(self, opcode, flag_name, flag_for_taken):
        cpu = make_cpu()
        cpu.SP = 0xFFFD
        cpu._write16(0xFFFD, 0x5000)
        set_flags(cpu, **{flag_name: flag_for_taken})
        load_prog(cpu, [opcode])
        cycles = cpu.step()
        assert cpu.PC == 0x5000
        assert cpu.SP == 0xFFFF
        assert cycles == 11

    @pytest.mark.parametrize("opcode,flag_name,flag_for_taken", [
        (0xC0, 'Z',  False),
        (0xC8, 'Z',  True),
        (0xD0, 'C',  False),
        (0xD8, 'C',  True),
        (0xE0, 'PV', False),
        (0xE8, 'PV', True),
        (0xF0, 'S',  False),
        (0xF8, 'S',  True),
    ])
    def test_ret_cc_not_taken(self, opcode, flag_name, flag_for_taken):
        cpu = make_cpu()
        cpu.SP = 0xFFFD
        cpu.PC = 0x0000
        set_flags(cpu, **{flag_name: not flag_for_taken})
        load_prog(cpu, [opcode])
        cycles = cpu.step()
        assert cpu.PC == 0x0001    # past the 1-byte opcode only
        assert cpu.SP == 0xFFFD   # stack untouched
        assert cycles == 5

    # -----------------------------------------------------------------------
    # RST vectors
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("opcode,vector", [
        (0xC7, 0x0000), (0xCF, 0x0008), (0xD7, 0x0010), (0xDF, 0x0018),
        (0xE7, 0x0020), (0xEF, 0x0028), (0xF7, 0x0030), (0xFF, 0x0038),
    ])
    def test_rst_vectors(self, opcode, vector):
        cpu = make_cpu()
        cpu.SP = 0xFFFF
        load_prog(cpu, [opcode])
        cycles = cpu.step()
        assert cpu.PC == vector
        assert cycles == 11

    def test_rst_pushes_return_address(self):
        cpu = make_cpu()
        cpu.SP = 0xFFFF
        load_prog(cpu, [0xCF], origin=0x0200)   # RST 08H at 0x0200
        cpu.step()
        assert cpu._read16(cpu.SP) == 0x0201    # byte after RST

    # -----------------------------------------------------------------------
    # DJNZ
    # -----------------------------------------------------------------------

    def test_djnz_taken(self):
        cpu = make_cpu()
        cpu.B = 0x02
        load_prog(cpu, [0x10, 0xFE])   # DJNZ -2 → loops back to itself
        cycles = cpu.step()
        assert cpu.B == 0x01
        assert cpu.PC == 0x0000   # jumped back: 2 + (-2) = 0
        assert cycles == 13

    def test_djnz_falls_through(self):
        cpu = make_cpu()
        cpu.B = 0x01
        load_prog(cpu, [0x10, 0x10])   # DJNZ +16
        cycles = cpu.step()
        assert cpu.B == 0x00
        assert cpu.PC == 0x0002   # past opcode + offset, no jump
        assert cycles == 8

    def test_djnz_loop_count(self):
        # Run DJNZ until B reaches 0; verify correct number of iterations
        cpu = make_cpu()
        cpu.B = 5
        load_prog(cpu, [0x10, 0xFE])   # DJNZ to itself
        for _ in range(4):
            cpu.step()                  # taken 4 times
        cpu.step()                      # 5th: B→0, falls through
        assert cpu.B == 0x00
        assert cpu.PC == 0x0002

    # -----------------------------------------------------------------------
    # SCF / CCF
    # -----------------------------------------------------------------------

    def test_scf(self):
        cpu = make_cpu()
        set_flags(cpu, S=True, Z=True, PV=True, H=True, N=True, C=False)
        load_prog(cpu, [0x37])
        cycles = cpu.step()
        assert flag(cpu, 'C')
        assert not flag(cpu, 'H')
        assert not flag(cpu, 'N')
        assert flag(cpu, 'S')    # preserved
        assert flag(cpu, 'Z')    # preserved
        assert flag(cpu, 'PV')   # preserved
        assert cycles == 4

    def test_ccf_flips_carry_set_to_clear(self):
        cpu = make_cpu()
        set_flags(cpu, C=True, H=False)
        load_prog(cpu, [0x3F])
        cycles = cpu.step()
        assert not flag(cpu, 'C')
        assert flag(cpu, 'H')    # old C copied to H
        assert not flag(cpu, 'N')
        assert cycles == 4

    def test_ccf_flips_carry_clear_to_set(self):
        cpu = make_cpu()
        set_flags(cpu, C=False, H=True)
        load_prog(cpu, [0x3F])
        cpu.step()
        assert flag(cpu, 'C')
        assert not flag(cpu, 'H')   # old C (0) → H


class TestStack:
    """PUSH / POP for all four register pairs."""

    @pytest.mark.parametrize("push_op,pop_op,reg_attr,val", [
        (0xC5, 0xC1, 'BC', 0x1234),
        (0xD5, 0xD1, 'DE', 0x5678),
        (0xE5, 0xE1, 'HL', 0x9ABC),
        (0xF5, 0xF1, 'AF', 0xDEF0),
    ])
    def test_push_pop_roundtrip(self, push_op, pop_op, reg_attr, val):
        cpu = make_cpu()
        cpu.SP = 0xFFFF
        setattr(cpu, reg_attr, val)
        load_prog(cpu, [push_op, pop_op])
        cpu.step()   # PUSH
        cpu.step()   # POP
        assert getattr(cpu, reg_attr) == val
        assert cpu.SP == 0xFFFF

    @pytest.mark.parametrize("push_op,reg_attr,val", [
        (0xC5, 'BC', 0x1234),
        (0xD5, 'DE', 0x5678),
        (0xE5, 'HL', 0x9ABC),
        (0xF5, 'AF', 0xDEF0),
    ])
    def test_push_decrements_sp(self, push_op, reg_attr, val):
        cpu = make_cpu()
        cpu.SP = 0xFFFF
        setattr(cpu, reg_attr, val)
        load_prog(cpu, [push_op])
        cycles = cpu.step()
        assert cpu.SP == 0xFFFD
        assert cycles == 11

    @pytest.mark.parametrize("pop_op,reg_attr", [
        (0xC1, 'BC'), (0xD1, 'DE'), (0xE1, 'HL'), (0xF1, 'AF'),
    ])
    def test_pop_increments_sp(self, pop_op, reg_attr):
        cpu = make_cpu()
        cpu.SP = 0xFFFD
        cpu._write16(0xFFFD, 0xBEEF)
        load_prog(cpu, [pop_op])
        cycles = cpu.step()
        assert cpu.SP == 0xFFFF
        assert getattr(cpu, reg_attr) == 0xBEEF
        assert cycles == 10

    def test_push_stores_hi_then_lo(self):
        # PUSH stores high byte at SP-1 and low byte at SP-2
        cpu = make_cpu()
        cpu.SP = 0xFFFF
        cpu.BC = 0xABCD
        load_prog(cpu, [0xC5])
        cpu.step()
        assert cpu.bus.mem[0xFFFE] == 0xAB   # high byte
        assert cpu.bus.mem[0xFFFD] == 0xCD   # low byte

    def test_stack_lifo(self):
        # PUSH BC then PUSH DE; POP gives DE first, then BC
        cpu = make_cpu()
        cpu.SP = 0xFFFF
        cpu.BC = 0x1111
        cpu.DE = 0x2222
        load_prog(cpu, [0xC5, 0xD5, 0xD1, 0xC1])
        cpu.step()   # PUSH BC
        cpu.step()   # PUSH DE
        cpu.step()   # POP DE  (gets 0x2222 back)
        cpu.step()   # POP BC  (gets 0x1111 back)
        assert cpu.DE == 0x2222
        assert cpu.BC == 0x1111


class TestExchange:
    """EX AF,AF' / EXX / EX DE,HL / EX (SP),HL."""

    def test_ex_af_af_prime(self):
        cpu = make_cpu()
        cpu.A = 0x12; cpu.F = 0x34
        cpu.A_ = 0x56; cpu.F_ = 0x78
        load_prog(cpu, [0x08])
        cycles = cpu.step()
        assert cpu.A == 0x56 and cpu.F == 0x78
        assert cpu.A_ == 0x12 and cpu.F_ == 0x34
        assert cycles == 4

    def test_ex_af_af_prime_twice_restores(self):
        cpu = make_cpu()
        cpu.A = 0xAA; cpu.F = 0xBB
        load_prog(cpu, [0x08, 0x08])
        cpu.step(); cpu.step()
        assert cpu.A == 0xAA and cpu.F == 0xBB

    def test_exx(self):
        cpu = make_cpu()
        cpu.BC = 0x1111; cpu.DE = 0x2222; cpu.HL = 0x3333
        cpu.B_ = 0xAA; cpu.C_ = 0xBB
        cpu.D_ = 0xCC; cpu.E_ = 0xDD
        cpu.H_ = 0xEE; cpu.L_ = 0xFF
        load_prog(cpu, [0xD9])
        cycles = cpu.step()
        assert cpu.BC == 0xAABB
        assert cpu.DE == 0xCCDD
        assert cpu.HL == 0xEEFF
        assert cpu.B_ == 0x11 and cpu.C_ == 0x11
        assert cycles == 4

    def test_exx_twice_restores(self):
        cpu = make_cpu()
        cpu.BC = 0x1234
        load_prog(cpu, [0xD9, 0xD9])
        cpu.step(); cpu.step()
        assert cpu.BC == 0x1234

    def test_ex_de_hl(self):
        cpu = make_cpu()
        cpu.DE = 0xDEAD; cpu.HL = 0xBEEF
        load_prog(cpu, [0xEB])
        cycles = cpu.step()
        assert cpu.DE == 0xBEEF
        assert cpu.HL == 0xDEAD
        assert cycles == 4

    def test_ex_sp_hl(self):
        cpu = make_cpu()
        cpu.SP = 0xFFFD
        cpu._write16(0xFFFD, 0xABCD)
        cpu.HL = 0x1234
        load_prog(cpu, [0xE3])
        cycles = cpu.step()
        assert cpu.HL == 0xABCD          # HL gets old stack top
        assert cpu._read16(0xFFFD) == 0x1234  # stack gets old HL
        assert cpu.SP == 0xFFFD          # SP itself unchanged
        assert cycles == 19

    def test_ex_sp_hl_roundtrip(self):
        cpu = make_cpu()
        cpu.SP = 0xFFFD
        cpu._write16(0xFFFD, 0x5678)
        cpu.HL = 0x1234
        load_prog(cpu, [0xE3, 0xE3])
        cpu.step(); cpu.step()
        assert cpu.HL == 0x1234   # back to original
        assert cpu._read16(0xFFFD) == 0x5678
