"""Main and ED opcode table builders for the PyGear Z80.

Usage
-----
    from .opcodes import build_all_tables
    main_table, ed_table = build_all_tables(cpu)

build_all_tables(cpu) returns a 256-entry list (main_table) and a dict
(ed_table) keyed by opcode byte.  Every entry is a zero-argument callable
that reads/modifies cpu state and returns the number of T-states consumed.

DD / FD prefix handling
-----------------------
Handlers that involve HL as a *memory address* (e.g. LD A,(HL)) check
cpu._dd / cpu._fd and use cpu._idx_addr (pre-loaded by _execute_indexed)
instead of cpu.HL.  They also return a reduced cycle count so that the
+4 added by _execute_indexed produces the correct total.

  (IX+d) cycles − 4 prefix cycles = value returned by handler
  e.g.  LD r,(IX+d) = 19 total → handler returns 15 in DD/FD mode
        INC (IX+d)  = 23 total → handler returns 19 in DD/FD mode

Handlers that use HL only as a *16-bit register* (LD HL,nn etc.) call
gHL()/sHL() which transparently redirect to IX/IY, and return their
normal cycle count; _execute_indexed adds 4.
"""

# ---------------------------------------------------------------------------
# Flag bit positions  (Z80 F register)
S_FLAG  = 0x80
Z_FLAG  = 0x40
Y_FLAG  = 0x20   # undocumented (bit 5 of result)
H_FLAG  = 0x10
X_FLAG  = 0x08   # undocumented (bit 3 of result)
PV_FLAG = 0x04
N_FLAG  = 0x02
C_FLAG  = 0x01


def _parity(v: int) -> bool:
    """True if v has even parity (even number of set bits)."""
    v &= 0xFF
    v ^= v >> 4
    v ^= v >> 2
    v ^= v >> 1
    return (v & 1) == 0


# ---------------------------------------------------------------------------
def build_all_tables(cpu):
    """Return (main_table[256], ed_table{opcode: handler})."""

    bus = cpu.bus

    # -----------------------------------------------------------------------
    # Register accessor helpers — DD/FD aware
    # Encoding: 0=B 1=C 2=D 3=E 4=H 5=L 6=(HL) 7=A
    # -----------------------------------------------------------------------
    def getr(i: int) -> int:
        if   i == 0: return cpu.B
        elif i == 1: return cpu.C
        elif i == 2: return cpu.D
        elif i == 3: return cpu.E
        elif i == 4:
            if cpu._dd: return (cpu.IX >> 8) & 0xFF   # IXH
            if cpu._fd: return (cpu.IY >> 8) & 0xFF   # IYH
            return cpu.H
        elif i == 5:
            if cpu._dd: return cpu.IX & 0xFF           # IXL
            if cpu._fd: return cpu.IY & 0xFF           # IYL
            return cpu.L
        elif i == 6:
            if cpu._dd or cpu._fd: return bus.read(cpu._idx_addr)
            return bus.read(cpu.HL)
        else: return cpu.A

    def setr(i: int, v: int):
        v &= 0xFF
        if   i == 0: cpu.B = v
        elif i == 1: cpu.C = v
        elif i == 2: cpu.D = v
        elif i == 3: cpu.E = v
        elif i == 4:
            if   cpu._dd: cpu.IX = (cpu.IX & 0x00FF) | (v << 8)
            elif cpu._fd: cpu.IY = (cpu.IY & 0x00FF) | (v << 8)
            else:         cpu.H  = v
        elif i == 5:
            if   cpu._dd: cpu.IX = (cpu.IX & 0xFF00) | v
            elif cpu._fd: cpu.IY = (cpu.IY & 0xFF00) | v
            else:         cpu.L  = v
        elif i == 6:
            if cpu._dd or cpu._fd: bus.write(cpu._idx_addr, v)
            else:                  bus.write(cpu.HL, v)
        else: cpu.A = v

    # HL as 16-bit register — redirects to IX/IY in DD/FD mode
    def gHL() -> int:
        if cpu._dd: return cpu.IX
        if cpu._fd: return cpu.IY
        return cpu.HL

    def sHL(v: int):
        v &= 0xFFFF
        if   cpu._dd: cpu.IX = v
        elif cpu._fd: cpu.IY = v
        else:         cpu.HL = v

    # -----------------------------------------------------------------------
    # Arithmetic / flag helpers (all update cpu.F)
    # -----------------------------------------------------------------------
    def do_add8(a: int, b: int, c: int = 0) -> int:
        """8-bit add: A = a + b + c.  Sets all flags.  Returns result byte."""
        full = a + b + c
        r    = full & 0xFF
        f    = r & (Y_FLAG | X_FLAG)
        if r & 0x80:     f |= S_FLAG
        if r == 0:       f |= Z_FLAG
        if full > 0xFF:  f |= C_FLAG
        if ((a & 0xF) + (b & 0xF) + c) > 0xF: f |= H_FLAG
        if (~(a ^ b) & (a ^ r) & 0x80):        f |= PV_FLAG
        cpu.F = f
        return r

    def do_sub8(a: int, b: int, c: int = 0) -> int:
        """8-bit sub: A = a − b − c.  Sets all flags (N=1).  Returns result."""
        full = a - b - c
        r    = full & 0xFF
        f    = N_FLAG | (r & (Y_FLAG | X_FLAG))
        if r & 0x80:    f |= S_FLAG
        if r == 0:      f |= Z_FLAG
        if full < 0:    f |= C_FLAG
        if ((a & 0xF) - (b & 0xF) - c) < 0: f |= H_FLAG
        if ((a ^ b) & (a ^ r) & 0x80):       f |= PV_FLAG
        cpu.F = f
        return r

    def do_add16(hl: int, rr: int) -> int:
        """16-bit add: only C, H, N changed; S/Z/PV preserved."""
        full = hl + rr
        r    = full & 0xFFFF
        f    = cpu.F & (S_FLAG | Z_FLAG | PV_FLAG)
        if full > 0xFFFF:                          f |= C_FLAG
        if ((hl & 0xFFF) + (rr & 0xFFF)) > 0xFFF: f |= H_FLAG
        f |= (r >> 8) & (Y_FLAG | X_FLAG)
        cpu.F = f
        return r

    def do_adc16(hl: int, rr: int) -> int:
        c    = 1 if (cpu.F & C_FLAG) else 0
        full = hl + rr + c
        r    = full & 0xFFFF
        f    = (r >> 8) & (Y_FLAG | X_FLAG)
        if r & 0x8000:   f |= S_FLAG
        if r == 0:       f |= Z_FLAG
        if full > 0xFFFF: f |= C_FLAG
        if ((hl & 0xFFF) + (rr & 0xFFF) + c) > 0xFFF: f |= H_FLAG
        if (~(hl ^ rr) & (hl ^ r) & 0x8000):           f |= PV_FLAG
        cpu.F = f
        return r

    def do_sbc16(hl: int, rr: int) -> int:
        c    = 1 if (cpu.F & C_FLAG) else 0
        full = hl - rr - c
        r    = full & 0xFFFF
        f    = N_FLAG | ((r >> 8) & (Y_FLAG | X_FLAG))
        if r & 0x8000:   f |= S_FLAG
        if r == 0:       f |= Z_FLAG
        if full < 0:     f |= C_FLAG
        if ((hl & 0xFFF) - (rr & 0xFFF) - c) < 0: f |= H_FLAG
        if ((hl ^ rr) & (hl ^ r) & 0x8000):        f |= PV_FLAG
        cpu.F = f
        return r

    def do_inc8(v: int) -> int:
        r = (v + 1) & 0xFF
        f = cpu.F & C_FLAG                 # C unchanged
        f |= r & (Y_FLAG | X_FLAG)
        if r & 0x80:          f |= S_FLAG
        if r == 0:            f |= Z_FLAG
        if (v & 0xF) == 0xF:  f |= H_FLAG  # carry from bit 3
        if r == 0x80:         f |= PV_FLAG  # overflow: 0x7F→0x80
        cpu.F = f
        return r

    def do_dec8(v: int) -> int:
        r = (v - 1) & 0xFF
        f = N_FLAG | (cpu.F & C_FLAG)      # N=1, C unchanged
        f |= r & (Y_FLAG | X_FLAG)
        if r & 0x80:          f |= S_FLAG
        if r == 0:            f |= Z_FLAG
        if (v & 0xF) == 0x0:  f |= H_FLAG  # borrow from bit 4
        if r == 0x7F:         f |= PV_FLAG  # overflow: 0x80→0x7F
        cpu.F = f
        return r

    def do_and8(v: int):
        r = cpu.A & v
        f = H_FLAG | (r & (Y_FLAG | X_FLAG))
        if r & 0x80: f |= S_FLAG
        if r == 0:   f |= Z_FLAG
        if _parity(r): f |= PV_FLAG
        cpu.F = f; cpu.A = r

    def do_xor8(v: int):
        r = cpu.A ^ v
        f = r & (Y_FLAG | X_FLAG)
        if r & 0x80: f |= S_FLAG
        if r == 0:   f |= Z_FLAG
        if _parity(r): f |= PV_FLAG
        cpu.F = f; cpu.A = r

    def do_or8(v: int):
        r = cpu.A | v
        f = r & (Y_FLAG | X_FLAG)
        if r & 0x80: f |= S_FLAG
        if r == 0:   f |= Z_FLAG
        if _parity(r): f |= PV_FLAG
        cpu.F = f; cpu.A = r

    def do_cp8(v: int):
        """CP — like SUB but A is not written; Y/X come from *operand*."""
        do_sub8(cpu.A, v)
        # Overwrite undocumented bits with bits from the comparand
        cpu.F = (cpu.F & ~(Y_FLAG | X_FLAG)) | (v & (Y_FLAG | X_FLAG))

    # -----------------------------------------------------------------------
    # Control-flow micro-helpers
    # -----------------------------------------------------------------------
    def _jr(taken: bool) -> int:
        offs = cpu._fetch()
        if offs >= 0x80: offs -= 0x100
        if taken:
            cpu.PC = (cpu.PC + offs) & 0xFFFF
            return 12
        return 7

    def _jp(taken: bool) -> int:
        addr = cpu._fetch16()
        if taken: cpu.PC = addr
        return 10

    def _call(taken: bool) -> int:
        addr = cpu._fetch16()
        if taken:
            cpu._push(cpu.PC)
            cpu.PC = addr
            return 17
        return 10

    def _ret(taken: bool) -> int:
        if taken:
            cpu.PC = cpu._pop()
            return 11
        return 5

    # -----------------------------------------------------------------------
    # Opcode table
    # -----------------------------------------------------------------------
    table = [None] * 256

    # =======================================================================
    # PART 1 — opcodes 0x00 – 0x3F
    # =======================================================================

    # 0x00  NOP
    def op_00(): return 4
    table[0x00] = op_00

    # 0x01  LD BC, nn
    def op_01(): cpu.BC = cpu._fetch16(); return 10
    table[0x01] = op_01

    # 0x02  LD (BC), A
    def op_02(): bus.write(cpu.BC, cpu.A); return 7
    table[0x02] = op_02

    # 0x03  INC BC
    def op_03(): cpu.BC = (cpu.BC + 1) & 0xFFFF; return 6
    table[0x03] = op_03

    # 0x04  INC B
    def op_04(): cpu.B = do_inc8(cpu.B); return 4
    table[0x04] = op_04

    # 0x05  DEC B
    def op_05(): cpu.B = do_dec8(cpu.B); return 4
    table[0x05] = op_05

    # 0x06  LD B, n
    def op_06(): cpu.B = cpu._fetch(); return 7
    table[0x06] = op_06

    # 0x07  RLCA
    def op_07():
        a = cpu.A; c = (a >> 7) & 1
        r = ((a << 1) | c) & 0xFF; cpu.A = r
        cpu.F = (cpu.F & (S_FLAG | Z_FLAG | PV_FLAG)) | (r & (Y_FLAG | X_FLAG)) | (C_FLAG if c else 0)
        return 4
    table[0x07] = op_07

    # 0x08  EX AF, AF'
    def op_08():
        cpu.A, cpu.A_ = cpu.A_, cpu.A
        cpu.F, cpu.F_ = cpu.F_, cpu.F
        return 4
    table[0x08] = op_08

    # 0x09  ADD HL, BC  (DD→ADD IX,BC)
    def op_09(): sHL(do_add16(gHL(), cpu.BC)); return 11
    table[0x09] = op_09

    # 0x0A  LD A, (BC)
    def op_0A(): cpu.A = bus.read(cpu.BC); return 7
    table[0x0A] = op_0A

    # 0x0B  DEC BC
    def op_0B(): cpu.BC = (cpu.BC - 1) & 0xFFFF; return 6
    table[0x0B] = op_0B

    # 0x0C  INC C
    def op_0C(): cpu.C = do_inc8(cpu.C); return 4
    table[0x0C] = op_0C

    # 0x0D  DEC C
    def op_0D(): cpu.C = do_dec8(cpu.C); return 4
    table[0x0D] = op_0D

    # 0x0E  LD C, n
    def op_0E(): cpu.C = cpu._fetch(); return 7
    table[0x0E] = op_0E

    # 0x0F  RRCA
    def op_0F():
        a = cpu.A; c = a & 1
        r = ((a >> 1) | (c << 7)) & 0xFF; cpu.A = r
        cpu.F = (cpu.F & (S_FLAG | Z_FLAG | PV_FLAG)) | (r & (Y_FLAG | X_FLAG)) | (C_FLAG if c else 0)
        return 4
    table[0x0F] = op_0F

    # 0x10  DJNZ e
    def op_10():
        cpu.B = (cpu.B - 1) & 0xFF
        offs  = cpu._fetch()
        if offs >= 0x80: offs -= 0x100
        if cpu.B:
            cpu.PC = (cpu.PC + offs) & 0xFFFF
            return 13
        return 8
    table[0x10] = op_10

    # 0x11  LD DE, nn
    def op_11(): cpu.DE = cpu._fetch16(); return 10
    table[0x11] = op_11

    # 0x12  LD (DE), A
    def op_12(): bus.write(cpu.DE, cpu.A); return 7
    table[0x12] = op_12

    # 0x13  INC DE
    def op_13(): cpu.DE = (cpu.DE + 1) & 0xFFFF; return 6
    table[0x13] = op_13

    # 0x14  INC D
    def op_14(): cpu.D = do_inc8(cpu.D); return 4
    table[0x14] = op_14

    # 0x15  DEC D
    def op_15(): cpu.D = do_dec8(cpu.D); return 4
    table[0x15] = op_15

    # 0x16  LD D, n
    def op_16(): cpu.D = cpu._fetch(); return 7
    table[0x16] = op_16

    # 0x17  RLA
    def op_17():
        a = cpu.A; old_c = 1 if (cpu.F & C_FLAG) else 0
        c = (a >> 7) & 1
        r = ((a << 1) | old_c) & 0xFF; cpu.A = r
        cpu.F = (cpu.F & (S_FLAG | Z_FLAG | PV_FLAG)) | (r & (Y_FLAG | X_FLAG)) | (C_FLAG if c else 0)
        return 4
    table[0x17] = op_17

    # 0x18  JR e
    def op_18(): return _jr(True)
    table[0x18] = op_18

    # 0x19  ADD HL, DE  (DD→ADD IX,DE)
    def op_19(): sHL(do_add16(gHL(), cpu.DE)); return 11
    table[0x19] = op_19

    # 0x1A  LD A, (DE)
    def op_1A(): cpu.A = bus.read(cpu.DE); return 7
    table[0x1A] = op_1A

    # 0x1B  DEC DE
    def op_1B(): cpu.DE = (cpu.DE - 1) & 0xFFFF; return 6
    table[0x1B] = op_1B

    # 0x1C  INC E
    def op_1C(): cpu.E = do_inc8(cpu.E); return 4
    table[0x1C] = op_1C

    # 0x1D  DEC E
    def op_1D(): cpu.E = do_dec8(cpu.E); return 4
    table[0x1D] = op_1D

    # 0x1E  LD E, n
    def op_1E(): cpu.E = cpu._fetch(); return 7
    table[0x1E] = op_1E

    # 0x1F  RRA
    def op_1F():
        a = cpu.A; old_c = 1 if (cpu.F & C_FLAG) else 0
        c = a & 1
        r = ((a >> 1) | (old_c << 7)) & 0xFF; cpu.A = r
        cpu.F = (cpu.F & (S_FLAG | Z_FLAG | PV_FLAG)) | (r & (Y_FLAG | X_FLAG)) | (C_FLAG if c else 0)
        return 4
    table[0x1F] = op_1F

    # 0x20  JR NZ, e
    def op_20(): return _jr(not (cpu.F & Z_FLAG))
    table[0x20] = op_20

    # 0x21  LD HL, nn  (DD→LD IX,nn)
    def op_21(): sHL(cpu._fetch16()); return 10
    table[0x21] = op_21

    # 0x22  LD (nn), HL  (DD→LD (nn),IX)
    def op_22(): cpu._write16(cpu._fetch16(), gHL()); return 16
    table[0x22] = op_22

    # 0x23  INC HL  (DD→INC IX)
    def op_23(): sHL((gHL() + 1) & 0xFFFF); return 6
    table[0x23] = op_23

    # 0x24  INC H  (DD→INC IXH)
    def op_24(): setr(4, do_inc8(getr(4))); return 4
    table[0x24] = op_24

    # 0x25  DEC H  (DD→DEC IXH)
    def op_25(): setr(4, do_dec8(getr(4))); return 4
    table[0x25] = op_25

    # 0x26  LD H, n  (DD→LD IXH, n)
    def op_26(): setr(4, cpu._fetch()); return 7
    table[0x26] = op_26

    # 0x27  DAA
    def op_27():
        a    = cpu.A
        f    = cpu.F
        c_in = bool(f & C_FLAG)
        h_in = bool(f & H_FLAG)
        n_in = bool(f & N_FLAG)
        c_out = c_in
        h_out = False
        if not n_in:                        # after addition
            if h_in or (a & 0xF) > 9:
                h_out = ((a & 0xF) + 6) > 0xF
                a    += 6
            if c_in or a > 0x9F:
                a     += 0x60
                c_out  = True
        else:                               # after subtraction
            if h_in:
                h_out = (a & 0xF) < 6
                a     = (a - 6) & 0xFF
            if c_in:
                a     = (a - 0x60) & 0xFF
                c_out = True
        a &= 0xFF; cpu.A = a
        cpu.F = (
            (S_FLAG  if a & 0x80    else 0) |
            (Z_FLAG  if a == 0      else 0) |
            (a & (Y_FLAG | X_FLAG))         |
            (H_FLAG  if h_out       else 0) |
            (PV_FLAG if _parity(a)  else 0) |
            (N_FLAG  if n_in        else 0) |
            (C_FLAG  if c_out       else 0)
        )
        return 4
    table[0x27] = op_27

    # 0x28  JR Z, e
    def op_28(): return _jr(bool(cpu.F & Z_FLAG))
    table[0x28] = op_28

    # 0x29  ADD HL, HL  (DD→ADD IX,IX)
    def op_29(): sHL(do_add16(gHL(), gHL())); return 11
    table[0x29] = op_29

    # 0x2A  LD HL, (nn)  (DD→LD IX,(nn))
    def op_2A(): sHL(cpu._read16(cpu._fetch16())); return 16
    table[0x2A] = op_2A

    # 0x2B  DEC HL  (DD→DEC IX)
    def op_2B(): sHL((gHL() - 1) & 0xFFFF); return 6
    table[0x2B] = op_2B

    # 0x2C  INC L  (DD→INC IXL)
    def op_2C(): setr(5, do_inc8(getr(5))); return 4
    table[0x2C] = op_2C

    # 0x2D  DEC L  (DD→DEC IXL)
    def op_2D(): setr(5, do_dec8(getr(5))); return 4
    table[0x2D] = op_2D

    # 0x2E  LD L, n  (DD→LD IXL, n)
    def op_2E(): setr(5, cpu._fetch()); return 7
    table[0x2E] = op_2E

    # 0x2F  CPL
    def op_2F():
        cpu.A ^= 0xFF
        cpu.F = ((cpu.F & (S_FLAG | Z_FLAG | PV_FLAG | C_FLAG))
                 | H_FLAG | N_FLAG | (cpu.A & (Y_FLAG | X_FLAG)))
        return 4
    table[0x2F] = op_2F

    # 0x30  JR NC, e
    def op_30(): return _jr(not (cpu.F & C_FLAG))
    table[0x30] = op_30

    # 0x31  LD SP, nn
    def op_31(): cpu.SP = cpu._fetch16(); return 10
    table[0x31] = op_31

    # 0x32  LD (nn), A
    def op_32(): bus.write(cpu._fetch16(), cpu.A); return 13
    table[0x32] = op_32

    # 0x33  INC SP
    def op_33(): cpu.SP = (cpu.SP + 1) & 0xFFFF; return 6
    table[0x33] = op_33

    # 0x34  INC (HL)  (DD→INC (IX+d); handler returns 19 so +4 = 23 total)
    def op_34():
        if cpu._dd or cpu._fd:
            addr = cpu._idx_addr
            bus.write(addr, do_inc8(bus.read(addr)))
            return 19
        bus.write(cpu.HL, do_inc8(bus.read(cpu.HL)))
        return 11
    table[0x34] = op_34

    # 0x35  DEC (HL)  (DD→DEC (IX+d))
    def op_35():
        if cpu._dd or cpu._fd:
            addr = cpu._idx_addr
            bus.write(addr, do_dec8(bus.read(addr)))
            return 19
        bus.write(cpu.HL, do_dec8(bus.read(cpu.HL)))
        return 11
    table[0x35] = op_35

    # 0x36  LD (HL), n  (DD→LD (IX+d),n; handler returns 15 so +4 = 19 total)
    def op_36():
        if cpu._dd or cpu._fd:
            bus.write(cpu._idx_addr, cpu._fetch())
            return 15
        bus.write(cpu.HL, cpu._fetch())
        return 10
    table[0x36] = op_36

    # 0x37  SCF
    def op_37():
        cpu.F = ((cpu.F & (S_FLAG | Z_FLAG | PV_FLAG))
                 | C_FLAG | (cpu.A & (Y_FLAG | X_FLAG)))
        return 4
    table[0x37] = op_37

    # 0x38  JR C, e
    def op_38(): return _jr(bool(cpu.F & C_FLAG))
    table[0x38] = op_38

    # 0x39  ADD HL, SP  (DD→ADD IX,SP)
    def op_39(): sHL(do_add16(gHL(), cpu.SP)); return 11
    table[0x39] = op_39

    # 0x3A  LD A, (nn)
    def op_3A(): cpu.A = bus.read(cpu._fetch16()); return 13
    table[0x3A] = op_3A

    # 0x3B  DEC SP
    def op_3B(): cpu.SP = (cpu.SP - 1) & 0xFFFF; return 6
    table[0x3B] = op_3B

    # 0x3C  INC A
    def op_3C(): cpu.A = do_inc8(cpu.A); return 4
    table[0x3C] = op_3C

    # 0x3D  DEC A
    def op_3D(): cpu.A = do_dec8(cpu.A); return 4
    table[0x3D] = op_3D

    # 0x3E  LD A, n
    def op_3E(): cpu.A = cpu._fetch(); return 7
    table[0x3E] = op_3E

    # 0x3F  CCF
    def op_3F():
        old_c = cpu.F & C_FLAG
        cpu.F = ((cpu.F & (S_FLAG | Z_FLAG | PV_FLAG))
                 | (H_FLAG if old_c else 0)
                 | (0      if old_c else C_FLAG)
                 | (cpu.A & (Y_FLAG | X_FLAG)))
        return 4
    table[0x3F] = op_3F

    # =======================================================================
    # PART 2 — opcodes 0x40 – 0xBF  (filled in below)
    # PART 3 — opcodes 0xC0 – 0xFF and ED table (filled in below)
    # =======================================================================

    # -----------------------------------------------------------------------
    # Stubs for not-yet-filled entries (will be overwritten by parts 2/3)
    # -----------------------------------------------------------------------
    def _nop_stub(): return 4
    for _i in range(0x40, 0x100):
        table[_i] = _nop_stub

    ed = {}

    return table, ed
