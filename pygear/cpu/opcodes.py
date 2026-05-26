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
    # PART 2 — opcodes 0x40 – 0xBF
    # =======================================================================

    # -----------------------------------------------------------------------
    # 0x40 – 0x7F  LD r, r'  (0x76 = HALT)
    # dst = bits 5-3,  src = bits 2-0
    # Encoding: 0=B 1=C 2=D 3=E 4=H 5=L 6=(HL) 7=A
    # -----------------------------------------------------------------------

    # 0x76  HALT
    def op_76():
        cpu.halted = True
        cpu.PC = (cpu.PC - 1) & 0xFFFF   # re-execute HALT each cycle
        return 4
    table[0x76] = op_76

    for _code in range(0x40, 0x80):
        if _code == 0x76:
            continue
        _dst = (_code >> 3) & 0x07
        _src = _code & 0x07

        def _make_ld_rr(d, s):
            def handler():
                if (cpu._dd or cpu._fd) and (d == 6 or s == 6):
                    # When one operand is (IX+d)/(IY+d) memory, H and L on
                    # the register side stay as plain H/L, NOT IXH/IXL.
                    # e.g. LD H,(IX+d) writes to H; LD (IX+d),H reads from H.
                    if s == 6:
                        val = bus.read(cpu._idx_addr)
                        if   d == 4: cpu.H = val & 0xFF
                        elif d == 5: cpu.L = val & 0xFF
                        else:        setr(d, val)
                    else:
                        val = (cpu.H if s == 4 else
                               cpu.L if s == 5 else getr(s))
                        bus.write(cpu._idx_addr, val)
                    return 15           # 15 + 4 prefix = 19 total
                setr(d, getr(s))
                return 7 if (d == 6 or s == 6) else 4
            return handler

        table[_code] = _make_ld_rr(_dst, _src)

    # -----------------------------------------------------------------------
    # 0x80 – 0xBF  ALU A, r
    # op  = bits 5-3   (ADD=0 ADC=1 SUB=2 SBC=3 AND=4 XOR=5 OR=6 CP=7)
    # src = bits 2-0
    # -----------------------------------------------------------------------

    def _alu_cycles(s):
        """Return T-states for ALU op on register s, accounting for DD/FD."""
        if (cpu._dd or cpu._fd) and s == 6:
            return 15           # 15 + 4 prefix = 19 total
        return 7 if s == 6 else 4

    # ADD A, r  (0x80–0x87)
    for _code in range(0x80, 0x88):
        _src = _code & 0x07
        def _make_add(s):
            def handler():
                cpu.A = do_add8(cpu.A, getr(s))
                return _alu_cycles(s)
            return handler
        table[_code] = _make_add(_src)

    # ADC A, r  (0x88–0x8F)
    for _code in range(0x88, 0x90):
        _src = _code & 0x07
        def _make_adc(s):
            def handler():
                cpu.A = do_add8(cpu.A, getr(s), 1 if (cpu.F & C_FLAG) else 0)
                return _alu_cycles(s)
            return handler
        table[_code] = _make_adc(_src)

    # SUB r  (0x90–0x97)
    for _code in range(0x90, 0x98):
        _src = _code & 0x07
        def _make_sub(s):
            def handler():
                cpu.A = do_sub8(cpu.A, getr(s))
                return _alu_cycles(s)
            return handler
        table[_code] = _make_sub(_src)

    # SBC A, r  (0x98–0x9F)
    for _code in range(0x98, 0xA0):
        _src = _code & 0x07
        def _make_sbc(s):
            def handler():
                cpu.A = do_sub8(cpu.A, getr(s), 1 if (cpu.F & C_FLAG) else 0)
                return _alu_cycles(s)
            return handler
        table[_code] = _make_sbc(_src)

    # AND r  (0xA0–0xA7)
    for _code in range(0xA0, 0xA8):
        _src = _code & 0x07
        def _make_and(s):
            def handler():
                do_and8(getr(s))
                return _alu_cycles(s)
            return handler
        table[_code] = _make_and(_src)

    # XOR r  (0xA8–0xAF)
    for _code in range(0xA8, 0xB0):
        _src = _code & 0x07
        def _make_xor(s):
            def handler():
                do_xor8(getr(s))
                return _alu_cycles(s)
            return handler
        table[_code] = _make_xor(_src)

    # OR r  (0xB0–0xB7)
    for _code in range(0xB0, 0xB8):
        _src = _code & 0x07
        def _make_or(s):
            def handler():
                do_or8(getr(s))
                return _alu_cycles(s)
            return handler
        table[_code] = _make_or(_src)

    # CP r  (0xB8–0xBF)
    for _code in range(0xB8, 0xC0):
        _src = _code & 0x07
        def _make_cp(s):
            def handler():
                do_cp8(getr(s))
                return _alu_cycles(s)
            return handler
        table[_code] = _make_cp(_src)

    # =======================================================================
    # PART 3 — opcodes 0xC0 – 0xFF
    # =======================================================================

    # 0xC0  RET NZ
    def op_C0(): return _ret(not (cpu.F & Z_FLAG))
    table[0xC0] = op_C0

    # 0xC1  POP BC
    def op_C1(): cpu.BC = cpu._pop(); return 10
    table[0xC1] = op_C1

    # 0xC2  JP NZ, nn
    def op_C2(): return _jp(not (cpu.F & Z_FLAG))
    table[0xC2] = op_C2

    # 0xC3  JP nn
    def op_C3(): cpu.PC = cpu._fetch16(); return 10
    table[0xC3] = op_C3

    # 0xC4  CALL NZ, nn
    def op_C4(): return _call(not (cpu.F & Z_FLAG))
    table[0xC4] = op_C4

    # 0xC5  PUSH BC
    def op_C5(): cpu._push(cpu.BC); return 11
    table[0xC5] = op_C5

    # 0xC6  ADD A, n
    def op_C6(): cpu.A = do_add8(cpu.A, cpu._fetch()); return 7
    table[0xC6] = op_C6

    # 0xC7  RST 00H
    def op_C7(): cpu._push(cpu.PC); cpu.PC = 0x0000; return 11
    table[0xC7] = op_C7

    # 0xC8  RET Z
    def op_C8(): return _ret(bool(cpu.F & Z_FLAG))
    table[0xC8] = op_C8

    # 0xC9  RET
    def op_C9(): cpu.PC = cpu._pop(); return 10
    table[0xC9] = op_C9

    # 0xCA  JP Z, nn
    def op_CA(): return _jp(bool(cpu.F & Z_FLAG))
    table[0xCA] = op_CA

    # 0xCB  — prefix, handled in step(); placeholder keeps slot non-None
    def op_CB(): return 4
    table[0xCB] = op_CB

    # 0xCC  CALL Z, nn
    def op_CC(): return _call(bool(cpu.F & Z_FLAG))
    table[0xCC] = op_CC

    # 0xCD  CALL nn
    def op_CD(): return _call(True)
    table[0xCD] = op_CD

    # 0xCE  ADC A, n
    def op_CE():
        cpu.A = do_add8(cpu.A, cpu._fetch(), 1 if (cpu.F & C_FLAG) else 0)
        return 7
    table[0xCE] = op_CE

    # 0xCF  RST 08H
    def op_CF(): cpu._push(cpu.PC); cpu.PC = 0x0008; return 11
    table[0xCF] = op_CF

    # 0xD0  RET NC
    def op_D0(): return _ret(not (cpu.F & C_FLAG))
    table[0xD0] = op_D0

    # 0xD1  POP DE
    def op_D1(): cpu.DE = cpu._pop(); return 10
    table[0xD1] = op_D1

    # 0xD2  JP NC, nn
    def op_D2(): return _jp(not (cpu.F & C_FLAG))
    table[0xD2] = op_D2

    # 0xD3  OUT (n), A
    def op_D3(): cpu.ports.write(cpu._fetch(), cpu.A); return 11
    table[0xD3] = op_D3

    # 0xD4  CALL NC, nn
    def op_D4(): return _call(not (cpu.F & C_FLAG))
    table[0xD4] = op_D4

    # 0xD5  PUSH DE
    def op_D5(): cpu._push(cpu.DE); return 11
    table[0xD5] = op_D5

    # 0xD6  SUB n
    def op_D6(): cpu.A = do_sub8(cpu.A, cpu._fetch()); return 7
    table[0xD6] = op_D6

    # 0xD7  RST 10H
    def op_D7(): cpu._push(cpu.PC); cpu.PC = 0x0010; return 11
    table[0xD7] = op_D7

    # 0xD8  RET C
    def op_D8(): return _ret(bool(cpu.F & C_FLAG))
    table[0xD8] = op_D8

    # 0xD9  EXX
    def op_D9():
        cpu.B, cpu.B_ = cpu.B_, cpu.B
        cpu.C, cpu.C_ = cpu.C_, cpu.C
        cpu.D, cpu.D_ = cpu.D_, cpu.D
        cpu.E, cpu.E_ = cpu.E_, cpu.E
        cpu.H, cpu.H_ = cpu.H_, cpu.H
        cpu.L, cpu.L_ = cpu.L_, cpu.L
        return 4
    table[0xD9] = op_D9

    # 0xDA  JP C, nn
    def op_DA(): return _jp(bool(cpu.F & C_FLAG))
    table[0xDA] = op_DA

    # 0xDB  IN A, (n)
    def op_DB(): cpu.A = cpu.ports.read(cpu._fetch()); return 11
    table[0xDB] = op_DB

    # 0xDC  CALL C, nn
    def op_DC(): return _call(bool(cpu.F & C_FLAG))
    table[0xDC] = op_DC

    # 0xDD  — prefix, handled in step()
    def op_DD(): return 4
    table[0xDD] = op_DD

    # 0xDE  SBC A, n
    def op_DE():
        cpu.A = do_sub8(cpu.A, cpu._fetch(), 1 if (cpu.F & C_FLAG) else 0)
        return 7
    table[0xDE] = op_DE

    # 0xDF  RST 18H
    def op_DF(): cpu._push(cpu.PC); cpu.PC = 0x0018; return 11
    table[0xDF] = op_DF

    # 0xE0  RET PO  (parity odd = PV clear)
    def op_E0(): return _ret(not (cpu.F & PV_FLAG))
    table[0xE0] = op_E0

    # 0xE1  POP HL  (DD→POP IX)
    def op_E1(): sHL(cpu._pop()); return 10
    table[0xE1] = op_E1

    # 0xE2  JP PO, nn
    def op_E2(): return _jp(not (cpu.F & PV_FLAG))
    table[0xE2] = op_E2

    # 0xE3  EX (SP), HL  (DD→EX (SP),IX)
    def op_E3():
        tmp = cpu._read16(cpu.SP)
        cpu._write16(cpu.SP, gHL())
        sHL(tmp)
        return 19
    table[0xE3] = op_E3

    # 0xE4  CALL PO, nn
    def op_E4(): return _call(not (cpu.F & PV_FLAG))
    table[0xE4] = op_E4

    # 0xE5  PUSH HL  (DD→PUSH IX)
    def op_E5(): cpu._push(gHL()); return 11
    table[0xE5] = op_E5

    # 0xE6  AND n
    def op_E6(): do_and8(cpu._fetch()); return 7
    table[0xE6] = op_E6

    # 0xE7  RST 20H
    def op_E7(): cpu._push(cpu.PC); cpu.PC = 0x0020; return 11
    table[0xE7] = op_E7

    # 0xE8  RET PE  (parity even = PV set)
    def op_E8(): return _ret(bool(cpu.F & PV_FLAG))
    table[0xE8] = op_E8

    # 0xE9  JP (HL)  (DD→JP (IX))
    def op_E9(): cpu.PC = gHL(); return 4
    table[0xE9] = op_E9

    # 0xEA  JP PE, nn
    def op_EA(): return _jp(bool(cpu.F & PV_FLAG))
    table[0xEA] = op_EA

    # 0xEB  EX DE, HL
    def op_EB():
        cpu.D, cpu.H = cpu.H, cpu.D
        cpu.E, cpu.L = cpu.L, cpu.E
        return 4
    table[0xEB] = op_EB

    # 0xEC  CALL PE, nn
    def op_EC(): return _call(bool(cpu.F & PV_FLAG))
    table[0xEC] = op_EC

    # 0xED  — prefix, handled in step()
    def op_ED(): return 4
    table[0xED] = op_ED

    # 0xEE  XOR n
    def op_EE(): do_xor8(cpu._fetch()); return 7
    table[0xEE] = op_EE

    # 0xEF  RST 28H
    def op_EF(): cpu._push(cpu.PC); cpu.PC = 0x0028; return 11
    table[0xEF] = op_EF

    # 0xF0  RET P  (positive = S clear)
    def op_F0(): return _ret(not (cpu.F & S_FLAG))
    table[0xF0] = op_F0

    # 0xF1  POP AF
    def op_F1(): cpu.AF = cpu._pop(); return 10
    table[0xF1] = op_F1

    # 0xF2  JP P, nn
    def op_F2(): return _jp(not (cpu.F & S_FLAG))
    table[0xF2] = op_F2

    # 0xF3  DI
    def op_F3(): cpu.IFF1 = cpu.IFF2 = False; return 4
    table[0xF3] = op_F3

    # 0xF4  CALL P, nn
    def op_F4(): return _call(not (cpu.F & S_FLAG))
    table[0xF4] = op_F4

    # 0xF5  PUSH AF
    def op_F5(): cpu._push(cpu.AF); return 11
    table[0xF5] = op_F5

    # 0xF6  OR n
    def op_F6(): do_or8(cpu._fetch()); return 7
    table[0xF6] = op_F6

    # 0xF7  RST 30H
    def op_F7(): cpu._push(cpu.PC); cpu.PC = 0x0030; return 11
    table[0xF7] = op_F7

    # 0xF8  RET M  (minus = S set)
    def op_F8(): return _ret(bool(cpu.F & S_FLAG))
    table[0xF8] = op_F8

    # 0xF9  LD SP, HL  (DD→LD SP,IX)
    def op_F9(): cpu.SP = gHL(); return 6
    table[0xF9] = op_F9

    # 0xFA  JP M, nn
    def op_FA(): return _jp(bool(cpu.F & S_FLAG))
    table[0xFA] = op_FA

    # 0xFB  EI — interrupt enabled after *next* instruction
    def op_FB():
        cpu.IFF1 = cpu.IFF2 = True
        cpu._ei_delay = True
        return 4
    table[0xFB] = op_FB

    # 0xFC  CALL M, nn
    def op_FC(): return _call(bool(cpu.F & S_FLAG))
    table[0xFC] = op_FC

    # 0xFD  — prefix, handled in step()
    def op_FD(): return 4
    table[0xFD] = op_FD

    # 0xFE  CP n
    def op_FE(): do_cp8(cpu._fetch()); return 7
    table[0xFE] = op_FE

    # 0xFF  RST 38H
    def op_FF(): cpu._push(cpu.PC); cpu.PC = 0x0038; return 11
    table[0xFF] = op_FF

    # =======================================================================
    # ED prefix table
    # =======================================================================
    ed = {}

    # -----------------------------------------------------------------------
    # Helper: set SZP flags from 8-bit value v (H=0 N=0, C unchanged)
    # -----------------------------------------------------------------------
    def _szp(v: int) -> int:
        f = cpu.F & C_FLAG
        f |= v & (Y_FLAG | X_FLAG)
        if v & 0x80: f |= S_FLAG
        if v == 0:   f |= Z_FLAG
        if _parity(v): f |= PV_FLAG
        return f

    # -----------------------------------------------------------------------
    # IN r, (C)  — 0x40/48/50/58/60/68/70/78
    # -----------------------------------------------------------------------
    _in_regs = {0x40: 'B', 0x48: 'C', 0x50: 'D', 0x58: 'E',
                0x60: 'H', 0x68: 'L', 0x70: None, 0x78: 'A'}

    def _make_in_r_c(reg_name):
        def handler():
            val = cpu.ports.read(cpu.C)
            cpu.F = _szp(val)
            if reg_name == 'B': cpu.B = val
            elif reg_name == 'C': cpu.C = val
            elif reg_name == 'D': cpu.D = val
            elif reg_name == 'E': cpu.E = val
            elif reg_name == 'H': cpu.H = val
            elif reg_name == 'L': cpu.L = val
            elif reg_name == 'A': cpu.A = val
            # None (0x70): read but discard — flags still set
            return 12
        return handler

    for _op, _rn in _in_regs.items():
        ed[_op] = _make_in_r_c(_rn)

    # -----------------------------------------------------------------------
    # OUT (C), r  — 0x41/49/51/59/61/69/71/79
    # -----------------------------------------------------------------------
    _out_regs = {0x41: 'B', 0x49: 'C', 0x51: 'D', 0x59: 'E',
                 0x61: 'H', 0x69: 'L', 0x71: None, 0x79: 'A'}

    def _make_out_c_r(reg_name):
        def handler():
            if   reg_name == 'B': val = cpu.B
            elif reg_name == 'C': val = cpu.C
            elif reg_name == 'D': val = cpu.D
            elif reg_name == 'E': val = cpu.E
            elif reg_name == 'H': val = cpu.H
            elif reg_name == 'L': val = cpu.L
            elif reg_name == 'A': val = cpu.A
            else: val = 0               # 0x71: OUT (C), 0  (undoc)
            cpu.ports.write(cpu.C, val)
            return 12
        return handler

    for _op, _rn in _out_regs.items():
        ed[_op] = _make_out_c_r(_rn)

    # -----------------------------------------------------------------------
    # SBC HL, rr  — 0x42/52/62/72
    # ADC HL, rr  — 0x4A/5A/6A/7A
    # -----------------------------------------------------------------------
    def _make_sbc_hl(get_rr):
        def handler():
            cpu.HL = do_sbc16(cpu.HL, get_rr()); return 15
        return handler

    def _make_adc_hl(get_rr):
        def handler():
            cpu.HL = do_adc16(cpu.HL, get_rr()); return 15
        return handler

    ed[0x42] = _make_sbc_hl(lambda: cpu.BC)
    ed[0x52] = _make_sbc_hl(lambda: cpu.DE)
    ed[0x62] = _make_sbc_hl(lambda: cpu.HL)
    ed[0x72] = _make_sbc_hl(lambda: cpu.SP)
    ed[0x4A] = _make_adc_hl(lambda: cpu.BC)
    ed[0x5A] = _make_adc_hl(lambda: cpu.DE)
    ed[0x6A] = _make_adc_hl(lambda: cpu.HL)
    ed[0x7A] = _make_adc_hl(lambda: cpu.SP)

    # -----------------------------------------------------------------------
    # LD (nn), rr  — 0x43/53/63/73
    # LD rr, (nn)  — 0x4B/5B/6B/7B
    # -----------------------------------------------------------------------
    def _make_ld_nn_rr(get_rr):
        def handler():
            cpu._write16(cpu._fetch16(), get_rr()); return 20
        return handler

    def _make_ld_rr_nn(set_rr):
        def handler():
            set_rr(cpu._read16(cpu._fetch16())); return 20
        return handler

    ed[0x43] = _make_ld_nn_rr(lambda: cpu.BC)
    ed[0x53] = _make_ld_nn_rr(lambda: cpu.DE)
    ed[0x63] = _make_ld_nn_rr(lambda: cpu.HL)
    ed[0x73] = _make_ld_nn_rr(lambda: cpu.SP)

    def _set_bc(v): cpu.BC = v
    def _set_de(v): cpu.DE = v
    def _set_hl(v): cpu.HL = v
    def _set_sp(v): cpu.SP = v & 0xFFFF

    ed[0x4B] = _make_ld_rr_nn(_set_bc)
    ed[0x5B] = _make_ld_rr_nn(_set_de)
    ed[0x6B] = _make_ld_rr_nn(_set_hl)
    ed[0x7B] = _make_ld_rr_nn(_set_sp)

    # -----------------------------------------------------------------------
    # NEG  — 0x44 (+ undoc mirrors 0x4C/54/5C/64/6C/74/7C)
    # -----------------------------------------------------------------------
    def op_neg():
        a = cpu.A
        cpu.A = do_sub8(0, a)
        # Overflow: only when A was 0x80
        if a == 0x80: cpu.F |= PV_FLAG
        else:         cpu.F &= ~PV_FLAG
        # Carry: set whenever A was non-zero
        if a != 0: cpu.F |= C_FLAG
        else:      cpu.F &= ~C_FLAG
        return 8

    for _op in (0x44, 0x4C, 0x54, 0x5C, 0x64, 0x6C, 0x74, 0x7C):
        ed[_op] = op_neg

    # -----------------------------------------------------------------------
    # RETN  — 0x45 (+ mirrors 0x55/65/75/5D/6D/7D)
    # RETI  — 0x4D
    # -----------------------------------------------------------------------
    def op_retn():
        cpu.IFF1 = cpu.IFF2
        cpu.PC   = cpu._pop()
        return 14

    for _op in (0x45, 0x55, 0x65, 0x75, 0x5D, 0x6D, 0x7D):
        ed[_op] = op_retn

    def op_reti():
        cpu.IFF1 = cpu.IFF2
        cpu.PC   = cpu._pop()
        return 14

    ed[0x4D] = op_reti

    # -----------------------------------------------------------------------
    # IM 0 / IM 1 / IM 2
    # -----------------------------------------------------------------------
    def op_im0(): cpu.IM = 0; return 8
    def op_im1(): cpu.IM = 1; return 8
    def op_im2(): cpu.IM = 2; return 8

    for _op in (0x46, 0x4E, 0x66, 0x6E): ed[_op] = op_im0
    for _op in (0x56, 0x76):             ed[_op] = op_im1
    for _op in (0x5E, 0x7E):             ed[_op] = op_im2

    # -----------------------------------------------------------------------
    # LD I, A / LD R, A / LD A, I / LD A, R
    # -----------------------------------------------------------------------
    def op_47(): cpu.I = cpu.A; return 9     # LD I, A
    def op_4F(): cpu.R = cpu.A; return 9     # LD R, A

    def _ld_a_ir(val):
        cpu.A = val
        f = cpu.F & C_FLAG
        f |= val & (Y_FLAG | X_FLAG)
        if val & 0x80: f |= S_FLAG
        if val == 0:   f |= Z_FLAG
        if cpu.IFF2:   f |= PV_FLAG
        cpu.F = f

    def op_57(): _ld_a_ir(cpu.I); return 9  # LD A, I
    def op_5F(): _ld_a_ir(cpu.R); return 9  # LD A, R

    ed[0x47] = op_47; ed[0x4F] = op_4F
    ed[0x57] = op_57; ed[0x5F] = op_5F

    # -----------------------------------------------------------------------
    # RLD / RRD
    # -----------------------------------------------------------------------
    def op_6F():  # RLD
        m = bus.read(cpu.HL)
        bus.write(cpu.HL, ((m << 4) | (cpu.A & 0x0F)) & 0xFF)
        cpu.A = (cpu.A & 0xF0) | (m >> 4)
        cpu.F = _szp(cpu.A) | (cpu.F & C_FLAG)
        return 18

    def op_67():  # RRD
        m = bus.read(cpu.HL)
        bus.write(cpu.HL, ((cpu.A & 0x0F) << 4) | (m >> 4))
        cpu.A = (cpu.A & 0xF0) | (m & 0x0F)
        cpu.F = _szp(cpu.A) | (cpu.F & C_FLAG)
        return 18

    ed[0x6F] = op_6F; ed[0x67] = op_67

    # -----------------------------------------------------------------------
    # Block transfer / search / IO instructions
    # -----------------------------------------------------------------------

    def _ldi_step() -> int:
        """Single LDI step; returns value copied (for flag Y/X)."""
        val = bus.read(cpu.HL)
        bus.write(cpu.DE, val)
        cpu.HL = (cpu.HL + 1) & 0xFFFF
        cpu.DE = (cpu.DE + 1) & 0xFFFF
        cpu.BC = (cpu.BC - 1) & 0xFFFF
        return val

    def _ldd_step() -> int:
        val = bus.read(cpu.HL)
        bus.write(cpu.DE, val)
        cpu.HL = (cpu.HL - 1) & 0xFFFF
        cpu.DE = (cpu.DE - 1) & 0xFFFF
        cpu.BC = (cpu.BC - 1) & 0xFFFF
        return val

    def _ldi_flags(val: int):
        n = (cpu.A + val) & 0xFF
        f = cpu.F & (S_FLAG | Z_FLAG | C_FLAG)
        if cpu.BC:      f |= PV_FLAG
        f |= (n << 1) & Y_FLAG   # bit 1 of (A+val) → F bit 5
        f |= n & X_FLAG           # bit 3 of (A+val) → F bit 3
        cpu.F = f

    def op_A0():  # LDI
        _ldi_flags(_ldi_step()); return 16
    def op_A8():  # LDD
        _ldi_flags(_ldd_step()); return 16
    ed[0xA0] = op_A0; ed[0xA8] = op_A8

    def op_B0():  # LDIR — one iteration per step(); re-execute if BC != 0
        val = _ldi_step(); _ldi_flags(val)
        if cpu.BC != 0: cpu.PC = (cpu.PC - 2) & 0xFFFF; return 21
        return 16

    def op_B8():  # LDDR — one iteration per step(); re-execute if BC != 0
        val = _ldd_step(); _ldi_flags(val)
        if cpu.BC != 0: cpu.PC = (cpu.PC - 2) & 0xFFFF; return 21
        return 16

    ed[0xB0] = op_B0; ed[0xB8] = op_B8

    # CPI / CPD flags
    def _cpi_flags(val: int, inc: bool):
        diff = (cpu.A - val) & 0xFF
        h    = ((cpu.A & 0xF) - (val & 0xF)) < 0
        n    = (diff - (1 if h else 0)) & 0xFF
        if inc: cpu.HL = (cpu.HL + 1) & 0xFFFF
        else:   cpu.HL = (cpu.HL - 1) & 0xFFFF
        cpu.BC = (cpu.BC - 1) & 0xFFFF
        f    = (cpu.F & C_FLAG) | N_FLAG
        if h:           f |= H_FLAG
        if diff & 0x80: f |= S_FLAG
        if diff == 0:   f |= Z_FLAG
        if cpu.BC:      f |= PV_FLAG   # PV reflects BC after decrement
        f |= (n << 1) & Y_FLAG
        f |= n & X_FLAG
        cpu.F = f

    def op_A1():  # CPI
        val = bus.read(cpu.HL); _cpi_flags(val, True); return 16
    def op_A9():  # CPD
        val = bus.read(cpu.HL); _cpi_flags(val, False); return 16
    ed[0xA1] = op_A1; ed[0xA9] = op_A9

    def op_B1():  # CPIR — one iteration per step(); re-execute if BC != 0 and not found
        val = bus.read(cpu.HL); _cpi_flags(val, True)
        if cpu.BC == 0 or (cpu.F & Z_FLAG): return 16
        cpu.PC = (cpu.PC - 2) & 0xFFFF; return 21

    def op_B9():  # CPDR — one iteration per step(); re-execute if BC != 0 and not found
        val = bus.read(cpu.HL); _cpi_flags(val, False)
        if cpu.BC == 0 or (cpu.F & Z_FLAG): return 16
        cpu.PC = (cpu.PC - 2) & 0xFFFF; return 21

    ed[0xB1] = op_B1; ed[0xB9] = op_B9

    # INI / IND / INIR / INDR
    def _ini_step(inc: bool):
        val    = cpu.ports.read(cpu.C)
        bus.write(cpu.HL, val)
        cpu.HL = (cpu.HL + (1 if inc else -1)) & 0xFFFF
        cpu.B  = (cpu.B - 1) & 0xFF
        c_adj  = (cpu.C + (1 if inc else -1)) & 0xFF
        t      = val + c_adj
        f      = cpu.B & (S_FLAG | Y_FLAG | X_FLAG)   # S/YF/XF from B after dec
        if cpu.B == 0:              f |= Z_FLAG
        if val & 0x80:              f |= N_FLAG
        if t > 0xFF:                f |= H_FLAG | C_FLAG
        if _parity((t & 7) ^ cpu.B): f |= PV_FLAG
        cpu.F = f

    def op_A2(): _ini_step(True);  return 16   # INI
    def op_AA(): _ini_step(False); return 16   # IND
    ed[0xA2] = op_A2; ed[0xAA] = op_AA

    def op_B2():  # INIR — one iteration per step(); re-execute if B != 0
        _ini_step(True)
        if cpu.B == 0: return 16
        cpu.PC = (cpu.PC - 2) & 0xFFFF; return 21

    def op_BA():  # INDR — one iteration per step(); re-execute if B != 0
        _ini_step(False)
        if cpu.B == 0: return 16
        cpu.PC = (cpu.PC - 2) & 0xFFFF; return 21

    ed[0xB2] = op_B2; ed[0xBA] = op_BA

    # OUTI / OUTD / OTIR / OTDR
    def _outi_step(inc: bool):
        cpu.B  = (cpu.B - 1) & 0xFF
        val    = bus.read(cpu.HL)
        cpu.ports.write(cpu.C, val)
        cpu.HL = (cpu.HL + (1 if inc else -1)) & 0xFFFF
        t      = val + (cpu.HL & 0xFF)           # L after HL ±1
        f      = cpu.B & (S_FLAG | Y_FLAG | X_FLAG)   # S/YF/XF from B after dec
        if cpu.B == 0:              f |= Z_FLAG
        if val & 0x80:              f |= N_FLAG
        if t > 0xFF:                f |= H_FLAG | C_FLAG
        if _parity((t & 7) ^ cpu.B): f |= PV_FLAG
        cpu.F = f

    def op_A3(): _outi_step(True);  return 16   # OUTI
    def op_AB(): _outi_step(False); return 16   # OUTD
    ed[0xA3] = op_A3; ed[0xAB] = op_AB

    def op_B3():  # OTIR — one iteration per step(); re-execute if B != 0
        _outi_step(True)
        if cpu.B == 0: return 16
        cpu.PC = (cpu.PC - 2) & 0xFFFF; return 21

    def op_BB():  # OTDR — one iteration per step(); re-execute if B != 0
        _outi_step(False)
        if cpu.B == 0: return 16
        cpu.PC = (cpu.PC - 2) & 0xFFFF; return 21

    ed[0xB3] = op_B3; ed[0xBB] = op_BB

    return table, ed
