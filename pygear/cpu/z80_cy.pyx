# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c
"""Cython Z80 CPU — native C implementation for performance.

Replaces pygear/cpu/z80.py on platforms where the extension can be compiled.
All register arithmetic runs as C ints; flag computation is inline.
Python-level bus.read/write and ports.read/write calls still cross the GIL
boundary, but all internal logic is native.
"""

# ---------------------------------------------------------------------------
# Flag bit positions (compile-time constants)
DEF S_FLAG  = 0x80
DEF Z_FLAG  = 0x40
DEF Y_FLAG  = 0x20
DEF H_FLAG  = 0x10
DEF X_FLAG  = 0x08
DEF PV_FLAG = 0x04
DEF N_FLAG  = 0x02
DEF C_FLAG  = 0x01

# Parity lookup table (1 = even parity)
cdef int _PARITY[256]
cdef int _NEEDS_DISP[256]   # 1 if opcode needs displacement fetch in DD/FD mode

cdef void _build_tables():
    cdef int i, v
    for i in range(256):
        v = i; v ^= v >> 4; v ^= v >> 2; v ^= v >> 1
        _PARITY[i] = 1 if (v & 1) == 0 else 0
    for i in range(256):
        _NEEDS_DISP[i] = 0
    # Opcodes that access (HL) as memory when in DD/FD mode
    for i in [0x34, 0x35, 0x36,
              0x46, 0x4E, 0x56, 0x5E, 0x66, 0x6E,
              0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x77,
              0x7E,
              0x86, 0x8E, 0x96, 0x9E,
              0xA6, 0xAE, 0xB6, 0xBE]:
        _NEEDS_DISP[i] = 1

_build_tables()


cdef class CZ80:
    """Z80 CPU, fully compatible with the Python Z80 class."""

    # 8-bit main registers
    cdef public int A, F, B, C, D, E, H, L
    # Shadow registers
    cdef public int A_, F_, B_, C_, D_, E_, H_, L_
    # 16-bit registers
    cdef public int IX, IY, SP, PC, I, R
    # Interrupt flags
    cdef public bint IFF1, IFF2
    cdef public int IM
    cdef public bint halted
    cdef public bint _int_pending, _nmi_pending, _ei_delay
    # DD/FD prefix state
    cdef int _dd, _fd, _idx_addr
    # Cycle counter
    cdef public int cycles
    # External interfaces (Python objects — must be public for test access)
    cdef public object bus, ports

    def __init__(self, bus, ports):
        self.bus = bus
        self.ports = ports
        self.A = self.F = 0xFF
        self.B = self.C = 0xFF
        self.D = self.E = 0xFF
        self.H = self.L = 0xFF
        self.A_ = self.F_ = 0
        self.B_ = self.C_ = 0
        self.D_ = self.E_ = 0
        self.H_ = self.L_ = 0
        self.IX = self.IY = 0xFFFF
        self.SP = 0xFFFF
        self.PC = 0
        self.I = self.R = 0
        self.IFF1 = self.IFF2 = False
        self.IM = 1
        self.halted = False
        self._int_pending = self._nmi_pending = self._ei_delay = False
        self._dd = self._fd = 0
        self._idx_addr = 0
        self.cycles = 0

    # -------------------------------------------------------------------------
    # 16-bit register-pair properties
    # -------------------------------------------------------------------------
    @property
    def BC(self): return (self.B << 8) | self.C
    @BC.setter
    def BC(self, v): self.B = (v >> 8) & 0xFF; self.C = v & 0xFF

    @property
    def DE(self): return (self.D << 8) | self.E
    @DE.setter
    def DE(self, v): self.D = (v >> 8) & 0xFF; self.E = v & 0xFF

    @property
    def HL(self): return (self.H << 8) | self.L
    @HL.setter
    def HL(self, v): self.H = (v >> 8) & 0xFF; self.L = v & 0xFF

    @property
    def AF(self): return (self.A << 8) | self.F
    @AF.setter
    def AF(self, v): self.A = (v >> 8) & 0xFF; self.F = v & 0xFF

    # -------------------------------------------------------------------------
    # Memory / fetch helpers
    # -------------------------------------------------------------------------
    cdef inline int _fetch(self):
        cdef int v
        v = self.bus.read(self.PC)
        self.PC = (self.PC + 1) & 0xFFFF
        self.R = (self.R + 1) & 0x7F
        return v

    cdef inline int _fetch16(self):
        cdef int lo, hi
        lo = self._fetch()
        hi = self._fetch()
        return (hi << 8) | lo

    cpdef int _read16(self, int addr):
        cdef int lo, hi
        lo = self.bus.read(addr & 0xFFFF)
        hi = self.bus.read((addr + 1) & 0xFFFF)
        return (hi << 8) | lo

    cpdef void _write16(self, int addr, int v):
        self.bus.write(addr & 0xFFFF, v & 0xFF)
        self.bus.write((addr + 1) & 0xFFFF, (v >> 8) & 0xFF)

    cdef inline void _push(self, int v):
        self.SP = (self.SP - 1) & 0xFFFF
        self.bus.write(self.SP, (v >> 8) & 0xFF)
        self.SP = (self.SP - 1) & 0xFFFF
        self.bus.write(self.SP, v & 0xFF)

    cdef inline int _pop(self):
        cdef int lo, hi
        lo = self.bus.read(self.SP); self.SP = (self.SP + 1) & 0xFFFF
        hi = self.bus.read(self.SP); self.SP = (self.SP + 1) & 0xFFFF
        return (hi << 8) | lo

    # -------------------------------------------------------------------------
    # Register accessor helpers — DD/FD aware for H/L/(HL)
    # -------------------------------------------------------------------------
    cdef inline int _get_r(self, int idx):
        if idx == 0: return self.B
        if idx == 1: return self.C
        if idx == 2: return self.D
        if idx == 3: return self.E
        if idx == 4:
            if self._dd: return (self.IX >> 8) & 0xFF
            if self._fd: return (self.IY >> 8) & 0xFF
            return self.H
        if idx == 5:
            if self._dd: return self.IX & 0xFF
            if self._fd: return self.IY & 0xFF
            return self.L
        if idx == 6:
            if self._dd or self._fd: return self.bus.read(self._idx_addr)
            return self.bus.read((self.H << 8) | self.L)
        return self.A

    cdef inline void _set_r(self, int idx, int v):
        v &= 0xFF
        if idx == 0: self.B = v
        elif idx == 1: self.C = v
        elif idx == 2: self.D = v
        elif idx == 3: self.E = v
        elif idx == 4:
            if self._dd: self.IX = (self.IX & 0x00FF) | (v << 8)
            elif self._fd: self.IY = (self.IY & 0x00FF) | (v << 8)
            else: self.H = v
        elif idx == 5:
            if self._dd: self.IX = (self.IX & 0xFF00) | v
            elif self._fd: self.IY = (self.IY & 0xFF00) | v
            else: self.L = v
        elif idx == 6:
            if self._dd or self._fd: self.bus.write(self._idx_addr, v)
            else: self.bus.write((self.H << 8) | self.L, v)
        else: self.A = v

    cdef inline int _get_hl(self):
        if self._dd: return self.IX
        if self._fd: return self.IY
        return (self.H << 8) | self.L

    cdef inline void _set_hl(self, int v):
        v &= 0xFFFF
        if self._dd: self.IX = v
        elif self._fd: self.IY = v
        else:
            self.H = (v >> 8) & 0xFF
            self.L = v & 0xFF

    # -------------------------------------------------------------------------
    # ALU helpers — all inline for zero Python overhead
    # -------------------------------------------------------------------------
    cdef inline int _do_add8(self, int a, int b, int c):
        cdef int full, r, f
        full = a + b + c
        r = full & 0xFF
        f = r & (Y_FLAG | X_FLAG)
        if r & 0x80: f |= S_FLAG
        if r == 0: f |= Z_FLAG
        if full > 0xFF: f |= C_FLAG
        if ((a & 0xF) + (b & 0xF) + c) > 0xF: f |= H_FLAG
        if (~(a ^ b) & (a ^ r) & 0x80): f |= PV_FLAG
        self.F = f
        return r

    cdef inline int _do_sub8(self, int a, int b, int c):
        cdef int full, r, f
        full = a - b - c
        r = full & 0xFF
        f = N_FLAG | (r & (Y_FLAG | X_FLAG))
        if r & 0x80: f |= S_FLAG
        if r == 0: f |= Z_FLAG
        if full < 0: f |= C_FLAG
        if ((a & 0xF) - (b & 0xF) - c) < 0: f |= H_FLAG
        if ((a ^ b) & (a ^ r) & 0x80): f |= PV_FLAG
        self.F = f
        return r

    cdef inline int _do_add16(self, int hl, int rr):
        cdef int full, r, f
        full = hl + rr
        r = full & 0xFFFF
        f = self.F & (S_FLAG | Z_FLAG | PV_FLAG)
        if full > 0xFFFF: f |= C_FLAG
        if ((hl & 0xFFF) + (rr & 0xFFF)) > 0xFFF: f |= H_FLAG
        f |= (r >> 8) & (Y_FLAG | X_FLAG)
        self.F = f
        return r

    cdef inline int _do_adc16(self, int hl, int rr):
        cdef int cy, full, r, f
        cy = 1 if (self.F & C_FLAG) else 0
        full = hl + rr + cy
        r = full & 0xFFFF
        f = (r >> 8) & (Y_FLAG | X_FLAG)
        if r & 0x8000: f |= S_FLAG
        if r == 0: f |= Z_FLAG
        if full > 0xFFFF: f |= C_FLAG
        if ((hl & 0xFFF) + (rr & 0xFFF) + cy) > 0xFFF: f |= H_FLAG
        if (~(hl ^ rr) & (hl ^ r) & 0x8000): f |= PV_FLAG
        self.F = f
        return r

    cdef inline int _do_sbc16(self, int hl, int rr):
        cdef int cy, full, r, f
        cy = 1 if (self.F & C_FLAG) else 0
        full = hl - rr - cy
        r = full & 0xFFFF
        f = N_FLAG | ((r >> 8) & (Y_FLAG | X_FLAG))
        if r & 0x8000: f |= S_FLAG
        if r == 0: f |= Z_FLAG
        if full < 0: f |= C_FLAG
        if ((hl & 0xFFF) - (rr & 0xFFF) - cy) < 0: f |= H_FLAG
        if ((hl ^ rr) & (hl ^ r) & 0x8000): f |= PV_FLAG
        self.F = f
        return r

    cdef inline int _do_inc8(self, int v):
        cdef int r, f
        r = (v + 1) & 0xFF
        f = self.F & C_FLAG
        f |= r & (Y_FLAG | X_FLAG)
        if r & 0x80: f |= S_FLAG
        if r == 0: f |= Z_FLAG
        if (v & 0xF) == 0xF: f |= H_FLAG
        if r == 0x80: f |= PV_FLAG
        self.F = f
        return r

    cdef inline int _do_dec8(self, int v):
        cdef int r, f
        r = (v - 1) & 0xFF
        f = N_FLAG | (self.F & C_FLAG)
        f |= r & (Y_FLAG | X_FLAG)
        if r & 0x80: f |= S_FLAG
        if r == 0: f |= Z_FLAG
        if (v & 0xF) == 0: f |= H_FLAG
        if r == 0x7F: f |= PV_FLAG
        self.F = f
        return r

    cdef inline void _do_and8(self, int v):
        cdef int r, f
        r = self.A & v
        f = H_FLAG | (r & (Y_FLAG | X_FLAG))
        if r & 0x80: f |= S_FLAG
        if r == 0: f |= Z_FLAG
        if _PARITY[r]: f |= PV_FLAG
        self.F = f; self.A = r

    cdef inline void _do_xor8(self, int v):
        cdef int r, f
        r = (self.A ^ v) & 0xFF
        f = r & (Y_FLAG | X_FLAG)
        if r & 0x80: f |= S_FLAG
        if r == 0: f |= Z_FLAG
        if _PARITY[r]: f |= PV_FLAG
        self.F = f; self.A = r

    cdef inline void _do_or8(self, int v):
        cdef int r, f
        r = (self.A | v) & 0xFF
        f = r & (Y_FLAG | X_FLAG)
        if r & 0x80: f |= S_FLAG
        if r == 0: f |= Z_FLAG
        if _PARITY[r]: f |= PV_FLAG
        self.F = f; self.A = r

    cdef inline void _do_cp8(self, int v):
        self._do_sub8(self.A, v, 0)
        self.F = (self.F & ~(Y_FLAG | X_FLAG)) | (v & (Y_FLAG | X_FLAG))

    cdef inline int _szp(self, int v):
        cdef int f
        f = self.F & C_FLAG
        f |= v & (Y_FLAG | X_FLAG)
        if v & 0x80: f |= S_FLAG
        if v == 0: f |= Z_FLAG
        if _PARITY[v & 0xFF]: f |= PV_FLAG
        return f

    # -------------------------------------------------------------------------
    # CB-prefix shift/rotate helpers
    # -------------------------------------------------------------------------
    cdef inline int _sz_flags(self, int v):
        cdef int f
        f = v & (Y_FLAG | X_FLAG)
        if v & 0x80: f |= S_FLAG
        if v == 0: f |= Z_FLAG
        if _PARITY[v]: f |= PV_FLAG
        return f

    cdef inline int _rlc(self, int v):
        cdef int c, r
        c = (v >> 7) & 1
        r = ((v << 1) | c) & 0xFF
        self.F = self._sz_flags(r) | (C_FLAG if c else 0)
        return r

    cdef inline int _rrc(self, int v):
        cdef int c, r
        c = v & 1
        r = ((v >> 1) | (c << 7)) & 0xFF
        self.F = self._sz_flags(r) | (C_FLAG if c else 0)
        return r

    cdef inline int _rl(self, int v):
        cdef int old_c, c, r
        old_c = 1 if (self.F & C_FLAG) else 0
        c = (v >> 7) & 1
        r = ((v << 1) | old_c) & 0xFF
        self.F = self._sz_flags(r) | (C_FLAG if c else 0)
        return r

    cdef inline int _rr(self, int v):
        cdef int old_c, c, r
        old_c = 1 if (self.F & C_FLAG) else 0
        c = v & 1
        r = ((v >> 1) | (old_c << 7)) & 0xFF
        self.F = self._sz_flags(r) | (C_FLAG if c else 0)
        return r

    cdef inline int _sla(self, int v):
        cdef int c, r
        c = (v >> 7) & 1
        r = (v << 1) & 0xFF
        self.F = self._sz_flags(r) | (C_FLAG if c else 0)
        return r

    cdef inline int _sra(self, int v):
        cdef int c, r
        c = v & 1
        r = ((v >> 1) | (v & 0x80)) & 0xFF
        self.F = self._sz_flags(r) | (C_FLAG if c else 0)
        return r

    cdef inline int _sll(self, int v):
        cdef int c, r
        c = (v >> 7) & 1
        r = ((v << 1) | 1) & 0xFF
        self.F = self._sz_flags(r) | (C_FLAG if c else 0)
        return r

    cdef inline int _srl(self, int v):
        cdef int c, r
        c = v & 1
        r = (v >> 1) & 0xFF
        self.F = self._sz_flags(r) | (C_FLAG if c else 0)
        return r

    cdef inline int _cb_do_shift(self, int op_idx, int v):
        if op_idx == 0: return self._rlc(v)
        elif op_idx == 1: return self._rrc(v)
        elif op_idx == 2: return self._rl(v)
        elif op_idx == 3: return self._rr(v)
        elif op_idx == 4: return self._sla(v)
        elif op_idx == 5: return self._sra(v)
        elif op_idx == 6: return self._sll(v)
        else: return self._srl(v)

    # CB-plain register access (no DD/FD redirect)
    cdef inline int _cb_get_r(self, int idx):
        if idx == 0: return self.B
        if idx == 1: return self.C
        if idx == 2: return self.D
        if idx == 3: return self.E
        if idx == 4: return self.H
        if idx == 5: return self.L
        if idx == 6: return self.bus.read((self.H << 8) | self.L)
        return self.A

    cdef inline void _cb_set_r(self, int idx, int v):
        v &= 0xFF
        if idx == 0: self.B = v
        elif idx == 1: self.C = v
        elif idx == 2: self.D = v
        elif idx == 3: self.E = v
        elif idx == 4: self.H = v
        elif idx == 5: self.L = v
        elif idx == 6: self.bus.write((self.H << 8) | self.L, v)
        else: self.A = v

    # -------------------------------------------------------------------------
    # CB-prefix dispatch
    # -------------------------------------------------------------------------
    cdef int _step_cb(self):
        cdef int op, op_idx, r_idx, v, result, mask, tested, f
        op = self._fetch()
        op_idx = (op >> 3) & 7
        r_idx = op & 7

        if op < 0x40:
            v = self._cb_get_r(r_idx)
            result = self._cb_do_shift(op_idx, v)
            self._cb_set_r(r_idx, result)
            return 15 if r_idx == 6 else 8

        elif op < 0x80:
            # BIT b, r
            mask = 1 << op_idx
            v = self._cb_get_r(r_idx)
            tested = v & mask
            f = self.F & C_FLAG
            f |= H_FLAG
            if not tested: f |= Z_FLAG | PV_FLAG
            if tested & S_FLAG: f |= S_FLAG
            if r_idx != 6:
                f |= v & (Y_FLAG | X_FLAG)
            else:
                f |= self.H & (Y_FLAG | X_FLAG)
            self.F = f
            return 12 if r_idx == 6 else 8

        elif op < 0xC0:
            # RES b, r
            mask = (~(1 << op_idx)) & 0xFF
            v = self._cb_get_r(r_idx) & mask
            self._cb_set_r(r_idx, v)
            return 15 if r_idx == 6 else 8

        else:
            # SET b, r
            mask = 1 << op_idx
            v = self._cb_get_r(r_idx) | mask
            self._cb_set_r(r_idx, v)
            return 15 if r_idx == 6 else 8

    # -------------------------------------------------------------------------
    # DDCB / FDCB dispatch (_idx_addr already set by caller)
    # -------------------------------------------------------------------------
    cdef int _step_ddcb(self):
        cdef int op, op_idx, r_idx, v, result, mask, tested, f
        op = self._fetch()
        op_idx = (op >> 3) & 7
        r_idx = op & 7

        if op < 0x40:
            v = self.bus.read(self._idx_addr)
            result = self._cb_do_shift(op_idx, v)
            self.bus.write(self._idx_addr, result)
            if r_idx != 6:
                if r_idx == 0: self.B = result
                elif r_idx == 1: self.C = result
                elif r_idx == 2: self.D = result
                elif r_idx == 3: self.E = result
                elif r_idx == 4: self.H = result
                elif r_idx == 5: self.L = result
                elif r_idx == 7: self.A = result
            return 23

        elif op < 0x80:
            v = self.bus.read(self._idx_addr)
            tested = v & (1 << op_idx)
            f = self.F & C_FLAG
            f |= H_FLAG
            if not tested: f |= Z_FLAG | PV_FLAG
            if tested & 0x80: f |= S_FLAG
            f |= (self._idx_addr >> 8) & (Y_FLAG | X_FLAG)
            self.F = f
            return 20

        elif op < 0xC0:
            mask = (~(1 << op_idx)) & 0xFF
            result = self.bus.read(self._idx_addr) & mask
            self.bus.write(self._idx_addr, result)
            if r_idx != 6:
                if r_idx == 0: self.B = result
                elif r_idx == 1: self.C = result
                elif r_idx == 2: self.D = result
                elif r_idx == 3: self.E = result
                elif r_idx == 4: self.H = result
                elif r_idx == 5: self.L = result
                elif r_idx == 7: self.A = result
            return 23

        else:
            mask = 1 << op_idx
            result = self.bus.read(self._idx_addr) | mask
            self.bus.write(self._idx_addr, result)
            if r_idx != 6:
                if r_idx == 0: self.B = result
                elif r_idx == 1: self.C = result
                elif r_idx == 2: self.D = result
                elif r_idx == 3: self.E = result
                elif r_idx == 4: self.H = result
                elif r_idx == 5: self.L = result
                elif r_idx == 7: self.A = result
            return 23

    # -------------------------------------------------------------------------
    # DD / FD prefix dispatcher
    # -------------------------------------------------------------------------
    cdef int _step_indexed(self, int is_dd):
        cdef int op, d, base, c
        op = self._fetch()

        if op == 0xCB:
            d = self._fetch()
            if d >= 128: d -= 256
            base = self.IX if is_dd else self.IY
            self._idx_addr = (base + d) & 0xFFFF
            return self._step_ddcb()

        if _NEEDS_DISP[op]:
            d = self._fetch()
            if d >= 128: d -= 256
            base = self.IX if is_dd else self.IY
            self._idx_addr = (base + d) & 0xFFFF

        self._dd = is_dd
        self._fd = 1 - is_dd
        c = self._step_main(op)
        self._dd = 0
        self._fd = 0
        return c + 4

    # -------------------------------------------------------------------------
    # ED-prefix dispatch
    # -------------------------------------------------------------------------
    cdef int _step_ed(self):
        cdef int op, val, n, f, rr, lo, hi, c, t, c_adj
        op = self._fetch()

        # IN r,(C)
        if op == 0x40 or op == 0x48 or op == 0x50 or op == 0x58 or \
           op == 0x60 or op == 0x68 or op == 0x70 or op == 0x78:
            val = self.ports.read(self.C)
            self.F = self._szp(val)
            if   op == 0x40: self.B = val
            elif op == 0x48: self.C = val
            elif op == 0x50: self.D = val
            elif op == 0x58: self.E = val
            elif op == 0x60: self.H = val
            elif op == 0x68: self.L = val
            elif op == 0x70: pass  # discard
            elif op == 0x78: self.A = val
            return 12

        # OUT (C),r
        if op == 0x41 or op == 0x49 or op == 0x51 or op == 0x59 or \
           op == 0x61 or op == 0x69 or op == 0x71 or op == 0x79:
            if   op == 0x41: val = self.B
            elif op == 0x49: val = self.C
            elif op == 0x51: val = self.D
            elif op == 0x59: val = self.E
            elif op == 0x61: val = self.H
            elif op == 0x69: val = self.L
            elif op == 0x71: val = 0
            else:            val = self.A
            self.ports.write(self.C, val)
            return 12

        # SBC HL, rr
        if op == 0x42:
            self.HL = self._do_sbc16((self.H << 8) | self.L, (self.B << 8) | self.C)
            return 15
        if op == 0x52:
            self.HL = self._do_sbc16((self.H << 8) | self.L, (self.D << 8) | self.E)
            return 15
        if op == 0x62:
            rr = (self.H << 8) | self.L
            self.HL = self._do_sbc16(rr, rr)
            return 15
        if op == 0x72:
            self.HL = self._do_sbc16((self.H << 8) | self.L, self.SP)
            return 15

        # LD (nn), rr
        if op == 0x43:
            self._write16(self._fetch16(), (self.B << 8) | self.C); return 20
        if op == 0x53:
            self._write16(self._fetch16(), (self.D << 8) | self.E); return 20
        if op == 0x63:
            self._write16(self._fetch16(), (self.H << 8) | self.L); return 20
        if op == 0x73:
            self._write16(self._fetch16(), self.SP); return 20

        # NEG (+ mirrors)
        if op == 0x44 or op == 0x4C or op == 0x54 or op == 0x5C or \
           op == 0x64 or op == 0x6C or op == 0x74 or op == 0x7C:
            val = self.A
            self.A = self._do_sub8(0, val, 0)
            if val == 0x80: self.F |= PV_FLAG
            else: self.F &= ~PV_FLAG
            if val != 0: self.F |= C_FLAG
            else: self.F &= ~C_FLAG
            return 8

        # RETN (+ mirrors) and RETI
        if op == 0x45 or op == 0x55 or op == 0x65 or op == 0x75 or \
           op == 0x5D or op == 0x6D or op == 0x7D or op == 0x4D:
            self.IFF1 = self.IFF2
            self.PC = self._pop()
            return 14

        # IM 0
        if op == 0x46 or op == 0x4E or op == 0x66 or op == 0x6E:
            self.IM = 0; return 8
        # IM 1
        if op == 0x56 or op == 0x76:
            self.IM = 1; return 8
        # IM 2
        if op == 0x5E or op == 0x7E:
            self.IM = 2; return 8

        # LD I,A / LD R,A
        if op == 0x47: self.I = self.A; return 9
        if op == 0x4F: self.R = self.A; return 9

        # ADC HL, rr
        if op == 0x4A:
            self.HL = self._do_adc16((self.H << 8) | self.L, (self.B << 8) | self.C)
            return 15
        if op == 0x5A:
            self.HL = self._do_adc16((self.H << 8) | self.L, (self.D << 8) | self.E)
            return 15
        if op == 0x6A:
            rr = (self.H << 8) | self.L
            self.HL = self._do_adc16(rr, rr)
            return 15
        if op == 0x7A:
            self.HL = self._do_adc16((self.H << 8) | self.L, self.SP)
            return 15

        # LD rr, (nn)
        if op == 0x4B:
            rr = self._read16(self._fetch16())
            self.B = (rr >> 8) & 0xFF; self.C = rr & 0xFF; return 20
        if op == 0x5B:
            rr = self._read16(self._fetch16())
            self.D = (rr >> 8) & 0xFF; self.E = rr & 0xFF; return 20
        if op == 0x6B:
            rr = self._read16(self._fetch16())
            self.H = (rr >> 8) & 0xFF; self.L = rr & 0xFF; return 20
        if op == 0x7B:
            self.SP = self._read16(self._fetch16()); return 20

        # LD A,I / LD A,R
        if op == 0x57:
            val = self.I; self.A = val
            f = self.F & C_FLAG
            f |= val & (Y_FLAG | X_FLAG)
            if val & 0x80: f |= S_FLAG
            if val == 0: f |= Z_FLAG
            if self.IFF2: f |= PV_FLAG
            self.F = f; return 9
        if op == 0x5F:
            val = self.R; self.A = val
            f = self.F & C_FLAG
            f |= val & (Y_FLAG | X_FLAG)
            if val & 0x80: f |= S_FLAG
            if val == 0: f |= Z_FLAG
            if self.IFF2: f |= PV_FLAG
            self.F = f; return 9

        # RRD
        if op == 0x67:
            rr = (self.H << 8) | self.L
            val = self.bus.read(rr)
            self.bus.write(rr, ((self.A & 0x0F) << 4) | (val >> 4))
            self.A = (self.A & 0xF0) | (val & 0x0F)
            self.F = self._szp(self.A) | (self.F & C_FLAG)
            return 18

        # RLD
        if op == 0x6F:
            rr = (self.H << 8) | self.L
            val = self.bus.read(rr)
            self.bus.write(rr, ((val << 4) | (self.A & 0x0F)) & 0xFF)
            self.A = (self.A & 0xF0) | (val >> 4)
            self.F = self._szp(self.A) | (self.F & C_FLAG)
            return 18

        # LDI
        if op == 0xA0:
            val = self.bus.read((self.H << 8) | self.L)
            self.bus.write((self.D << 8) | self.E, val)
            self.HL = ((self.H << 8) | self.L) + 1
            self.DE = ((self.D << 8) | self.E) + 1
            self.BC = ((self.B << 8) | self.C) - 1
            n = (self.A + val) & 0xFF
            f = self.F & (S_FLAG | Z_FLAG | C_FLAG)
            if (self.B << 8) | self.C: f |= PV_FLAG
            f |= (n << 1) & Y_FLAG; f |= n & X_FLAG
            self.F = f; return 16

        # CPI
        if op == 0xA1:
            val = self.bus.read((self.H << 8) | self.L)
            self._cpi_step(val, True); return 16

        # INI
        if op == 0xA2:
            self._ini_step(True); return 16

        # OUTI
        if op == 0xA3:
            self._outi_step(True); return 16

        # LDD
        if op == 0xA8:
            val = self.bus.read((self.H << 8) | self.L)
            self.bus.write((self.D << 8) | self.E, val)
            self.HL = ((self.H << 8) | self.L) - 1
            self.DE = ((self.D << 8) | self.E) - 1
            self.BC = ((self.B << 8) | self.C) - 1
            n = (self.A + val) & 0xFF
            f = self.F & (S_FLAG | Z_FLAG | C_FLAG)
            if (self.B << 8) | self.C: f |= PV_FLAG
            f |= (n << 1) & Y_FLAG; f |= n & X_FLAG
            self.F = f; return 16

        # CPD
        if op == 0xA9:
            val = self.bus.read((self.H << 8) | self.L)
            self._cpi_step(val, False); return 16

        # IND
        if op == 0xAA:
            self._ini_step(False); return 16

        # OUTD
        if op == 0xAB:
            self._outi_step(False); return 16

        # LDIR
        if op == 0xB0:
            val = self.bus.read((self.H << 8) | self.L)
            self.bus.write((self.D << 8) | self.E, val)
            self.HL = ((self.H << 8) | self.L) + 1
            self.DE = ((self.D << 8) | self.E) + 1
            self.BC = ((self.B << 8) | self.C) - 1
            n = (self.A + val) & 0xFF
            f = self.F & (S_FLAG | Z_FLAG | C_FLAG)
            if (self.B << 8) | self.C:
                f |= PV_FLAG
                self.F = f | ((n << 1) & Y_FLAG) | (n & X_FLAG)
                self.PC = (self.PC - 2) & 0xFFFF
                return 21
            self.F = f | ((n << 1) & Y_FLAG) | (n & X_FLAG)
            return 16

        # CPIR
        if op == 0xB1:
            val = self.bus.read((self.H << 8) | self.L)
            self._cpi_step(val, True)
            if ((self.B << 8) | self.C) == 0 or (self.F & Z_FLAG): return 16
            self.PC = (self.PC - 2) & 0xFFFF; return 21

        # INIR
        if op == 0xB2:
            self._ini_step(True)
            if self.B == 0: return 16
            self.PC = (self.PC - 2) & 0xFFFF; return 21

        # OTIR
        if op == 0xB3:
            self._outi_step(True)
            if self.B == 0: return 16
            self.PC = (self.PC - 2) & 0xFFFF; return 21

        # LDDR
        if op == 0xB8:
            val = self.bus.read((self.H << 8) | self.L)
            self.bus.write((self.D << 8) | self.E, val)
            self.HL = ((self.H << 8) | self.L) - 1
            self.DE = ((self.D << 8) | self.E) - 1
            self.BC = ((self.B << 8) | self.C) - 1
            n = (self.A + val) & 0xFF
            f = self.F & (S_FLAG | Z_FLAG | C_FLAG)
            if (self.B << 8) | self.C:
                f |= PV_FLAG
                self.F = f | ((n << 1) & Y_FLAG) | (n & X_FLAG)
                self.PC = (self.PC - 2) & 0xFFFF
                return 21
            self.F = f | ((n << 1) & Y_FLAG) | (n & X_FLAG)
            return 16

        # CPDR
        if op == 0xB9:
            val = self.bus.read((self.H << 8) | self.L)
            self._cpi_step(val, False)
            if ((self.B << 8) | self.C) == 0 or (self.F & Z_FLAG): return 16
            self.PC = (self.PC - 2) & 0xFFFF; return 21

        # INDR
        if op == 0xBA:
            self._ini_step(False)
            if self.B == 0: return 16
            self.PC = (self.PC - 2) & 0xFFFF; return 21

        # OTDR
        if op == 0xBB:
            self._outi_step(False)
            if self.B == 0: return 16
            self.PC = (self.PC - 2) & 0xFFFF; return 21

        return 8   # undefined ED ops: NOP-like

    # -------------------------------------------------------------------------
    # Block instruction helpers
    # -------------------------------------------------------------------------
    cdef inline void _cpi_step(self, int val, bint inc):
        cdef int diff, h, n, f
        diff = (self.A - val) & 0xFF
        h = ((self.A & 0xF) - (val & 0xF)) < 0
        n = (diff - (1 if h else 0)) & 0xFF
        if inc: self.HL = ((self.H << 8) | self.L) + 1
        else:   self.HL = ((self.H << 8) | self.L) - 1
        self.BC = ((self.B << 8) | self.C) - 1
        f = (self.F & C_FLAG) | N_FLAG
        if h: f |= H_FLAG
        if diff & 0x80: f |= S_FLAG
        if diff == 0: f |= Z_FLAG
        if (self.B << 8) | self.C: f |= PV_FLAG
        f |= (n << 1) & Y_FLAG; f |= n & X_FLAG
        self.F = f

    cdef inline void _ini_step(self, bint inc):
        cdef int val, c_adj, t, f
        val = self.ports.read(self.C)
        self.bus.write((self.H << 8) | self.L, val)
        if inc: self.HL = ((self.H << 8) | self.L) + 1
        else:   self.HL = ((self.H << 8) | self.L) - 1
        self.B = (self.B - 1) & 0xFF
        c_adj = (self.C + (1 if inc else -1)) & 0xFF
        t = val + c_adj
        f = self.B & (S_FLAG | Y_FLAG | X_FLAG)
        if self.B == 0: f |= Z_FLAG
        if val & 0x80: f |= N_FLAG
        if t > 0xFF: f |= H_FLAG | C_FLAG
        if _PARITY[(t & 7) ^ self.B]: f |= PV_FLAG
        self.F = f

    cdef inline void _outi_step(self, bint inc):
        cdef int val, t, f
        self.B = (self.B - 1) & 0xFF
        val = self.bus.read((self.H << 8) | self.L)
        self.ports.write(self.C, val)
        if inc: self.HL = ((self.H << 8) | self.L) + 1
        else:   self.HL = ((self.H << 8) | self.L) - 1
        t = val + self.L
        f = self.B & (S_FLAG | Y_FLAG | X_FLAG)
        if self.B == 0: f |= Z_FLAG
        if val & 0x80: f |= N_FLAG
        if t > 0xFF: f |= H_FLAG | C_FLAG
        if _PARITY[(t & 7) ^ self.B]: f |= PV_FLAG
        self.F = f

    # -------------------------------------------------------------------------
    # Main opcode dispatch (also called from _step_indexed with _dd/_fd set)
    # -------------------------------------------------------------------------
    cdef int _step_main(self, int op):
        cdef int nn, v, a, c, offs, dst, src, alu_op, hl, addr, t, old_c
        cdef bint taken

        # ------------------------------------------------------------------
        # 0x00 – 0x3F
        # ------------------------------------------------------------------
        if op < 0x40:
            if op == 0x00: return 4   # NOP
            elif op == 0x01:
                v = self._fetch16(); self.B = (v >> 8) & 0xFF; self.C = v & 0xFF; return 10
            elif op == 0x02: self.bus.write((self.B << 8) | self.C, self.A); return 7
            elif op == 0x03:
                v = ((self.B << 8) | self.C) + 1
                self.B = (v >> 8) & 0xFF; self.C = v & 0xFF; return 6
            elif op == 0x04: self.B = self._do_inc8(self.B); return 4
            elif op == 0x05: self.B = self._do_dec8(self.B); return 4
            elif op == 0x06: self.B = self._fetch(); return 7
            elif op == 0x07:  # RLCA
                a = self.A; c = (a >> 7) & 1
                v = ((a << 1) | c) & 0xFF; self.A = v
                self.F = (self.F & (S_FLAG | Z_FLAG | PV_FLAG)) | (v & (Y_FLAG | X_FLAG)) | (C_FLAG if c else 0)
                return 4
            elif op == 0x08:  # EX AF,AF'
                self.A, self.A_ = self.A_, self.A
                self.F, self.F_ = self.F_, self.F
                return 4
            elif op == 0x09: self._set_hl(self._do_add16(self._get_hl(), (self.B << 8) | self.C)); return 11
            elif op == 0x0A: self.A = self.bus.read((self.B << 8) | self.C); return 7
            elif op == 0x0B:
                v = ((self.B << 8) | self.C) - 1
                self.B = (v >> 8) & 0xFF; self.C = v & 0xFF; return 6
            elif op == 0x0C: self.C = self._do_inc8(self.C); return 4
            elif op == 0x0D: self.C = self._do_dec8(self.C); return 4
            elif op == 0x0E: self.C = self._fetch(); return 7
            elif op == 0x0F:  # RRCA
                a = self.A; c = a & 1
                v = ((a >> 1) | (c << 7)) & 0xFF; self.A = v
                self.F = (self.F & (S_FLAG | Z_FLAG | PV_FLAG)) | (v & (Y_FLAG | X_FLAG)) | (C_FLAG if c else 0)
                return 4
            elif op == 0x10:  # DJNZ
                self.B = (self.B - 1) & 0xFF
                offs = self._fetch()
                if offs >= 128: offs -= 256
                if self.B:
                    self.PC = (self.PC + offs) & 0xFFFF; return 13
                return 8
            elif op == 0x11:
                v = self._fetch16(); self.D = (v >> 8) & 0xFF; self.E = v & 0xFF; return 10
            elif op == 0x12: self.bus.write((self.D << 8) | self.E, self.A); return 7
            elif op == 0x13:
                v = ((self.D << 8) | self.E) + 1
                self.D = (v >> 8) & 0xFF; self.E = v & 0xFF; return 6
            elif op == 0x14: self.D = self._do_inc8(self.D); return 4
            elif op == 0x15: self.D = self._do_dec8(self.D); return 4
            elif op == 0x16: self.D = self._fetch(); return 7
            elif op == 0x17:  # RLA
                a = self.A; old_c = 1 if (self.F & C_FLAG) else 0
                c = (a >> 7) & 1
                v = ((a << 1) | old_c) & 0xFF; self.A = v
                self.F = (self.F & (S_FLAG | Z_FLAG | PV_FLAG)) | (v & (Y_FLAG | X_FLAG)) | (C_FLAG if c else 0)
                return 4
            elif op == 0x18:  # JR e
                offs = self._fetch()
                if offs >= 128: offs -= 256
                self.PC = (self.PC + offs) & 0xFFFF; return 12
            elif op == 0x19: self._set_hl(self._do_add16(self._get_hl(), (self.D << 8) | self.E)); return 11
            elif op == 0x1A: self.A = self.bus.read((self.D << 8) | self.E); return 7
            elif op == 0x1B:
                v = ((self.D << 8) | self.E) - 1
                self.D = (v >> 8) & 0xFF; self.E = v & 0xFF; return 6
            elif op == 0x1C: self.E = self._do_inc8(self.E); return 4
            elif op == 0x1D: self.E = self._do_dec8(self.E); return 4
            elif op == 0x1E: self.E = self._fetch(); return 7
            elif op == 0x1F:  # RRA
                a = self.A; old_c = 1 if (self.F & C_FLAG) else 0
                c = a & 1
                v = ((a >> 1) | (old_c << 7)) & 0xFF; self.A = v
                self.F = (self.F & (S_FLAG | Z_FLAG | PV_FLAG)) | (v & (Y_FLAG | X_FLAG)) | (C_FLAG if c else 0)
                return 4
            elif op == 0x20:  # JR NZ
                offs = self._fetch()
                if offs >= 128: offs -= 256
                if not (self.F & Z_FLAG):
                    self.PC = (self.PC + offs) & 0xFFFF; return 12
                return 7
            elif op == 0x21: self._set_hl(self._fetch16()); return 10
            elif op == 0x22: self._write16(self._fetch16(), self._get_hl()); return 16
            elif op == 0x23: self._set_hl((self._get_hl() + 1) & 0xFFFF); return 6
            elif op == 0x24: self._set_r(4, self._do_inc8(self._get_r(4))); return 4
            elif op == 0x25: self._set_r(4, self._do_dec8(self._get_r(4))); return 4
            elif op == 0x26: self._set_r(4, self._fetch()); return 7
            elif op == 0x27:  # DAA
                a = self.A; f = self.F
                c_in = bool(f & C_FLAG); h_in = bool(f & H_FLAG); n_in = bool(f & N_FLAG)
                c_out = c_in; h_out = False
                if not n_in:
                    if h_in or (a & 0xF) > 9:
                        h_out = ((a & 0xF) + 6) > 0xF; a += 6
                    if c_in or a > 0x9F:
                        a += 0x60; c_out = True
                else:
                    if h_in:
                        h_out = (a & 0xF) < 6; a = (a - 6) & 0xFF
                    if c_in:
                        a = (a - 0x60) & 0xFF; c_out = True
                a &= 0xFF; self.A = a
                self.F = (
                    (S_FLAG  if a & 0x80   else 0) |
                    (Z_FLAG  if a == 0     else 0) |
                    (a & (Y_FLAG | X_FLAG))        |
                    (H_FLAG  if h_out      else 0) |
                    (PV_FLAG if _PARITY[a] else 0) |
                    (N_FLAG  if n_in       else 0) |
                    (C_FLAG  if c_out      else 0)
                )
                return 4
            elif op == 0x28:  # JR Z
                offs = self._fetch()
                if offs >= 128: offs -= 256
                if self.F & Z_FLAG:
                    self.PC = (self.PC + offs) & 0xFFFF; return 12
                return 7
            elif op == 0x29:
                hl = self._get_hl()
                self._set_hl(self._do_add16(hl, hl)); return 11
            elif op == 0x2A: self._set_hl(self._read16(self._fetch16())); return 16
            elif op == 0x2B: self._set_hl((self._get_hl() - 1) & 0xFFFF); return 6
            elif op == 0x2C: self._set_r(5, self._do_inc8(self._get_r(5))); return 4
            elif op == 0x2D: self._set_r(5, self._do_dec8(self._get_r(5))); return 4
            elif op == 0x2E: self._set_r(5, self._fetch()); return 7
            elif op == 0x2F:  # CPL
                self.A ^= 0xFF
                self.F = ((self.F & (S_FLAG | Z_FLAG | PV_FLAG | C_FLAG))
                          | H_FLAG | N_FLAG | (self.A & (Y_FLAG | X_FLAG)))
                return 4
            elif op == 0x30:  # JR NC
                offs = self._fetch()
                if offs >= 128: offs -= 256
                if not (self.F & C_FLAG):
                    self.PC = (self.PC + offs) & 0xFFFF; return 12
                return 7
            elif op == 0x31: self.SP = self._fetch16(); return 10
            elif op == 0x32: self.bus.write(self._fetch16(), self.A); return 13
            elif op == 0x33: self.SP = (self.SP + 1) & 0xFFFF; return 6
            elif op == 0x34:  # INC (HL) / INC (IX+d)
                if self._dd or self._fd:
                    addr = self._idx_addr
                    self.bus.write(addr, self._do_inc8(self.bus.read(addr)))
                    return 19
                addr = (self.H << 8) | self.L
                self.bus.write(addr, self._do_inc8(self.bus.read(addr)))
                return 11
            elif op == 0x35:  # DEC (HL) / DEC (IX+d)
                if self._dd or self._fd:
                    addr = self._idx_addr
                    self.bus.write(addr, self._do_dec8(self.bus.read(addr)))
                    return 19
                addr = (self.H << 8) | self.L
                self.bus.write(addr, self._do_dec8(self.bus.read(addr)))
                return 11
            elif op == 0x36:  # LD (HL),n / LD (IX+d),n
                if self._dd or self._fd:
                    self.bus.write(self._idx_addr, self._fetch()); return 15
                self.bus.write((self.H << 8) | self.L, self._fetch()); return 10
            elif op == 0x37:  # SCF
                self.F = ((self.F & (S_FLAG | Z_FLAG | PV_FLAG))
                          | C_FLAG | (self.A & (Y_FLAG | X_FLAG)))
                return 4
            elif op == 0x38:  # JR C
                offs = self._fetch()
                if offs >= 128: offs -= 256
                if self.F & C_FLAG:
                    self.PC = (self.PC + offs) & 0xFFFF; return 12
                return 7
            elif op == 0x39: self._set_hl(self._do_add16(self._get_hl(), self.SP)); return 11
            elif op == 0x3A: self.A = self.bus.read(self._fetch16()); return 13
            elif op == 0x3B: self.SP = (self.SP - 1) & 0xFFFF; return 6
            elif op == 0x3C: self.A = self._do_inc8(self.A); return 4
            elif op == 0x3D: self.A = self._do_dec8(self.A); return 4
            elif op == 0x3E: self.A = self._fetch(); return 7
            else:  # 0x3F CCF
                old_c = self.F & C_FLAG
                self.F = ((self.F & (S_FLAG | Z_FLAG | PV_FLAG))
                          | (H_FLAG if old_c else 0)
                          | (0 if old_c else C_FLAG)
                          | (self.A & (Y_FLAG | X_FLAG)))
                return 4

        # ------------------------------------------------------------------
        # 0x40 – 0x7F  LD r, r'  (0x76 = HALT)
        # ------------------------------------------------------------------
        elif op < 0x80:
            if op == 0x76:  # HALT
                self.halted = True
                self.PC = (self.PC - 1) & 0xFFFF
                return 4
            dst = (op >> 3) & 7
            src = op & 7
            # DD/FD + (HL) memory access: H/L on the register side stay as H/L
            if (self._dd or self._fd) and (dst == 6 or src == 6):
                if src == 6:
                    v = self.bus.read(self._idx_addr)
                    if dst == 4: self.H = v
                    elif dst == 5: self.L = v
                    else: self._set_r(dst, v)
                else:
                    if src == 4: v = self.H
                    elif src == 5: v = self.L
                    else: v = self._get_r(src)
                    self.bus.write(self._idx_addr, v)
                return 15   # 15 + 4 prefix = 19 total
            self._set_r(dst, self._get_r(src))
            if dst == 6 or src == 6: return 7
            return 4

        # ------------------------------------------------------------------
        # 0x80 – 0xBF  ALU A, r
        # ------------------------------------------------------------------
        elif op < 0xC0:
            alu_op = (op >> 3) & 7
            src = op & 7
            v = self._get_r(src)
            if alu_op == 0: self.A = self._do_add8(self.A, v, 0)
            elif alu_op == 1: self.A = self._do_add8(self.A, v, 1 if (self.F & C_FLAG) else 0)
            elif alu_op == 2: self.A = self._do_sub8(self.A, v, 0)
            elif alu_op == 3: self.A = self._do_sub8(self.A, v, 1 if (self.F & C_FLAG) else 0)
            elif alu_op == 4: self._do_and8(v)
            elif alu_op == 5: self._do_xor8(v)
            elif alu_op == 6: self._do_or8(v)
            else: self._do_cp8(v)
            if (self._dd or self._fd) and src == 6: return 15  # +4 = 19 total
            if src == 6: return 7
            return 4

        # ------------------------------------------------------------------
        # 0xC0 – 0xFF  control flow
        # ------------------------------------------------------------------
        else:
            if op == 0xC0:  # RET NZ
                if not (self.F & Z_FLAG): self.PC = self._pop(); return 11
                return 5
            elif op == 0xC1:
                v = self._pop(); self.B = (v >> 8) & 0xFF; self.C = v & 0xFF; return 10
            elif op == 0xC2:  # JP NZ
                nn = self._fetch16()
                if not (self.F & Z_FLAG): self.PC = nn
                return 10
            elif op == 0xC3: self.PC = self._fetch16(); return 10   # JP nn
            elif op == 0xC4:  # CALL NZ
                nn = self._fetch16()
                if not (self.F & Z_FLAG): self._push(self.PC); self.PC = nn; return 17
                return 10
            elif op == 0xC5: self._push((self.B << 8) | self.C); return 11
            elif op == 0xC6: self.A = self._do_add8(self.A, self._fetch(), 0); return 7
            elif op == 0xC7: self._push(self.PC); self.PC = 0x0000; return 11
            elif op == 0xC8:  # RET Z
                if self.F & Z_FLAG: self.PC = self._pop(); return 11
                return 5
            elif op == 0xC9: self.PC = self._pop(); return 10   # RET
            elif op == 0xCA:  # JP Z
                nn = self._fetch16()
                if self.F & Z_FLAG: self.PC = nn
                return 10
            elif op == 0xCB: return 4   # handled in step()
            elif op == 0xCC:  # CALL Z
                nn = self._fetch16()
                if self.F & Z_FLAG: self._push(self.PC); self.PC = nn; return 17
                return 10
            elif op == 0xCD:  # CALL nn
                nn = self._fetch16(); self._push(self.PC); self.PC = nn; return 17
            elif op == 0xCE:
                self.A = self._do_add8(self.A, self._fetch(), 1 if (self.F & C_FLAG) else 0)
                return 7
            elif op == 0xCF: self._push(self.PC); self.PC = 0x0008; return 11
            elif op == 0xD0:  # RET NC
                if not (self.F & C_FLAG): self.PC = self._pop(); return 11
                return 5
            elif op == 0xD1:
                v = self._pop(); self.D = (v >> 8) & 0xFF; self.E = v & 0xFF; return 10
            elif op == 0xD2:  # JP NC
                nn = self._fetch16()
                if not (self.F & C_FLAG): self.PC = nn
                return 10
            elif op == 0xD3: self.ports.write(self._fetch(), self.A); return 11
            elif op == 0xD4:  # CALL NC
                nn = self._fetch16()
                if not (self.F & C_FLAG): self._push(self.PC); self.PC = nn; return 17
                return 10
            elif op == 0xD5: self._push((self.D << 8) | self.E); return 11
            elif op == 0xD6: self.A = self._do_sub8(self.A, self._fetch(), 0); return 7
            elif op == 0xD7: self._push(self.PC); self.PC = 0x0010; return 11
            elif op == 0xD8:  # RET C
                if self.F & C_FLAG: self.PC = self._pop(); return 11
                return 5
            elif op == 0xD9:  # EXX
                self.B, self.B_ = self.B_, self.B
                self.C, self.C_ = self.C_, self.C
                self.D, self.D_ = self.D_, self.D
                self.E, self.E_ = self.E_, self.E
                self.H, self.H_ = self.H_, self.H
                self.L, self.L_ = self.L_, self.L
                return 4
            elif op == 0xDA:  # JP C
                nn = self._fetch16()
                if self.F & C_FLAG: self.PC = nn
                return 10
            elif op == 0xDB: self.A = self.ports.read(self._fetch()); return 11
            elif op == 0xDC:  # CALL C
                nn = self._fetch16()
                if self.F & C_FLAG: self._push(self.PC); self.PC = nn; return 17
                return 10
            elif op == 0xDD: return 4   # handled in step()
            elif op == 0xDE:
                self.A = self._do_sub8(self.A, self._fetch(), 1 if (self.F & C_FLAG) else 0)
                return 7
            elif op == 0xDF: self._push(self.PC); self.PC = 0x0018; return 11
            elif op == 0xE0:  # RET PO (PV clear)
                if not (self.F & PV_FLAG): self.PC = self._pop(); return 11
                return 5
            elif op == 0xE1:
                self._set_hl(self._pop()); return 10
            elif op == 0xE2:  # JP PO
                nn = self._fetch16()
                if not (self.F & PV_FLAG): self.PC = nn
                return 10
            elif op == 0xE3:  # EX (SP), HL
                t = self._read16(self.SP)
                self._write16(self.SP, self._get_hl())
                self._set_hl(t)
                return 19
            elif op == 0xE4:  # CALL PO
                nn = self._fetch16()
                if not (self.F & PV_FLAG): self._push(self.PC); self.PC = nn; return 17
                return 10
            elif op == 0xE5: self._push(self._get_hl()); return 11
            elif op == 0xE6: self._do_and8(self._fetch()); return 7
            elif op == 0xE7: self._push(self.PC); self.PC = 0x0020; return 11
            elif op == 0xE8:  # RET PE (PV set)
                if self.F & PV_FLAG: self.PC = self._pop(); return 11
                return 5
            elif op == 0xE9: self.PC = self._get_hl(); return 4   # JP (HL)
            elif op == 0xEA:  # JP PE
                nn = self._fetch16()
                if self.F & PV_FLAG: self.PC = nn
                return 10
            elif op == 0xEB:  # EX DE, HL
                self.D, self.H = self.H, self.D
                self.E, self.L = self.L, self.E
                return 4
            elif op == 0xEC:  # CALL PE
                nn = self._fetch16()
                if self.F & PV_FLAG: self._push(self.PC); self.PC = nn; return 17
                return 10
            elif op == 0xED: return 4   # handled in step()
            elif op == 0xEE: self._do_xor8(self._fetch()); return 7
            elif op == 0xEF: self._push(self.PC); self.PC = 0x0028; return 11
            elif op == 0xF0:  # RET P (S clear)
                if not (self.F & S_FLAG): self.PC = self._pop(); return 11
                return 5
            elif op == 0xF1:
                v = self._pop(); self.A = (v >> 8) & 0xFF; self.F = v & 0xFF; return 10
            elif op == 0xF2:  # JP P
                nn = self._fetch16()
                if not (self.F & S_FLAG): self.PC = nn
                return 10
            elif op == 0xF3: self.IFF1 = self.IFF2 = False; return 4   # DI
            elif op == 0xF4:  # CALL P
                nn = self._fetch16()
                if not (self.F & S_FLAG): self._push(self.PC); self.PC = nn; return 17
                return 10
            elif op == 0xF5: self._push((self.A << 8) | self.F); return 11
            elif op == 0xF6: self._do_or8(self._fetch()); return 7
            elif op == 0xF7: self._push(self.PC); self.PC = 0x0030; return 11
            elif op == 0xF8:  # RET M (S set)
                if self.F & S_FLAG: self.PC = self._pop(); return 11
                return 5
            elif op == 0xF9: self.SP = self._get_hl(); return 6   # LD SP,HL
            elif op == 0xFA:  # JP M
                nn = self._fetch16()
                if self.F & S_FLAG: self.PC = nn
                return 10
            elif op == 0xFB:  # EI
                self.IFF1 = self.IFF2 = True
                self._ei_delay = True
                return 4
            elif op == 0xFC:  # CALL M
                nn = self._fetch16()
                if self.F & S_FLAG: self._push(self.PC); self.PC = nn; return 17
                return 10
            elif op == 0xFD: return 4   # handled in step()
            elif op == 0xFE: self._do_cp8(self._fetch()); return 7
            else:  # 0xFF RST 38H
                self._push(self.PC); self.PC = 0x0038; return 11

    # -------------------------------------------------------------------------
    # Public step / run_cycles interface
    # -------------------------------------------------------------------------
    def step(self) -> int:
        """Execute one instruction; return T-states consumed."""
        cdef int c
        c = self._step()
        return c

    cdef int _step(self):
        cdef int c, op, vec

        # NMI — highest priority
        if self._nmi_pending:
            self._nmi_pending = False
            self.halted = False
            self.IFF2 = self.IFF1
            self.IFF1 = False
            self._push(self.PC)
            self.PC = 0x0066
            self.cycles += 11
            return 11

        # Maskable interrupt
        if self._int_pending and self.IFF1 and not self._ei_delay:
            self._int_pending = False
            self.halted = False
            self.IFF1 = self.IFF2 = False
            if self.IM == 1:
                self._push(self.PC)
                self.PC = 0x0038
                self.cycles += 13
                return 13
            elif self.IM == 2:
                self._push(self.PC)
                vec = (self.I << 8) | 0xFF
                self.PC = self._read16(vec)
                self.cycles += 19
                return 19
            else:
                self._push(self.PC)
                self.PC = 0x0038
                self.cycles += 13
                return 13

        self._ei_delay = False

        if self.halted:
            self.cycles += 4
            return 4

        op = self._fetch()

        if op == 0xCB:
            c = self._step_cb()
            self.cycles += c
            return c

        if op == 0xED:
            c = self._step_ed()
            self.cycles += c
            return c

        if op == 0xDD:
            c = self._step_indexed(1)
            self.cycles += c
            return c

        if op == 0xFD:
            c = self._step_indexed(0)
            self.cycles += c
            return c

        c = self._step_main(op)
        self.cycles += c
        return c

    cpdef int run_cycles(self, int target):
        """Run until at least *target* T-states have been consumed.
        Returns actual cycles executed this call."""
        cdef int spent = 0
        while spent < target:
            spent += self._step()
        return spent

    # -------------------------------------------------------------------------
    # Interrupt interface
    # -------------------------------------------------------------------------
    def request_interrupt(self):
        self._int_pending = True

    def request_nmi(self):
        self._nmi_pending = True

    # -------------------------------------------------------------------------
    # Reset / state
    # -------------------------------------------------------------------------
    def reset(self):
        self.A = self.F = 0xFF
        self.B = self.C = self.D = self.E = self.H = self.L = 0xFF
        self.IX = self.IY = 0xFFFF
        self.SP = 0xFFFF
        self.PC = 0
        self.I = self.R = 0
        self.IFF1 = self.IFF2 = False
        self.IM = 1
        self.halted = False
        self._int_pending = self._nmi_pending = self._ei_delay = False

    def get_state(self) -> dict:
        return {
            'A': self.A,  'F': self.F,  'B': self.B,  'C': self.C,
            'D': self.D,  'E': self.E,  'H': self.H,  'L': self.L,
            'A_': self.A_, 'F_': self.F_, 'B_': self.B_, 'C_': self.C_,
            'D_': self.D_, 'E_': self.E_, 'H_': self.H_, 'L_': self.L_,
            'IX': self.IX, 'IY': self.IY, 'SP': self.SP, 'PC': self.PC,
            'I': self.I,   'R': self.R,
            'IM': self.IM,
            'IFF1': self.IFF1, 'IFF2': self.IFF2,
            'halted': self.halted,
            '_int_pending': self._int_pending,
            '_nmi_pending': self._nmi_pending,
            '_ei_delay': self._ei_delay,
            'cycles': self.cycles,
        }

    def set_state(self, s: dict) -> None:
        self.A  = s['A'];  self.F  = s['F']
        self.B  = s['B'];  self.C  = s['C']
        self.D  = s['D'];  self.E  = s['E']
        self.H  = s['H'];  self.L  = s['L']
        self.A_ = s['A_']; self.F_ = s['F_']
        self.B_ = s['B_']; self.C_ = s['C_']
        self.D_ = s['D_']; self.E_ = s['E_']
        self.H_ = s['H_']; self.L_ = s['L_']
        self.IX = s['IX']; self.IY = s['IY']
        self.SP = s['SP']; self.PC = s['PC']
        self.I  = s['I'];  self.R  = s['R']
        self.IM = s['IM']
        self.IFF1 = s['IFF1']; self.IFF2 = s['IFF2']
        self.halted       = s['halted']
        self._int_pending = s['_int_pending']
        self._nmi_pending = s['_nmi_pending']
        self._ei_delay    = s['_ei_delay']
        self.cycles       = s['cycles']
        self._dd = self._fd = 0
