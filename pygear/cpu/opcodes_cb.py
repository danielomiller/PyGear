"""CB-prefix opcode table builder.

All 256 CB opcodes: rotates/shifts (0x00–0x3F), BIT (0x40–0x7F),
RES (0x80–0xBF), SET (0xC0–0xFF).

Each function closes over the CPU instance and returns T-state count.
"""

# ---------------------------------------------------------------------------
# Flag constants (must match z80.py)
S_FLAG  = 0x80
Z_FLAG  = 0x40
Y_FLAG  = 0x20
H_FLAG  = 0x10
X_FLAG  = 0x08
PV_FLAG = 0x04
N_FLAG  = 0x02
C_FLAG  = 0x01


def _parity(v: int) -> bool:
    v &= 0xFF
    v ^= v >> 4
    v ^= v >> 2
    v ^= v >> 1
    return (v & 1) == 0  # True = even parity


def build_cb_table(cpu):
    """Return list[256] of CB-prefix opcode handlers."""

    # -----------------------------------------------------------------------
    # Register accessor helpers (index 0–7 = B,C,D,E,H,L,(HL),A)
    def _get_r(idx: int) -> int:
        if   idx == 0: return cpu.B
        elif idx == 1: return cpu.C
        elif idx == 2: return cpu.D
        elif idx == 3: return cpu.E
        elif idx == 4: return cpu.H
        elif idx == 5: return cpu.L
        elif idx == 6: return cpu.bus.read(cpu.HL)
        else:          return cpu.A

    def _set_r(idx: int, v: int):
        v &= 0xFF
        if   idx == 0: cpu.B = v
        elif idx == 1: cpu.C = v
        elif idx == 2: cpu.D = v
        elif idx == 3: cpu.E = v
        elif idx == 4: cpu.H = v
        elif idx == 5: cpu.L = v
        elif idx == 6: cpu.bus.write(cpu.HL, v)
        else:          cpu.A = v

    # -----------------------------------------------------------------------
    def _sz_flags(v: int) -> int:
        f = 0
        if v & 0x80: f |= S_FLAG
        if v == 0:   f |= Z_FLAG
        f |= v & (Y_FLAG | X_FLAG)
        if _parity(v): f |= PV_FLAG
        return f

    # -----------------------------------------------------------------------
    # Shift/rotate implementations
    def _rlc(v: int) -> int:
        c = (v >> 7) & 1
        r = ((v << 1) | c) & 0xFF
        cpu.F = _sz_flags(r) | (C_FLAG if c else 0)
        return r

    def _rrc(v: int) -> int:
        c = v & 1
        r = ((v >> 1) | (c << 7)) & 0xFF
        cpu.F = _sz_flags(r) | (C_FLAG if c else 0)
        return r

    def _rl(v: int) -> int:
        old_c = 1 if (cpu.F & C_FLAG) else 0
        c = (v >> 7) & 1
        r = ((v << 1) | old_c) & 0xFF
        cpu.F = _sz_flags(r) | (C_FLAG if c else 0)
        return r

    def _rr(v: int) -> int:
        old_c = 1 if (cpu.F & C_FLAG) else 0
        c = v & 1
        r = ((v >> 1) | (old_c << 7)) & 0xFF
        cpu.F = _sz_flags(r) | (C_FLAG if c else 0)
        return r

    def _sla(v: int) -> int:
        c = (v >> 7) & 1
        r = (v << 1) & 0xFF
        cpu.F = _sz_flags(r) | (C_FLAG if c else 0)
        return r

    def _sra(v: int) -> int:
        c = v & 1
        r = ((v >> 1) | (v & 0x80)) & 0xFF
        cpu.F = _sz_flags(r) | (C_FLAG if c else 0)
        return r

    def _sll(v: int) -> int:  # undocumented SLL/SLS
        c = (v >> 7) & 1
        r = ((v << 1) | 1) & 0xFF
        cpu.F = _sz_flags(r) | (C_FLAG if c else 0)
        return r

    def _srl(v: int) -> int:
        c = v & 1
        r = (v >> 1) & 0xFF
        cpu.F = _sz_flags(r) | (C_FLAG if c else 0)
        return r

    _shift_ops = [_rlc, _rrc, _rl, _rr, _sla, _sra, _sll, _srl]

    # -----------------------------------------------------------------------
    table = [None] * 256

    # 0x00–0x3F: shift/rotate
    for code in range(0x40):
        op_idx = code >> 3   # which shift op
        r_idx  = code & 0x07  # which register

        def _make_shift(oi, ri):
            def handler():
                v = _get_r(ri)
                result = _shift_ops[oi](v)
                _set_r(ri, result)
                return 15 if ri == 6 else 8
            return handler

        table[code] = _make_shift(op_idx, r_idx)

    # 0x40–0x7F: BIT b, r
    for code in range(0x40, 0x80):
        bit   = (code >> 3) & 0x07
        r_idx = code & 0x07
        mask  = 1 << bit

        def _make_bit(b, ri, m):
            def handler():
                v = _get_r(ri)
                # BIT flags: Z=~(v&mask), S=bit7set, H=1, P=Z, N=0, C unchanged
                tested = v & m
                f = cpu.F & C_FLAG
                f |= H_FLAG
                if not tested:
                    f |= Z_FLAG | PV_FLAG
                if tested & S_FLAG:
                    f |= S_FLAG
                # undocumented Y/X come from the tested byte when reg != (HL)
                if ri != 6:
                    f |= v & (Y_FLAG | X_FLAG)
                else:
                    # For (HL), Y/X come from high byte of address
                    f |= (cpu.H) & (Y_FLAG | X_FLAG)
                cpu.F = f
                return 12 if ri == 6 else 8
            return handler

        table[code] = _make_bit(bit, r_idx, mask)

    # 0x80–0xBF: RES b, r
    for code in range(0x80, 0xC0):
        bit   = (code >> 3) & 0x07
        r_idx = code & 0x07
        mask  = ~(1 << bit) & 0xFF

        def _make_res(ri, m):
            def handler():
                _set_r(ri, _get_r(ri) & m)
                return 15 if ri == 6 else 8
            return handler

        table[code] = _make_res(r_idx, mask)

    # 0xC0–0xFF: SET b, r
    for code in range(0xC0, 0x100):
        bit   = (code >> 3) & 0x07
        r_idx = code & 0x07
        mask  = 1 << bit

        def _make_set(ri, m):
            def handler():
                _set_r(ri, _get_r(ri) | m)
                return 15 if ri == 6 else 8
            return handler

        table[code] = _make_set(r_idx, mask)

    return table


# ---------------------------------------------------------------------------
def build_ddcb_table(cpu):
    """Return list[256] of DDCB/FDCB opcode handlers.

    Called as build_ddcb_table(cpu); the caller must supply eff_addr
    (the IX+d or IY+d effective address) at call time via cpu._idx_addr.
    Each handler reads/writes cpu._idx_addr.
    """

    def _parity(v: int) -> bool:
        v &= 0xFF
        v ^= v >> 4
        v ^= v >> 2
        v ^= v >> 1
        return (v & 1) == 0

    def _sz_flags(v: int) -> int:
        f = 0
        if v & 0x80: f |= S_FLAG
        if v == 0:   f |= Z_FLAG
        f |= v & (Y_FLAG | X_FLAG)
        if _parity(v): f |= PV_FLAG
        return f

    table = [None] * 256

    # 0x00–0x3F: shift/rotate on (idx+d); result optionally written to r
    def _rlc_m(v):
        c = (v >> 7) & 1
        r = ((v << 1) | c) & 0xFF
        cpu.F = _sz_flags(r) | (C_FLAG if c else 0)
        return r
    def _rrc_m(v):
        c = v & 1
        r = ((v >> 1) | (c << 7)) & 0xFF
        cpu.F = _sz_flags(r) | (C_FLAG if c else 0)
        return r
    def _rl_m(v):
        old_c = 1 if (cpu.F & C_FLAG) else 0
        c = (v >> 7) & 1
        r = ((v << 1) | old_c) & 0xFF
        cpu.F = _sz_flags(r) | (C_FLAG if c else 0)
        return r
    def _rr_m(v):
        old_c = 1 if (cpu.F & C_FLAG) else 0
        c = v & 1
        r = ((v >> 1) | (old_c << 7)) & 0xFF
        cpu.F = _sz_flags(r) | (C_FLAG if c else 0)
        return r
    def _sla_m(v):
        c = (v >> 7) & 1
        r = (v << 1) & 0xFF
        cpu.F = _sz_flags(r) | (C_FLAG if c else 0)
        return r
    def _sra_m(v):
        c = v & 1
        r = ((v >> 1) | (v & 0x80)) & 0xFF
        cpu.F = _sz_flags(r) | (C_FLAG if c else 0)
        return r
    def _sll_m(v):
        c = (v >> 7) & 1
        r = ((v << 1) | 1) & 0xFF
        cpu.F = _sz_flags(r) | (C_FLAG if c else 0)
        return r
    def _srl_m(v):
        c = v & 1
        r = (v >> 1) & 0xFF
        cpu.F = _sz_flags(r) | (C_FLAG if c else 0)
        return r

    _shift_fns = [_rlc_m, _rrc_m, _rl_m, _rr_m, _sla_m, _sra_m, _sll_m, _srl_m]

    # helper to optionally also write result to a register
    def _reg_set(idx, v):
        v &= 0xFF
        if   idx == 0: cpu.B = v
        elif idx == 1: cpu.C = v
        elif idx == 2: cpu.D = v
        elif idx == 3: cpu.E = v
        elif idx == 4: cpu.H = v
        elif idx == 5: cpu.L = v
        elif idx == 7: cpu.A = v
        # idx == 6: no extra register write

    for code in range(0x40):
        op_idx = code >> 3
        r_idx  = code & 0x07

        def _make(oi, ri):
            def handler():
                addr = cpu._idx_addr
                v = cpu.bus.read(addr)
                result = _shift_fns[oi](v)
                cpu.bus.write(addr, result)
                _reg_set(ri, result)
                return 23
            return handler

        table[code] = _make(op_idx, r_idx)

    # 0x40–0x7F: BIT b, (idx+d)
    for code in range(0x40, 0x80):
        bit  = (code >> 3) & 0x07
        mask = 1 << bit

        def _make_bit(m):
            def handler():
                v = cpu.bus.read(cpu._idx_addr)
                tested = v & m
                f = cpu.F & C_FLAG
                f |= H_FLAG
                if not tested:
                    f |= Z_FLAG | PV_FLAG
                if tested & 0x80:
                    f |= S_FLAG
                # Y/X undocumented: from high byte of effective address
                f |= (cpu._idx_addr >> 8) & (Y_FLAG | X_FLAG)
                cpu.F = f
                return 20
            return handler

        table[code] = _make_bit(mask)

    # 0x80–0xBF: RES b, (idx+d)  [optionally also write to r]
    for code in range(0x80, 0xC0):
        bit  = (code >> 3) & 0x07
        r_idx = code & 0x07
        mask = ~(1 << bit) & 0xFF

        def _make_res(ri, m):
            def handler():
                addr = cpu._idx_addr
                result = cpu.bus.read(addr) & m
                cpu.bus.write(addr, result)
                _reg_set(ri, result)
                return 23
            return handler

        table[code] = _make_res(r_idx, mask)

    # 0xC0–0xFF: SET b, (idx+d)  [optionally also write to r]
    for code in range(0xC0, 0x100):
        bit  = (code >> 3) & 0x07
        r_idx = code & 0x07
        mask = 1 << bit

        def _make_set(ri, m):
            def handler():
                addr = cpu._idx_addr
                result = cpu.bus.read(addr) | m
                cpu.bus.write(addr, result)
                _reg_set(ri, result)
                return 23
            return handler

        table[code] = _make_set(r_idx, mask)

    return table
