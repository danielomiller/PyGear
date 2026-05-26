"""Z80 disassembler — decodes bytes at a given address to mnemonic strings."""

_R    = ('B', 'C', 'D', 'E', 'H', 'L', '(HL)', 'A')
_RP   = ('BC', 'DE', 'HL', 'SP')
_RP2  = ('BC', 'DE', 'HL', 'AF')   # PUSH/POP
_ALU  = ('ADD A,', 'ADC A,', 'SUB ', 'SBC A,', 'AND ', 'XOR ', 'OR ', 'CP ')
_ROT  = ('RLC', 'RRC', 'RL', 'RR', 'SLA', 'SRA', 'SLL', 'SRL')
_CC   = ('NZ', 'Z', 'NC', 'C', 'PO', 'PE', 'P', 'M')

# Opcodes that need a displacement byte in DD/FD context (mirrors z80.py _NEEDS_DISP)
_NEEDS_DISP = frozenset({
    0x34, 0x35, 0x36,
    0x46, 0x4E, 0x56, 0x5E, 0x66, 0x6E,
    0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x77,
    0x7E,
    0x86, 0x8E, 0x96, 0x9E,
    0xA6, 0xAE, 0xB6, 0xBE,
})


def _disp(d: int) -> str:
    """Format an unsigned displacement byte as a signed offset string."""
    if d == 0:    return ''
    if d < 0x80:  return f'+${d:02X}'
    return f'-${0x100 - d:02X}'


def disasm(read_byte, pc: int) -> tuple[str, int]:
    """Disassemble one Z80 instruction at *pc*.

    *read_byte* is callable(addr: int) -> int (0–255).
    Returns (mnemonic: str, length_in_bytes: int).
    """
    def b(o: int) -> int:
        return read_byte((pc + o) & 0xFFFF) & 0xFF

    def w(o: int) -> int:
        return b(o) | (b(o + 1) << 8)

    def rel(o: int) -> int:
        """Decode signed relative offset at position o → absolute target."""
        d = b(o)
        return (pc + o + 1 + (d - 0x100 if d >= 0x80 else d)) & 0xFFFF

    op = b(0)

    # -----------------------------------------------------------------------
    # CB prefix
    # -----------------------------------------------------------------------
    if op == 0xCB:
        o2 = b(1)
        r  = _R[o2 & 7]
        if   o2 < 0x40: return f'{_ROT[o2 >> 3]} {r}',       2
        elif o2 < 0x80: return f'BIT {(o2>>3)&7},{r}',        2
        elif o2 < 0xC0: return f'RES {(o2>>3)&7},{r}',        2
        else:           return f'SET {(o2>>3)&7},{r}',        2

    # -----------------------------------------------------------------------
    # ED prefix
    # -----------------------------------------------------------------------
    if op == 0xED:
        o2 = b(1)
        if (o2 & 0xC7) == 0x40:   # IN r,(C)
            r = _R[(o2 >> 3) & 7]; r = 'F' if r == '(HL)' else r
            return f'IN {r},(C)', 2
        if (o2 & 0xC7) == 0x41:   # OUT (C),r
            r = _R[(o2 >> 3) & 7]; r = '0' if r == '(HL)' else r
            return f'OUT (C),{r}', 2
        if (o2 & 0xCF) == 0x42: return f'SBC HL,{_RP[(o2>>4)&3]}', 2
        if (o2 & 0xCF) == 0x4A: return f'ADC HL,{_RP[(o2>>4)&3]}', 2
        if (o2 & 0xCF) == 0x43:
            return f'LD (${w(2):04X}),{_RP[(o2>>4)&3]}', 4
        if (o2 & 0xCF) == 0x4B:
            return f'LD {_RP[(o2>>4)&3]},(${w(2):04X})', 4
        _ED = {
            0x44:'NEG',   0x4C:'NEG',   0x54:'NEG',   0x5C:'NEG',
            0x64:'NEG',   0x6C:'NEG',   0x74:'NEG',   0x7C:'NEG',
            0x45:'RETN',  0x55:'RETN',  0x65:'RETN',  0x75:'RETN',
            0x5D:'RETN',  0x6D:'RETN',  0x7D:'RETN',
            0x4D:'RETI',
            0x47:'LD I,A', 0x4F:'LD R,A',
            0x57:'LD A,I', 0x5F:'LD A,R',
            0x67:'RRD',    0x6F:'RLD',
            0xA0:'LDI',   0xA1:'CPI',   0xA2:'INI',   0xA3:'OUTI',
            0xA8:'LDD',   0xA9:'CPD',   0xAA:'IND',   0xAB:'OUTD',
            0xB0:'LDIR',  0xB1:'CPIR',  0xB2:'INIR',  0xB3:'OTIR',
            0xB8:'LDDR',  0xB9:'CPDR',  0xBA:'INDR',  0xBB:'OTDR',
        }
        if o2 in _ED: return _ED[o2], 2
        if o2 in (0x46, 0x4E, 0x66, 0x6E): return 'IM 0', 2
        if o2 in (0x56, 0x76):             return 'IM 1', 2
        if o2 in (0x5E, 0x7E):             return 'IM 2', 2
        return f'DB $ED,${o2:02X}', 2

    # -----------------------------------------------------------------------
    # DD / FD prefix
    # -----------------------------------------------------------------------
    if op in (0xDD, 0xFD):
        idx = 'IX' if op == 0xDD else 'IY'
        ih  = idx + 'H'
        il  = idx + 'L'
        o2  = b(1)

        # DDCB / FDCB
        if o2 == 0xCB:
            d   = b(2)
            o3  = b(3)
            tgt = f'({idx}{_disp(d)})'
            dst = _R[o3 & 7]
            if   o3 < 0x40:
                nm = f'{_ROT[o3>>3]} {tgt}'
                if dst != '(HL)': nm += f',{dst}'
            elif o3 < 0x80:
                nm = f'BIT {(o3>>3)&7},{tgt}'
            elif o3 < 0xC0:
                nm = f'RES {(o3>>3)&7},{tgt}'
                if dst != '(HL)': nm += f',{dst}'
            else:
                nm = f'SET {(o3>>3)&7},{tgt}'
                if dst != '(HL)': nm += f',{dst}'
            return nm, 4

        # Instructions that reference (IX+d) — displacement byte at offset 2
        if o2 in _NEEDS_DISP:
            d    = b(2)
            tgt  = f'({idx}{_disp(d)})'
            base = 3   # prefix + op + disp consumed so far
            if o2 == 0x36:            # LD (IX+d),n
                return f'LD {tgt},${b(3):02X}', 4
            if o2 == 0x34:            # INC (IX+d)
                return f'INC {tgt}', 3
            if o2 == 0x35:            # DEC (IX+d)
                return f'DEC {tgt}', 3
            if 0x40 <= o2 <= 0x7F:
                if (o2 & 7) == 6:     # LD r,(IX+d)  — dst is a real reg, H/L unchanged
                    dst = _R[(o2 >> 3) & 7]
                    return f'LD {dst},{tgt}', 3
                else:                  # LD (IX+d),r  — src is a real reg, H/L unchanged
                    src = _R[o2 & 7]
                    return f'LD {tgt},{src}', 3
            # ALU (IX+d)
            return f'{_ALU[(o2 >> 3) & 7]}{tgt}', 3

        # Non-indexed DD/FD instructions (plain HL/H/L substitution)
        # 16-bit load / add
        if o2 == 0x21: return f'LD {idx},${w(2):04X}', 4
        if o2 == 0x22: return f'LD (${w(2):04X}),{idx}', 4
        if o2 == 0x2A: return f'LD {idx},(${w(2):04X})', 4
        if o2 == 0x23: return f'INC {idx}', 2
        if o2 == 0x2B: return f'DEC {idx}', 2
        if (o2 & 0xCF) == 0x09:
            rr = _RP[(o2 >> 4) & 3]
            if rr == 'HL': rr = idx
            return f'ADD {idx},{rr}', 2
        # IXH / IXL
        if o2 == 0x24: return f'INC {ih}', 2
        if o2 == 0x25: return f'DEC {ih}', 2
        if o2 == 0x26: return f'LD {ih},${b(2):02X}', 3
        if o2 == 0x2C: return f'INC {il}', 2
        if o2 == 0x2D: return f'DEC {il}', 2
        if o2 == 0x2E: return f'LD {il},${b(2):02X}', 3
        # Stack / jump
        if o2 == 0xE1: return f'POP {idx}', 2
        if o2 == 0xE5: return f'PUSH {idx}', 2
        if o2 == 0xE3: return f'EX (SP),{idx}', 2
        if o2 == 0xE9: return f'JP ({idx})', 2
        if o2 == 0xF9: return f'LD SP,{idx}', 2
        # HALT
        if o2 == 0x76: return 'HALT', 2
        # LD r,r' — substitute H/L
        if 0x40 <= o2 <= 0x7F:
            dst = _R[(o2 >> 3) & 7]
            src = _R[o2 & 7]
            if dst == 'H': dst = ih
            elif dst == 'L': dst = il
            if src == 'H': src = ih
            elif src == 'L': src = il
            return f'LD {dst},{src}', 2
        # ALU r — substitute H/L
        if 0x80 <= o2 <= 0xBF:
            src = _R[o2 & 7]
            if src == 'H': src = ih
            elif src == 'L': src = il
            return f'{_ALU[(o2 >> 3) & 7]}{src}', 2
        # INC / DEC r — substitute H/L
        if (o2 & 0xC7) == 0x04:
            r = _R[(o2 >> 3) & 7]
            if r == 'H': r = ih
            elif r == 'L': r = il
            return f'INC {r}', 2
        if (o2 & 0xC7) == 0x05:
            r = _R[(o2 >> 3) & 7]
            if r == 'H': r = ih
            elif r == 'L': r = il
            return f'DEC {r}', 2
        # LD r,n — substitute H/L
        if (o2 & 0xC7) == 0x06:
            r = _R[(o2 >> 3) & 7]
            if r == 'H': r = ih
            elif r == 'L': r = il
            return f'LD {r},${b(2):02X}', 3
        return f'DB ${op:02X},${o2:02X}', 2

    # -----------------------------------------------------------------------
    # Unprefixed opcodes
    # -----------------------------------------------------------------------
    # Group 00 — miscellaneous
    if op < 0x40:
        if (op & 0xC7) == 0x06:   # LD r,n
            return f'LD {_R[(op>>3)&7]},${b(1):02X}', 2
        if (op & 0xC7) == 0x04:   # INC r
            return f'INC {_R[(op>>3)&7]}', 1
        if (op & 0xC7) == 0x05:   # DEC r
            return f'DEC {_R[(op>>3)&7]}', 1
        if (op & 0xCF) == 0x01:   # LD rr,nn
            return f'LD {_RP[(op>>4)&3]},${w(1):04X}', 3
        if (op & 0xCF) == 0x09:   # ADD HL,rr
            return f'ADD HL,{_RP[(op>>4)&3]}', 1
        if (op & 0xCF) == 0x03:   # INC rr
            return f'INC {_RP[(op>>4)&3]}', 1
        if (op & 0xCF) == 0x0B:   # DEC rr
            return f'DEC {_RP[(op>>4)&3]}', 1
        if op == 0x00: return 'NOP', 1
        if op == 0x02: return 'LD (BC),A', 1
        if op == 0x07: return 'RLCA', 1
        if op == 0x08: return "EX AF,AF'", 1
        if op == 0x0A: return 'LD A,(BC)', 1
        if op == 0x0F: return 'RRCA', 1
        if op == 0x10: return f'DJNZ ${rel(1):04X}', 2
        if op == 0x12: return 'LD (DE),A', 1
        if op == 0x17: return 'RLA', 1
        if op == 0x18: return f'JR ${rel(1):04X}', 2
        if op == 0x1A: return 'LD A,(DE)', 1
        if op == 0x1F: return 'RRA', 1
        if op in (0x20, 0x28, 0x30, 0x38):   # JR cc,e
            cc = ('NZ', 'Z', 'NC', 'C')[(op >> 3) & 3]
            return f'JR {cc},${rel(1):04X}', 2
        if op == 0x22: return f'LD (${w(1):04X}),HL', 3
        if op == 0x27: return 'DAA', 1
        if op == 0x2A: return f'LD HL,(${w(1):04X})', 3
        if op == 0x2F: return 'CPL', 1
        if op == 0x32: return f'LD (${w(1):04X}),A', 3
        if op == 0x36: return f'LD (HL),${b(1):02X}', 2
        if op == 0x37: return 'SCF', 1
        if op == 0x3A: return f'LD A,(${w(1):04X})', 3
        if op == 0x3E: return f'LD A,${b(1):02X}', 2
        if op == 0x3F: return 'CCF', 1

    # Group 01 — LD r,r'
    elif 0x40 <= op <= 0x7F:
        if op == 0x76: return 'HALT', 1
        return f'LD {_R[(op>>3)&7]},{_R[op&7]}', 1

    # Group 10 — ALU r
    elif 0x80 <= op <= 0xBF:
        return f'{_ALU[(op>>3)&7]}{_R[op&7]}', 1

    # Group 11 — control / stack / misc
    else:
        if (op & 0xC7) == 0xC0:   # RET cc
            return f'RET {_CC[(op>>3)&7]}', 1
        if (op & 0xCF) == 0xC1:   # POP rr
            return f'POP {_RP2[(op>>4)&3]}', 1
        if (op & 0xCF) == 0xC5:   # PUSH rr
            return f'PUSH {_RP2[(op>>4)&3]}', 1
        if (op & 0xC7) == 0xC2:   # JP cc,nn
            return f'JP {_CC[(op>>3)&7]},${w(1):04X}', 3
        if (op & 0xC7) == 0xC4:   # CALL cc,nn
            return f'CALL {_CC[(op>>3)&7]},${w(1):04X}', 3
        if (op & 0xC7) == 0xC6:   # ALU n
            return f'{_ALU[(op>>3)&7]}${b(1):02X}', 2
        if (op & 0xC7) == 0xC7:   # RST
            return f'RST ${op&0x38:02X}H', 1
        if op == 0xC3: return f'JP ${w(1):04X}', 3
        if op == 0xC9: return 'RET', 1
        if op == 0xCD: return f'CALL ${w(1):04X}', 3
        if op == 0xD3: return f'OUT (${b(1):02X}),A', 2
        if op == 0xD9: return 'EXX', 1
        if op == 0xDB: return f'IN A,(${b(1):02X})', 2
        if op == 0xE3: return 'EX (SP),HL', 1
        if op == 0xE9: return 'JP (HL)', 1
        if op == 0xEB: return 'EX DE,HL', 1
        if op == 0xF3: return 'DI', 1
        if op == 0xF9: return 'LD SP,HL', 1
        if op == 0xFB: return 'EI', 1

    return f'DB ${op:02X}', 1


def disasm_block(read_byte, start: int, count: int) -> list[tuple[int, str, int]]:
    """Disassemble *count* instructions starting at *start*.

    Returns a list of (address, mnemonic, length) tuples.
    """
    results = []
    addr = start
    for _ in range(count):
        mnem, length = disasm(read_byte, addr)
        results.append((addr, mnem, length))
        addr = (addr + length) & 0xFFFF
    return results
