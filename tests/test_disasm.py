"""Tests for the Z80 disassembler (pygear.cpu.disasm)."""

import pytest
from pygear.cpu.disasm import disasm, disasm_block


def dis(bytelist):
    """Helper: disassemble from a list of bytes starting at address 0."""
    mem = bytearray(bytelist + [0] * 4)   # padding so reads past end don't raise
    return disasm(lambda a: mem[a & 0xFFFF], 0)


# ---------------------------------------------------------------------------
# Unprefixed opcodes
# ---------------------------------------------------------------------------

class TestUnprefixed:

    def test_nop(self):
        assert dis([0x00]) == ('NOP', 1)

    def test_halt(self):
        assert dis([0x76]) == ('HALT', 1)

    def test_ld_r_r(self):
        # LD B,C  (0x41)
        assert dis([0x41]) == ('LD B,C', 1)
        # LD A,H  (0x7C)
        assert dis([0x7C]) == ('LD A,H', 1)

    def test_ld_r_hl(self):
        # LD D,(HL)  (0x56)
        assert dis([0x56]) == ('LD D,(HL)', 1)

    def test_ld_hl_r(self):
        # LD (HL),E  (0x73)
        assert dis([0x73]) == ('LD (HL),E', 1)

    def test_ld_r_n(self):
        # LD B,$FF
        assert dis([0x06, 0xFF]) == ('LD B,$FF', 2)
        # LD A,$42
        assert dis([0x3E, 0x42]) == ('LD A,$42', 2)

    def test_ld_hl_n(self):
        # LD (HL),$7F
        assert dis([0x36, 0x7F]) == ('LD (HL),$7F', 2)

    def test_ld_rr_nn(self):
        assert dis([0x01, 0x34, 0x12]) == ('LD BC,$1234', 3)
        assert dis([0x11, 0x00, 0x80]) == ('LD DE,$8000', 3)
        assert dis([0x21, 0xFF, 0xFF]) == ('LD HL,$FFFF', 3)
        assert dis([0x31, 0xF0, 0xFF]) == ('LD SP,$FFF0', 3)

    def test_ld_a_bc_de(self):
        assert dis([0x0A]) == ('LD A,(BC)', 1)
        assert dis([0x1A]) == ('LD A,(DE)', 1)

    def test_ld_bc_de_a(self):
        assert dis([0x02]) == ('LD (BC),A', 1)
        assert dis([0x12]) == ('LD (DE),A', 1)

    def test_ld_nn_a(self):
        assert dis([0x32, 0x00, 0xC0]) == ('LD ($C000),A', 3)

    def test_ld_a_nn(self):
        assert dis([0x3A, 0x00, 0xC0]) == ('LD A,($C000)', 3)

    def test_ld_nn_hl(self):
        assert dis([0x22, 0x34, 0x12]) == ('LD ($1234),HL', 3)

    def test_ld_hl_nn(self):
        assert dis([0x2A, 0x34, 0x12]) == ('LD HL,($1234)', 3)

    def test_ld_sp_hl(self):
        assert dis([0xF9]) == ('LD SP,HL', 1)

    def test_inc_dec_r(self):
        assert dis([0x04]) == ('INC B', 1)
        assert dis([0x3D]) == ('DEC A', 1)
        assert dis([0x34]) == ('INC (HL)', 1)
        assert dis([0x35]) == ('DEC (HL)', 1)

    def test_inc_dec_rr(self):
        assert dis([0x03]) == ('INC BC', 1)
        assert dis([0x1B]) == ('DEC DE', 1)
        assert dis([0x23]) == ('INC HL', 1)
        assert dis([0x33]) == ('INC SP', 1)

    def test_add_hl_rr(self):
        assert dis([0x09]) == ('ADD HL,BC', 1)
        assert dis([0x29]) == ('ADD HL,HL', 1)
        assert dis([0x39]) == ('ADD HL,SP', 1)

    def test_alu_r(self):
        assert dis([0x80]) == ('ADD A,B', 1)
        assert dis([0x88]) == ('ADC A,B', 1)
        assert dis([0x90]) == ('SUB B', 1)
        assert dis([0x98]) == ('SBC A,B', 1)
        assert dis([0xA0]) == ('AND B', 1)
        assert dis([0xA8]) == ('XOR B', 1)
        assert dis([0xB0]) == ('OR B', 1)
        assert dis([0xB8]) == ('CP B', 1)

    def test_alu_hl(self):
        assert dis([0x86]) == ('ADD A,(HL)', 1)
        assert dis([0xBE]) == ('CP (HL)', 1)

    def test_alu_n(self):
        assert dis([0xC6, 0x01]) == ('ADD A,$01', 2)
        assert dis([0xFE, 0xFF]) == ('CP $FF', 2)
        assert dis([0xE6, 0x0F]) == ('AND $0F', 2)

    def test_jr(self):
        mem = bytearray([0x18, 0x00] + [0] * 4)
        # JR 0: PC=0, offset=0, target = 0 + 2 + 0 = 2
        assert disasm(lambda a: mem[a], 0) == ('JR $0002', 2)

    def test_jr_backward(self):
        # Offset $FE = -2 signed → target = PC + 2 + (-2) = PC
        mem = bytearray([0x18, 0xFE] + [0] * 4)
        assert disasm(lambda a: mem[a], 0) == ('JR $0000', 2)

    def test_jr_cc(self):
        mem = bytearray([0x20, 0x05] + [0] * 8)  # JR NZ,$0007
        assert disasm(lambda a: mem[a], 0) == ('JR NZ,$0007', 2)

    def test_djnz(self):
        mem = bytearray([0x10, 0x03] + [0] * 4)  # DJNZ $0005
        assert disasm(lambda a: mem[a], 0) == ('DJNZ $0005', 2)

    def test_jp(self):
        assert dis([0xC3, 0x66, 0x00]) == ('JP $0066', 3)

    def test_jp_cc(self):
        assert dis([0xC2, 0x00, 0x40]) == ('JP NZ,$4000', 3)
        assert dis([0xCA, 0x00, 0x40]) == ('JP Z,$4000', 3)

    def test_jp_hl(self):
        assert dis([0xE9]) == ('JP (HL)', 1)

    def test_call(self):
        assert dis([0xCD, 0x00, 0x01]) == ('CALL $0100', 3)
        assert dis([0xCC, 0x00, 0x01]) == ('CALL Z,$0100', 3)

    def test_ret(self):
        assert dis([0xC9]) == ('RET', 1)
        assert dis([0xC0]) == ('RET NZ', 1)
        assert dis([0xC8]) == ('RET Z', 1)

    def test_rst(self):
        assert dis([0xC7]) == ('RST $00H', 1)
        assert dis([0xCF]) == ('RST $08H', 1)
        assert dis([0xFF]) == ('RST $38H', 1)

    def test_push_pop(self):
        assert dis([0xC5]) == ('PUSH BC', 1)
        assert dis([0xF5]) == ('PUSH AF', 1)
        assert dis([0xC1]) == ('POP BC', 1)
        assert dis([0xF1]) == ('POP AF', 1)

    def test_misc(self):
        assert dis([0x07]) == ('RLCA', 1)
        assert dis([0x0F]) == ('RRCA', 1)
        assert dis([0x17]) == ('RLA', 1)
        assert dis([0x1F]) == ('RRA', 1)
        assert dis([0x27]) == ('DAA', 1)
        assert dis([0x2F]) == ('CPL', 1)
        assert dis([0x37]) == ('SCF', 1)
        assert dis([0x3F]) == ('CCF', 1)
        assert dis([0x08]) == ("EX AF,AF'", 1)
        assert dis([0xD9]) == ('EXX', 1)
        assert dis([0xEB]) == ('EX DE,HL', 1)
        assert dis([0xE3]) == ('EX (SP),HL', 1)
        assert dis([0xF3]) == ('DI', 1)
        assert dis([0xFB]) == ('EI', 1)

    def test_in_out(self):
        assert dis([0xD3, 0x10]) == ('OUT ($10),A', 2)
        assert dis([0xDB, 0x10]) == ('IN A,($10)', 2)


# ---------------------------------------------------------------------------
# CB prefix
# ---------------------------------------------------------------------------

class TestCBPrefix:

    def test_rlc(self):
        assert dis([0xCB, 0x00]) == ('RLC B', 2)
        assert dis([0xCB, 0x07]) == ('RLC A', 2)

    def test_rrc(self):
        assert dis([0xCB, 0x08]) == ('RRC B', 2)

    def test_rl(self):
        assert dis([0xCB, 0x10]) == ('RL B', 2)

    def test_rr(self):
        assert dis([0xCB, 0x18]) == ('RR B', 2)

    def test_sla(self):
        assert dis([0xCB, 0x20]) == ('SLA B', 2)

    def test_sra(self):
        assert dis([0xCB, 0x28]) == ('SRA B', 2)

    def test_sll_undoc(self):
        assert dis([0xCB, 0x30]) == ('SLL B', 2)

    def test_srl(self):
        assert dis([0xCB, 0x38]) == ('SRL B', 2)

    def test_rotate_hl(self):
        assert dis([0xCB, 0x06]) == ('RLC (HL)', 2)
        assert dis([0xCB, 0x3E]) == ('SRL (HL)', 2)

    def test_bit(self):
        assert dis([0xCB, 0x40]) == ('BIT 0,B', 2)
        assert dis([0xCB, 0x7F]) == ('BIT 7,A', 2)
        assert dis([0xCB, 0x46]) == ('BIT 0,(HL)', 2)

    def test_res(self):
        assert dis([0xCB, 0x80]) == ('RES 0,B', 2)
        assert dis([0xCB, 0xBE]) == ('RES 7,(HL)', 2)

    def test_set(self):
        assert dis([0xCB, 0xC0]) == ('SET 0,B', 2)
        assert dis([0xCB, 0xFF]) == ('SET 7,A', 2)


# ---------------------------------------------------------------------------
# ED prefix
# ---------------------------------------------------------------------------

class TestEDPrefix:

    def test_in_r_c(self):
        assert dis([0xED, 0x40]) == ('IN B,(C)', 2)
        assert dis([0xED, 0x78]) == ('IN A,(C)', 2)
        assert dis([0xED, 0x70]) == ('IN F,(C)', 2)   # undoc: discard

    def test_out_c_r(self):
        assert dis([0xED, 0x41]) == ('OUT (C),B', 2)
        assert dis([0xED, 0x79]) == ('OUT (C),A', 2)

    def test_sbc_adc_hl(self):
        assert dis([0xED, 0x42]) == ('SBC HL,BC', 2)
        assert dis([0xED, 0x4A]) == ('ADC HL,BC', 2)
        assert dis([0xED, 0x72]) == ('SBC HL,SP', 2)

    def test_ld_nn_rr(self):
        assert dis([0xED, 0x43, 0x00, 0xC0]) == ('LD ($C000),BC', 4)
        assert dis([0xED, 0x73, 0x10, 0xFF]) == ('LD ($FF10),SP', 4)

    def test_ld_rr_nn(self):
        assert dis([0xED, 0x4B, 0x00, 0xC0]) == ('LD BC,($C000)', 4)

    def test_neg(self):
        assert dis([0xED, 0x44]) == ('NEG', 2)
        assert dis([0xED, 0x4C]) == ('NEG', 2)   # mirror

    def test_retn_reti(self):
        assert dis([0xED, 0x45]) == ('RETN', 2)
        assert dis([0xED, 0x4D]) == ('RETI', 2)
        assert dis([0xED, 0x5D]) == ('RETN', 2)  # mirror

    def test_ld_i_r(self):
        assert dis([0xED, 0x47]) == ('LD I,A', 2)
        assert dis([0xED, 0x4F]) == ('LD R,A', 2)
        assert dis([0xED, 0x57]) == ('LD A,I', 2)
        assert dis([0xED, 0x5F]) == ('LD A,R', 2)

    def test_im(self):
        assert dis([0xED, 0x46]) == ('IM 0', 2)
        assert dis([0xED, 0x56]) == ('IM 1', 2)
        assert dis([0xED, 0x5E]) == ('IM 2', 2)

    def test_rld_rrd(self):
        assert dis([0xED, 0x67]) == ('RRD', 2)
        assert dis([0xED, 0x6F]) == ('RLD', 2)

    def test_block_transfer(self):
        assert dis([0xED, 0xA0]) == ('LDI', 2)
        assert dis([0xED, 0xB0]) == ('LDIR', 2)
        assert dis([0xED, 0xA8]) == ('LDD', 2)
        assert dis([0xED, 0xB8]) == ('LDDR', 2)

    def test_block_search(self):
        assert dis([0xED, 0xA1]) == ('CPI', 2)
        assert dis([0xED, 0xB1]) == ('CPIR', 2)
        assert dis([0xED, 0xA9]) == ('CPD', 2)
        assert dis([0xED, 0xB9]) == ('CPDR', 2)

    def test_block_io(self):
        assert dis([0xED, 0xA2]) == ('INI', 2)
        assert dis([0xED, 0xB2]) == ('INIR', 2)
        assert dis([0xED, 0xAA]) == ('IND', 2)
        assert dis([0xED, 0xBA]) == ('INDR', 2)
        assert dis([0xED, 0xA3]) == ('OUTI', 2)
        assert dis([0xED, 0xB3]) == ('OTIR', 2)
        assert dis([0xED, 0xAB]) == ('OUTD', 2)
        assert dis([0xED, 0xBB]) == ('OTDR', 2)


# ---------------------------------------------------------------------------
# DD prefix (IX)
# ---------------------------------------------------------------------------

class TestDDPrefix:

    def test_ld_ix_nn(self):
        assert dis([0xDD, 0x21, 0x34, 0x12]) == ('LD IX,$1234', 4)

    def test_ld_nn_ix(self):
        assert dis([0xDD, 0x22, 0x00, 0xC0]) == ('LD ($C000),IX', 4)

    def test_ld_ix_nn_indirect(self):
        assert dis([0xDD, 0x2A, 0x00, 0xC0]) == ('LD IX,($C000)', 4)

    def test_inc_dec_ix(self):
        assert dis([0xDD, 0x23]) == ('INC IX', 2)
        assert dis([0xDD, 0x2B]) == ('DEC IX', 2)

    def test_inc_dec_ixhl(self):
        assert dis([0xDD, 0x24]) == ('INC IXH', 2)
        assert dis([0xDD, 0x25]) == ('DEC IXH', 2)
        assert dis([0xDD, 0x2C]) == ('INC IXL', 2)
        assert dis([0xDD, 0x2D]) == ('DEC IXL', 2)

    def test_ld_ixh_n(self):
        assert dis([0xDD, 0x26, 0xAB]) == ('LD IXH,$AB', 3)
        assert dis([0xDD, 0x2E, 0xCD]) == ('LD IXL,$CD', 3)

    def test_add_ix_rr(self):
        assert dis([0xDD, 0x09]) == ('ADD IX,BC', 2)
        assert dis([0xDD, 0x29]) == ('ADD IX,IX', 2)
        assert dis([0xDD, 0x39]) == ('ADD IX,SP', 2)

    def test_ld_r_ix_d(self):
        # LD B,(IX+$05)
        assert dis([0xDD, 0x46, 0x05]) == ('LD B,(IX+$05)', 3)
        # LD A,(IX+$00)
        assert dis([0xDD, 0x7E, 0x00]) == ('LD A,(IX)', 3)
        # LD H,(IX-$01)  — H stays H (not IXH)
        assert dis([0xDD, 0x66, 0xFF]) == ('LD H,(IX-$01)', 3)

    def test_ld_ix_d_r(self):
        # LD (IX+$03),B
        assert dis([0xDD, 0x70, 0x03]) == ('LD (IX+$03),B', 3)
        # LD (IX+$00),A
        assert dis([0xDD, 0x77, 0x00]) == ('LD (IX),A', 3)
        # LD (IX+$01),H  — H stays H (not IXH)
        assert dis([0xDD, 0x74, 0x01]) == ('LD (IX+$01),H', 3)

    def test_ld_ix_d_n(self):
        assert dis([0xDD, 0x36, 0x02, 0xFF]) == ('LD (IX+$02),$FF', 4)

    def test_inc_dec_ix_d(self):
        assert dis([0xDD, 0x34, 0x05]) == ('INC (IX+$05)', 3)
        assert dis([0xDD, 0x35, 0x80]) == ('DEC (IX-$80)', 3)

    def test_alu_ix_d(self):
        assert dis([0xDD, 0x86, 0x01]) == ('ADD A,(IX+$01)', 3)
        assert dis([0xDD, 0xBE, 0x00]) == ('CP (IX)', 3)
        assert dis([0xDD, 0xA6, 0xFF]) == ('AND (IX-$01)', 3)

    def test_ld_r_ixhl(self):
        # LD B,IXH
        assert dis([0xDD, 0x44]) == ('LD B,IXH', 2)
        # LD C,IXL
        assert dis([0xDD, 0x4D]) == ('LD C,IXL', 2)
        # LD IXH,A
        assert dis([0xDD, 0x67]) == ('LD IXH,A', 2)

    def test_alu_ixhl(self):
        assert dis([0xDD, 0x84]) == ('ADD A,IXH', 2)
        assert dis([0xDD, 0xBC]) == ('CP IXH', 2)
        assert dis([0xDD, 0xBD]) == ('CP IXL', 2)

    def test_push_pop_ix(self):
        assert dis([0xDD, 0xE5]) == ('PUSH IX', 2)
        assert dis([0xDD, 0xE1]) == ('POP IX', 2)

    def test_jp_ix(self):
        assert dis([0xDD, 0xE9]) == ('JP (IX)', 2)

    def test_ld_sp_ix(self):
        assert dis([0xDD, 0xF9]) == ('LD SP,IX', 2)

    def test_ex_sp_ix(self):
        assert dis([0xDD, 0xE3]) == ('EX (SP),IX', 2)

    def test_halt_in_dd_context(self):
        assert dis([0xDD, 0x76]) == ('HALT', 2)


# ---------------------------------------------------------------------------
# FD prefix (IY) — spot-check symmetry with DD
# ---------------------------------------------------------------------------

class TestFDPrefix:

    def test_ld_iy_nn(self):
        assert dis([0xFD, 0x21, 0x00, 0x80]) == ('LD IY,$8000', 4)

    def test_ld_r_iy_d(self):
        assert dis([0xFD, 0x4E, 0x10]) == ('LD C,(IY+$10)', 3)

    def test_alu_iy_d(self):
        assert dis([0xFD, 0x96, 0x02]) == ('SUB (IY+$02)', 3)

    def test_iyh_iyl(self):
        assert dis([0xFD, 0x84]) == ('ADD A,IYH', 2)
        assert dis([0xFD, 0x85]) == ('ADD A,IYL', 2)
        assert dis([0xFD, 0x24]) == ('INC IYH', 2)
        assert dis([0xFD, 0x2D]) == ('DEC IYL', 2)


# ---------------------------------------------------------------------------
# DDCB prefix
# ---------------------------------------------------------------------------

class TestDDCBPrefix:

    def test_bit_ix_d(self):
        assert dis([0xDD, 0xCB, 0x05, 0x46]) == ('BIT 0,(IX+$05)', 4)
        assert dis([0xDD, 0xCB, 0x00, 0x7E]) == ('BIT 7,(IX)', 4)

    def test_res_ix_d(self):
        assert dis([0xDD, 0xCB, 0x00, 0x86]) == ('RES 0,(IX)', 4)

    def test_set_ix_d(self):
        assert dis([0xDD, 0xCB, 0x01, 0xC6]) == ('SET 0,(IX+$01)', 4)

    def test_rot_ix_d(self):
        assert dis([0xDD, 0xCB, 0x00, 0x06]) == ('RLC (IX)', 4)
        assert dis([0xDD, 0xCB, 0xFF, 0x3E]) == ('SRL (IX-$01)', 4)

    def test_fdcb(self):
        assert dis([0xFD, 0xCB, 0x00, 0x46]) == ('BIT 0,(IY)', 4)


# ---------------------------------------------------------------------------
# disasm_block helper
# ---------------------------------------------------------------------------

class TestDisasmBlock:

    def test_block_lengths(self):
        mem = bytearray([0x00, 0x3E, 0xFF, 0xC9])  # NOP, LD A,$FF, RET
        result = disasm_block(lambda a: mem[a & 0xFFFF], 0, 3)
        assert len(result) == 3
        assert result[0] == (0, 'NOP', 1)
        assert result[1] == (1, 'LD A,$FF', 2)
        assert result[2] == (3, 'RET', 1)

    def test_block_advances_address(self):
        mem = bytearray([0x01, 0x34, 0x12, 0x00])  # LD BC,$1234, NOP
        result = disasm_block(lambda a: mem[a & 0xFFFF], 0, 2)
        assert result[0][0] == 0   # LD BC at addr 0
        assert result[1][0] == 3   # NOP at addr 3
