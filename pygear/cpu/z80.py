"""Zilog Z80 CPU — full standard Z80 (not Game Boy variant).

Registers
---------
Main : A F B C D E H L  (+ shadow set A' F' B' C' D' E' H' L')
16-bit pairs : BC DE HL AF  (properties, composed from 8-bit regs)
Index : IX IY
Control : SP PC I R
Interrupt : IFF1 IFF2 IM (modes 0/1/2)

Opcode dispatch
---------------
step() fetches one instruction (handling CB/ED/DD/FD prefixes) and
returns the number of T-states consumed.

run_cycles(n) calls step() until at least n T-states have elapsed.

DD/FD prefix
------------
_execute_indexed(is_dd) is called after DD or FD is consumed.  It:
  1. Fetches the next opcode byte.
  2. For opcodes in _NEEDS_DISP, fetches the signed displacement and
     stores the effective address in self._idx_addr.
  3. Sets self._dd / self._fd so opcode handlers can redirect HL/H/L
     accesses to IX/IY and (IX+d)/(IY+d).
  4. Adds 4 T-states overhead for the prefix byte.
"""

from .opcodes import build_all_tables
from .opcodes_cb import build_cb_table, build_ddcb_table

# Opcodes (un-prefixed) that access (HL) as a memory operand when used
# with a DD/FD prefix — these require a displacement byte to be fetched
# *before* any additional immediate operand bytes.
_NEEDS_DISP: frozenset = frozenset({
    0x34, 0x35, 0x36,                               # INC/DEC/LD (HL),n
    0x46, 0x4E, 0x56, 0x5E, 0x66, 0x6E,            # LD r,(HL)
    0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x77,      # LD (HL),r
    0x7E,                                            # LD A,(HL)
    0x86, 0x8E, 0x96, 0x9E,                         # ADD/ADC/SUB/SBC (HL)
    0xA6, 0xAE, 0xB6, 0xBE,                         # AND/XOR/OR/CP (HL)
})


class Z80:
    # ------------------------------------------------------------------
    def __init__(self, bus, ports):
        self.bus   = bus
        self.ports = ports

        # 8-bit main registers
        self.A = self.F = 0xFF
        self.B = self.C = 0xFF
        self.D = self.E = 0xFF
        self.H = self.L = 0xFF

        # 8-bit shadow registers
        self.A_ = self.F_ = 0x00
        self.B_ = self.C_ = 0x00
        self.D_ = self.E_ = 0x00
        self.H_ = self.L_ = 0x00

        # 16-bit index / control
        self.IX = 0xFFFF
        self.IY = 0xFFFF
        self.SP = 0xFFFF
        self.PC = 0x0000

        # Special purpose
        self.I  = 0x00
        self.R  = 0x00

        # Interrupt state
        self.IFF1 = False
        self.IFF2 = False
        self.IM   = 1        # GG games mostly use IM 1
        self.halted = False
        self._int_pending = False
        self._nmi_pending = False
        self._ei_delay    = False  # EI delays interrupt acceptance by 1 insn

        # DD/FD prefix state (set in _execute_indexed, cleared after)
        self._dd       = False
        self._fd       = False
        self._idx_addr = 0x0000

        # Cycle accumulator
        self.cycles = 0

        # Build dispatch tables
        self._main, self._ed = build_all_tables(self)
        self._cb   = build_cb_table(self)
        self._ddcb = build_ddcb_table(self)

    # ------------------------------------------------------------------
    # 16-bit register-pair properties
    # ------------------------------------------------------------------
    @property
    def BC(self):        return (self.B << 8) | self.C
    @BC.setter
    def BC(self, v):     self.B = (v >> 8) & 0xFF; self.C = v & 0xFF

    @property
    def DE(self):        return (self.D << 8) | self.E
    @DE.setter
    def DE(self, v):     self.D = (v >> 8) & 0xFF; self.E = v & 0xFF

    @property
    def HL(self):        return (self.H << 8) | self.L
    @HL.setter
    def HL(self, v):     self.H = (v >> 8) & 0xFF; self.L = v & 0xFF

    @property
    def AF(self):        return (self.A << 8) | self.F
    @AF.setter
    def AF(self, v):     self.A = (v >> 8) & 0xFF; self.F = v & 0xFF

    # ------------------------------------------------------------------
    # Memory / fetch helpers
    # ------------------------------------------------------------------
    def _fetch(self) -> int:
        """Fetch one byte from PC and advance PC.  Increments R (approximate)."""
        v = self.bus.read(self.PC)
        self.PC = (self.PC + 1) & 0xFFFF
        self.R  = (self.R  + 1) & 0x7F
        return v

    def _fetch16(self) -> int:
        lo = self._fetch()
        hi = self._fetch()
        return (hi << 8) | lo

    def _read16(self, addr: int) -> int:
        lo = self.bus.read(addr & 0xFFFF)
        hi = self.bus.read((addr + 1) & 0xFFFF)
        return (hi << 8) | lo

    def _write16(self, addr: int, v: int):
        self.bus.write(addr & 0xFFFF,        v & 0xFF)
        self.bus.write((addr + 1) & 0xFFFF, (v >> 8) & 0xFF)

    def _push(self, v: int):
        self.SP = (self.SP - 1) & 0xFFFF
        self.bus.write(self.SP, (v >> 8) & 0xFF)
        self.SP = (self.SP - 1) & 0xFFFF
        self.bus.write(self.SP, v & 0xFF)

    def _pop(self) -> int:
        lo = self.bus.read(self.SP);  self.SP = (self.SP + 1) & 0xFFFF
        hi = self.bus.read(self.SP);  self.SP = (self.SP + 1) & 0xFFFF
        return (hi << 8) | lo

    # ------------------------------------------------------------------
    # Interrupt interface
    # ------------------------------------------------------------------
    def request_interrupt(self):
        self._int_pending = True

    def request_nmi(self):
        self._nmi_pending = True

    # ------------------------------------------------------------------
    # Main execution entry points
    # ------------------------------------------------------------------
    def step(self) -> int:
        """Execute one instruction; return T-states consumed."""

        # NMI — highest priority, not maskable
        if self._nmi_pending:
            self._nmi_pending = False
            self.halted       = False
            self.IFF2         = self.IFF1
            self.IFF1         = False
            self._push(self.PC)
            self.PC = 0x0066
            self.cycles += 11
            return 11

        # Maskable interrupt
        if self._int_pending and self.IFF1 and not self._ei_delay:
            self._int_pending = False
            self.halted       = False
            self.IFF1 = self.IFF2 = False
            if self.IM == 1:
                self._push(self.PC)
                self.PC = 0x0038
                self.cycles += 13
                return 13
            elif self.IM == 2:
                self._push(self.PC)
                vec     = (self.I << 8) | 0xFF
                self.PC = self._read16(vec)
                self.cycles += 19
                return 19
            else:   # IM 0 — treat as RST 38h
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
            op2 = self._fetch()
            c = self._cb[op2]()
            self.cycles += c
            return c

        if op == 0xED:
            op2 = self._fetch()
            handler = self._ed.get(op2)
            c = handler() if handler else 8   # unknown ED ops: NOP-like
            self.cycles += c
            return c

        if op == 0xDD:
            c = self._execute_indexed(True)
            self.cycles += c
            return c

        if op == 0xFD:
            c = self._execute_indexed(False)
            self.cycles += c
            return c

        c = self._main[op]()
        self.cycles += c
        return c

    def run_cycles(self, target: int) -> int:
        """Run until at least *target* T-states have been consumed.
        Returns actual cycles executed this call."""
        spent = 0
        while spent < target:
            spent += self.step()
        return spent

    # ------------------------------------------------------------------
    # DD / FD prefix dispatcher
    # ------------------------------------------------------------------
    def _execute_indexed(self, is_dd: bool) -> int:
        """Dispatch a DD- or FD-prefixed instruction.

        Returns total T-states including the 4-cycle prefix overhead.
        Handlers for (IX+d)/(IY+d) memory ops return (dd_cycles - 4) so
        that adding 4 here gives the correct total.
        """
        op = self._fetch()

        # DDCB / FDCB — displacement then CB opcode
        if op == 0xCB:
            d    = self._fetch()
            if d >= 0x80: d -= 0x100
            base = self.IX if is_dd else self.IY
            self._idx_addr = (base + d) & 0xFFFF
            op2  = self._fetch()
            c    = self._ddcb[op2]()
            return c   # DDCB table already returns the full count (23 / 20)

        # Pre-fetch displacement for instructions that address (IX+d)/(IY+d)
        if op in _NEEDS_DISP:
            d    = self._fetch()
            if d >= 0x80: d -= 0x100
            base = self.IX if is_dd else self.IY
            self._idx_addr = (base + d) & 0xFFFF

        self._dd = is_dd
        self._fd = not is_dd
        try:
            c = self._main[op]()
        finally:
            self._dd = False
            self._fd = False

        return c + 4   # +4 for the prefix byte

    # ------------------------------------------------------------------
    def reset(self):
        self.A = self.F = 0xFF
        self.B = self.C = self.D = self.E = self.H = self.L = 0xFF
        self.IX = self.IY = 0xFFFF
        self.SP = 0xFFFF
        self.PC = 0x0000
        self.I  = self.R = 0
        self.IFF1 = self.IFF2 = False
        self.IM = 1
        self.halted = False
        self._int_pending = self._nmi_pending = self._ei_delay = False
        self._dd = self._fd = False
        self.cycles = 0
