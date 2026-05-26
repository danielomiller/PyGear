"""Interactive Z80 debugger for PyGear.

Usage:
    python -m pygear.debugger roms/game.gg

Or via the main emulator:
    python main.py roms/game.gg --debug

Commands
--------
  s [N]           step N instructions (default 1)
  c               continue until breakpoint (Ctrl+C to interrupt)
  b ADDR          add breakpoint at hex address
  del ADDR        delete breakpoint
  bl              list breakpoints
  regs            show all registers and flags
  dis [ADDR [N]]  disassemble N instructions (default 10) starting at ADDR
                  (default: current PC)
  x ADDR [N]      hex dump N bytes (default 16) at ADDR
  sp [N]          show top N stack entries (default 8)
  save [SLOT]     save emulator state to slot (default 0)
  load [SLOT]     load emulator state from slot (default 0)
  reset           reset console to power-on state
  q / quit        exit debugger
"""

import sys

try:
    import readline  # noqa: F401 — enables readline history/editing on Unix
except ImportError:
    pass

from .cartridge import Cartridge
from .console import GameGearConsole
from .cpu.disasm import disasm


class Debugger:
    """Wraps a GameGearConsole and provides an interactive debugging REPL."""

    def __init__(self, console: GameGearConsole) -> None:
        self.console     = console
        self.cpu         = console.cpu
        self.bus         = console.bus
        self._breakpoints: set[int] = set()

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _flag_str(self) -> str:
        F = self.cpu.F
        return (f"S={'1' if F & 0x80 else '0'} "
                f"Z={'1' if F & 0x40 else '0'} "
                f"Y={'1' if F & 0x20 else '0'} "
                f"H={'1' if F & 0x10 else '0'} "
                f"X={'1' if F & 0x08 else '0'} "
                f"PV={'1' if F & 0x04 else '0'} "
                f"N={'1' if F & 0x02 else '0'} "
                f"C={'1' if F & 0x01 else '0'}")

    def _regs_str(self) -> str:
        cpu = self.cpu
        af2 = (cpu.A_ << 8) | cpu.F_
        bc2 = (cpu.B_ << 8) | cpu.C_
        de2 = (cpu.D_ << 8) | cpu.E_
        hl2 = (cpu.H_ << 8) | cpu.L_
        return (
            f"  PC=${cpu.PC:04X}  SP=${cpu.SP:04X}  IX=${cpu.IX:04X}  IY=${cpu.IY:04X}\n"
            f"  AF=${cpu.AF:04X}  BC=${cpu.BC:04X}  DE=${cpu.DE:04X}  HL=${cpu.HL:04X}\n"
            f"  AF'=${af2:04X} BC'=${bc2:04X} DE'=${de2:04X} HL'=${hl2:04X}\n"
            f"  I=${cpu.I:02X}  R=${cpu.R:02X}  IM={cpu.IM}"
            f"  IFF1={int(cpu.IFF1)}  IFF2={int(cpu.IFF2)}"
            f"{'  HALT' if cpu.halted else ''}\n"
            f"  {self._flag_str()}"
        )

    def _disasm_str(self, addr: int, count: int = 10) -> str:
        lines = []
        cur = self.cpu.PC
        for _ in range(count):
            mnem, length = disasm(self.bus.read, addr)
            raw    = ' '.join(f'{self.bus.read((addr + i) & 0xFFFF):02X}'
                              for i in range(length))
            marker = '>' if addr == cur else ' '
            lines.append(f"  {marker} ${addr:04X}:  {raw:<12}  {mnem}")
            addr = (addr + length) & 0xFFFF
        return '\n'.join(lines)

    def _hex_str(self, addr: int, count: int = 16) -> str:
        lines = []
        for row in range(0, count, 16):
            a    = (addr + row) & 0xFFFF
            data = [self.bus.read((a + i) & 0xFFFF)
                    for i in range(min(16, count - row))]
            hex_part   = ' '.join(f'{v:02X}' for v in data)
            ascii_part = ''.join(chr(v) if 32 <= v < 127 else '.' for v in data)
            lines.append(f"  ${a:04X}: {hex_part:<48}  {ascii_part}")
        return '\n'.join(lines)

    def _stack_str(self, count: int = 8) -> str:
        lines = []
        sp = self.cpu.SP
        for i in range(count):
            a  = (sp + i * 2) & 0xFFFF
            lo = self.bus.read(a)
            hi = self.bus.read((a + 1) & 0xFFFF)
            offset = f'+{i*2}' if i else '  '
            lines.append(f"  [SP{offset:>3}]  ${a:04X}: ${(hi<<8|lo):04X}")
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def step(self, count: int = 1) -> bool:
        """Execute *count* instructions.  Returns True if a breakpoint was hit."""
        for _ in range(count):
            self.cpu.step()
            if self.cpu.PC in self._breakpoints:
                print(f"  Breakpoint at ${self.cpu.PC:04X}")
                return True
        return False

    def run(self) -> None:
        """Run until a breakpoint is hit or Ctrl+C is pressed."""
        print("  Running… (Ctrl+C to break)")
        n = 0
        try:
            while True:
                self.cpu.step()
                n += 1
                if self.cpu.PC in self._breakpoints:
                    print(f"  Breakpoint at ${self.cpu.PC:04X}  ({n:,} steps)")
                    break
        except KeyboardInterrupt:
            print(f"\n  Interrupted at ${self.cpu.PC:04X}  ({n:,} steps)")

    # ------------------------------------------------------------------
    # REPL
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_hex(s: str) -> int | None:
        try:
            return int(s.lstrip('$'), 16)
        except ValueError:
            print(f"  Invalid hex address: {s!r}")
            return None

    def _handle(self, cmd: str, args: list[str]) -> None:
        cpu = self.cpu

        # --- step ---
        if cmd in ('s', 'step', 'n', 'next'):
            n = int(args[0]) if args else 1
            self.step(n)
            print(self._regs_str())
            print()
            print(self._disasm_str(cpu.PC, 4))

        # --- continue ---
        elif cmd in ('c', 'cont', 'continue', 'run'):
            self.run()
            print(self._regs_str())
            print()
            print(self._disasm_str(cpu.PC, 4))

        # --- breakpoint set ---
        elif cmd in ('b', 'break', 'bp'):
            if not args:
                print("  Usage: b ADDR")
                return
            addr = self._parse_hex(args[0])
            if addr is not None:
                self._breakpoints.add(addr & 0xFFFF)
                print(f"  Breakpoint set at ${addr & 0xFFFF:04X}")

        # --- breakpoint delete ---
        elif cmd in ('del', 'delete', 'bd'):
            if not args:
                print("  Usage: del ADDR")
                return
            addr = self._parse_hex(args[0])
            if addr is not None:
                addr &= 0xFFFF
                if addr in self._breakpoints:
                    self._breakpoints.discard(addr)
                    print(f"  Removed breakpoint at ${addr:04X}")
                else:
                    print(f"  No breakpoint at ${addr:04X}")

        # --- breakpoint list ---
        elif cmd in ('bl', 'blist', 'breakpoints'):
            if self._breakpoints:
                for a in sorted(self._breakpoints):
                    print(f"  ${a:04X}")
            else:
                print("  No breakpoints set")

        # --- registers ---
        elif cmd in ('regs', 'reg', 'r', 'registers'):
            print(self._regs_str())

        # --- disassemble ---
        elif cmd in ('dis', 'd', 'disasm', 'u'):
            addr  = self._parse_hex(args[0]) if args else cpu.PC
            if addr is None: return
            count = int(args[1]) if len(args) > 1 else 10
            print(self._disasm_str(addr & 0xFFFF, count))

        # --- hex dump ---
        elif cmd in ('x', 'mem', 'dump'):
            if not args:
                print("  Usage: x ADDR [N]")
                return
            addr = self._parse_hex(args[0])
            if addr is None: return
            count = int(args[1]) if len(args) > 1 else 16
            print(self._hex_str(addr & 0xFFFF, count))

        # --- stack ---
        elif cmd in ('sp', 'stack'):
            count = int(args[0]) if args else 8
            print(self._stack_str(count))

        # --- save state ---
        elif cmd == 'save':
            slot = int(args[0]) if args else 0
            path = self.console.save_state(slot)
            print(f"  Saved to {path}")

        # --- load state ---
        elif cmd == 'load':
            slot = int(args[0]) if args else 0
            if self.console.load_state(slot):
                print(f"  Loaded slot {slot}")
                print(self._regs_str())
            else:
                print(f"  No state in slot {slot}")

        # --- reset ---
        elif cmd == 'reset':
            self.console.reset()
            print("  Console reset")
            print(self._regs_str())

        # --- help ---
        elif cmd in ('help', '?', 'h'):
            print(__doc__)

        # --- quit ---
        elif cmd in ('q', 'quit', 'exit'):
            raise SystemExit(0)

        else:
            print(f"  Unknown command: {cmd!r}  (type 'help' for list)")

    def repl(self) -> None:
        """Start the interactive debugger REPL."""
        print("\nPyGear Debugger  —  type 'help' for commands\n")
        print(self._regs_str())
        print()
        print(self._disasm_str(self.cpu.PC, 5))
        print()

        while True:
            try:
                line = input("(debug) ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            parts = line.split()
            try:
                self._handle(parts[0].lower(), parts[1:])
            except SystemExit:
                raise
            except Exception as exc:
                print(f"  Error: {exc}")


# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="PyGear Z80 Debugger")
    parser.add_argument("rom", help="path to .gg ROM file")
    args = parser.parse_args()

    try:
        cart = Cartridge(args.rom)
    except FileNotFoundError:
        print(f"ROM not found: {args.rom}", file=sys.stderr)
        sys.exit(1)

    console = GameGearConsole(cart)
    Debugger(console).repl()


if __name__ == "__main__":
    main()
