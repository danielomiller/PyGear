# PyGear — Sega Game Gear Emulator

Most Game Gear emulators get the games running. PyGear also gets the hardware right.

Written from scratch in Python, PyGear targets the kind of accuracy that makes the
difference between "it kind of works" and "the hardware would agree." The Z80 core
handles every undocumented flag bit and prefix form. The VDP implements all three
R0 scroll locks and composites with correct priority. The PSG advances per-scanline
so that register writes land at the right sample — not retroactively at the end of
the frame. If you care about that kind of thing, read on.

---

## What makes PyGear different

### Z80: undocumented behaviour, correctly

The Z80's F register has two bits — XF (bit 3) and YF (bit 5) — that the official
Zilog documentation doesn't acknowledge. They're set by the CPU as side-effects of
nearly every instruction, and a surprising number of games and demos depend on them
indirectly. PyGear sets XF and YF correctly for the full instruction set, including
the block I/O group (INI, IND, OUTI, OUTD) where the rules are especially tricky.
The implementation follows Sean Young's Z80 Undocumented document exactly.

The `DDCB`/`FDCB` prefix forms — indexed shift, rotate, and bit operations like
`RLC (IX+d),B` — also write their result into the named register, not just to
memory. That's the undocumented behaviour documented in the same source, and
PyGear implements it. The disassembler knows the difference between
`RLC (IX+d)` (no result register) and `RLC (IX+d),B` (result register write),
and labels them correctly.

All prefix forms are covered: CB, ED, DD/FD, and DDCB/FDCB. The undocumented
`SLL` opcode (shift left, force bit 0 = 1) is present. The DD/FD forms expose
`IXH`, `IXL`, `IYH`, `IYL` as half-register operands with correct disambiguation
from the `(IX+d)` addressing mode.

### VDP: scroll locks and correct compositing

The Game Gear displays a 160×144 window cropped from the SMS VDP's 256×192 internal
render. PyGear renders the full internal frame and crops it, so the coordinate
geometry is exact.

The three R0 scroll-lock bits are all implemented:

- **Bit 5 — Left column blank**: masks the leftmost 8 pixels with the backdrop
  colour after compositing. This hides the wraparound artefact that horizontal
  scrolling produces at the left screen edge. Games that set this bit expect a
  clean visual seam; PyGear delivers it.
- **Bit 6 — H-scroll lock**: the top 16 lines ignore the horizontal scroll register.
  Used to keep a status bar pinned while the playfield scrolls underneath.
- **Bit 7 — V-scroll lock**: the rightmost 8 columns ignore the vertical scroll
  register, for the same reason in the vertical axis.

Sprite rendering supports up to 8 sprites per scanline with overflow and collision
flags. The sprite shift-left-8 mode (used to hide sprites off-screen left) is
implemented. Per-scanline compositing respects sprite-over-background priority and
the per-tile priority flag. The NTSC V-counter produces the non-linear jump at
line 219, which games rely on for VBlank timing.

### PSG: per-scanline audio with hardware-accurate noise coupling

The SN76489 has a flaw that most emulators paper over: register writes take effect
across the entire rendered frame, not at the moment they're written. PyGear calls
`render_cycles(228)` once per scanline alongside the CPU, so a volume or frequency
change applied on scanline 60 hits the audio buffer at the sample corresponding to
scanline 60 — roughly 63 µs precision rather than 16.7 ms per-frame precision. The
`_frac` accumulator carries the sub-sample remainder across scanline boundaries so
no cycles are lost.

The noise channel's NF=3 mode — where the LFSR clock is derived from Tone channel 2
— is implemented as hardware coupling, not an approximation. When NF=3, the LFSR
advances exactly as many times per sample as Tone 2's internal counter crosses zero.
This matches the physical chip, where Tone 2's output pin directly drives the noise
clock input. The naive approach (using Tone 2's period as a separate noise clock)
produces a subtly wrong pitch in effects that rely on this coupling.

The 16-bit LFSR uses the correct SMS/GG tap polynomial (bits 0 XOR 3 for white
noise, bit 0 only for periodic). The 16-level attenuation table is computed as
2 dB per step on a linear scale. The Game Gear stereo register (port 0x06) gives
independent left/right routing for each of the four channels.

---

## Feature summary

### CPU (Z80)
- Full documented instruction set including all ED, CB, DD/FD prefix forms
- DDCB/FDCB indexed shift/rotate/bit ops with undocumented result-register writes
- XF (bit 3) and YF (bit 5) flag bits correct for all instructions including block I/O
- Maskable interrupts (IM0, IM1, IM2) and NMI (Pause button)
- HALT state; full shadow register set (AF', BC', DE', HL')

### VDP
- 256×192 internal render, hardware-cropped to 160×144 for display
- 4bpp planar tile rendering (32 bytes/tile, 4 bitplanes)
- 512-tile name table with per-entry H-flip, V-flip, palette select, and priority
- 32-colour CRAM with 12-bit BGR colour (4 bits per channel)
- Sprite rendering: up to 8 per scanline, overflow/collision flags, shift-left-8 mode
- All three R0 scroll locks (left column blank, H-scroll lock, V-scroll lock)
- NTSC V-counter with non-linear jump at line 219
- Line interrupts (R0 bit 4 enable, R10 reload value) and VBlank interrupt

### PSG (SN76489)
- 3 square-wave tone channels with 10-bit period registers
- 2 dB-per-step attenuation table, 16 levels
- 16-bit LFSR noise: periodic and white-noise modes; correct SMS/GG tap polynomial
- NF=3: LFSR hardware-coupled to Tone 2's counter output, not approximated
- Per-scanline `render_cycles()` with fractional sample accumulation
- Game Gear stereo register (port 0x06): independent left/right routing per channel

### Memory and mappers
- 8 KB main RAM with correct $C000–$FFFF mirror
- Sega bank mapper: 3 × 16 KB ROM slots, control registers at $FFFC–$FFFF
- Codemasters mapper: auto-detected by checksum, bank registers at $0000/$4000/$8000
- Battery-backed SRAM save support (.sav files alongside ROM)
- BIOS ROM overlay: 1 KB at $0000–$03FF, auto-discovered or specified via `--bios`

### Save states
- Full console state serialised with Python pickle (CPU, VDP, PSG, memory, joypad)
- F5 to save, F8 to load; multiple slots via the debugger

### Test suite
- 1140+ tests covering Z80 opcodes, VDP rendering pipeline, PSG synthesis,
  memory bus, and save states

---

## Requirements

- Python 3.9+
- pygame 2.x
- numpy

```
pip install -r requirements.txt
```

---

## Running a ROM

```
python main.py roms/game.gg
```

Opens at 3× scale (480×432 window). Use `--scale N` for a different multiplier.

### BIOS

The Game Gear BIOS is auto-discovered from the ROM directory or `~/.pygear/`.
To specify a path explicitly:

```
python main.py roms/game.gg --bios path/to/bios.gg
```

To skip BIOS emulation entirely:

```
python main.py roms/game.gg --no-bios
```

---

## Controls

| Key            | Function                          |
|----------------|-----------------------------------|
| Arrow keys     | D-pad                             |
| Z              | Button 1                          |
| X              | Button 2                          |
| Enter          | Start                             |
| P / Pause      | Pause (fires NMI)                 |
| F5             | Save state                        |
| F8             | Load state                        |
| F12            | Screenshot                        |
| Escape         | Quit                              |

---

## Interactive debugger

Launch with `--debug` to drop into a Z80 debugging REPL instead of the emulator:

```
python main.py roms/game.gg --debug
```

### Debugger commands

| Command          | Description                                          |
|------------------|------------------------------------------------------|
| `s [N]`          | Step N instructions (default 1)                      |
| `c`              | Continue until breakpoint (Ctrl+C to interrupt)      |
| `b ADDR`         | Add breakpoint at hex address                        |
| `del ADDR`       | Delete breakpoint                                    |
| `bl`             | List breakpoints                                     |
| `regs`           | Show all registers and flags                         |
| `dis [ADDR [N]]` | Disassemble N instructions at ADDR (default: PC, 10) |
| `x ADDR [N]`     | Hex dump N bytes at ADDR (default 16)                |
| `sp [N]`         | Show top N stack entries (default 8)                 |
| `save [SLOT]`    | Save emulator state to slot (default 0)              |
| `load [SLOT]`    | Load emulator state from slot (default 0)            |
| `reset`          | Reset console to power-on state                      |
| `q` / `quit`     | Exit debugger                                        |

The disassembler covers all prefix forms including CB, ED, DD/FD, and DDCB/FDCB,
with correct undocumented mnemonics (SLL, IXH/IXL) and proper `(IX+d)` vs
IXH/IXL disambiguation.

---

## Project structure

```
PyGear/
├── main.py              — entry point, pygame event loop
├── pygear/
│   ├── console.py       — top-level orchestrator (60 fps timing)
│   ├── cartridge.py     — ROM loading, Sega header parsing, mapper detection
│   ├── debugger.py      — interactive Z80 REPL
│   ├── cpu/
│   │   ├── z80.py       — registers, interrupts, prefix dispatch
│   │   ├── opcodes.py   — main + ED + DD/FD opcode tables
│   │   ├── opcodes_cb.py— CB-prefix and DDCB/FDCB opcode tables
│   │   └── disasm.py    — full Z80 disassembler
│   ├── vdp/
│   │   ├── vdp.py       — scanline renderer, CRAM, register file, compositing
│   │   ├── tiles.py     — 4bpp planar tile decoder
│   │   └── sprites.py   — SAT parsing and sprite rasteriser
│   ├── memory/
│   │   ├── bus.py       — address-space dispatcher
│   │   ├── mapper.py    — Sega and Codemasters bank mappers
│   │   └── ram.py       — 8 KB main RAM
│   ├── io/
│   │   ├── ports.py     — Z80 IN/OUT port map
│   │   └── joypad.py    — keyboard to joypad state
│   └── sound/
│       └── psg.py       — SN76489 synthesis and stereo mixing
└── tests/
    ├── test_z80.py
    ├── test_vdp.py
    ├── test_psg.py
    ├── test_memory.py
    └── test_save_state.py
```

## Running the tests

```
python -m pytest tests/
```
