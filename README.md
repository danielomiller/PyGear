# PyGear — Sega Game Gear Emulator

A from-scratch Sega Game Gear emulator written in Python, targeting playability of commercial Game Gear ROMs.

## Hardware Emulated

| Component | Details |
|-----------|---------|
| CPU | Zilog Z80 @ ~3.58 MHz |
| VDP | Modified SMS VDP — 256×192 internal, cropped to 160×144 |
| Color | 12-bit BGR CRAM, 32 colors on screen |
| Sound | SN76489 PSG (3 tone + 1 noise channel) |
| RAM | 8 KB main RAM, 16 KB VRAM |
| Mapper | Sega mapper (3 × 16 KB ROM slots) |

## Requirements

- Python 3.9+
- pygame 2.x
- numpy 1.20+

```
pip install -r requirements.txt
```

## Running a ROM

```
python main.py roms/game.gg
```

The window opens at 3× scale (480×432). Pass `--scale N` for a different multiplier.

## Controls

| Key | Function |
|-----|---------|
| Arrow keys | D-pad |
| Z | Button 1 |
| X | Button 2 |
| Enter | Start |
| Escape | Quit |

## Project Structure

```
PyGear/
├── main.py              — entry point, pygame main loop
├── pygear/
│   ├── console.py       — top-level orchestrator (60 fps timing)
│   ├── cartridge.py     — ROM loading + Sega header parsing
│   ├── cpu/
│   │   ├── z80.py       — Z80 CPU (registers, interrupts, dispatch)
│   │   ├── opcodes.py   — main + ED + DD/FD opcode tables
│   │   └── opcodes_cb.py— CB-prefix bit-manipulation opcodes
│   ├── vdp/
│   │   ├── vdp.py       — scanline renderer, CRAM, register file
│   │   ├── tiles.py     — 4bpp planar tile decoder
│   │   └── sprites.py   — SAT parsing + sprite rasteriser
│   ├── memory/
│   │   ├── bus.py       — address-space dispatcher
│   │   ├── mapper.py    — Sega bank mapper
│   │   └── ram.py       — 8 KB main RAM
│   ├── io/
│   │   ├── ports.py     — Z80 IN/OUT port map
│   │   └── joypad.py    — keyboard → joypad state
│   └── sound/
│       └── psg.py       — SN76489 PSG (pygame.mixer)
└── tests/
    ├── test_z80.py
    ├── test_vdp.py
    └── test_memory.py
```

## Testing

```
python -m pytest tests/
```

The Z80 test suite includes a harness for the **Zexdoc** Z80 exerciser ROM (place
`zexdoc.com` in `roms/`).  All 67 test groups must produce the canonical CRC
before VDP work is considered stable.

## Build Order / Status

1. ✅ Cartridge + memory bus
2. ✅ Z80 CPU (full instruction set)
3. ✅ VDP background tiles
4. ✅ I/O ports + joypad
5. ✅ Sprites
6. ✅ Sega mapper
7. ✅ PSG sound (mixer stub; tone synthesis implemented)

## Known Limitations

None known.
