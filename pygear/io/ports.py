"""Game Gear I/O port dispatcher.

Routes Z80 IN/OUT instructions to the appropriate hardware component.

Read map
--------
0x00        Joypad port_00()  (START button)
0x7E        VDP V-counter
0x7F        VDP H-counter
0xBE        VDP data port
0xBF        VDP control / status
0xC0/0xDC   Joypad port_c0() (directions + fire)
0xC1/0xDD   0xFF             (port 2 — unused on GG)
all others  0xFF

Write map
---------
0x06        GG stereo sound register (PSG channel left/right enable)
0x7E        PSG data port
0xBE        VDP data port
0xBF        VDP control port
all others  ignored
"""

_JOYPAD_C0 = frozenset({0xC0, 0xDC})
_VDP_READ  = frozenset({0x7E, 0x7F, 0xBE, 0xBF})
_VDP_WRITE = frozenset({0xBE, 0xBF})


class IOPorts:
    def __init__(self, vdp, joypad, psg=None):
        self._vdp    = vdp
        self._joypad = joypad
        self._psg    = psg

    def read(self, port: int) -> int:
        port &= 0xFF
        if port == 0x00:
            return self._joypad.port_00()
        if port in _VDP_READ:
            return self._vdp.port_read(port)
        if port in _JOYPAD_C0:
            return self._joypad.port_c0()
        return 0xFF

    def write(self, port: int, value: int) -> None:
        port  &= 0xFF
        value &= 0xFF
        if port == 0x06 and self._psg is not None:
            self._psg.set_stereo(value)
        elif port == 0x7E and self._psg is not None:
            self._psg.write(value)
        elif port in _VDP_WRITE:
            self._vdp.port_write(port, value)
