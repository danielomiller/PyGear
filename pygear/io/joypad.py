"""Game Gear joypad state tracker.

Button encoding (active-low)
-----------------------------
Port 0x00  (read):
  bit 7  START

Port 0xC0 / 0xDC  (read):
  bit 5  BUTTON2  (TR)
  bit 4  BUTTON1  (TL)
  bit 3  RIGHT
  bit 2  LEFT
  bit 1  DOWN
  bit 0  UP

All unreferenced bits read as 1.
"""

# Button constants
START   = "START"
UP      = "UP"
DOWN    = "DOWN"
LEFT    = "LEFT"
RIGHT   = "RIGHT"
BUTTON1 = "BUTTON1"
BUTTON2 = "BUTTON2"

# Bit positions in port 0xC0
_C0_BITS = {
    UP:      0,
    DOWN:    1,
    LEFT:    2,
    RIGHT:   3,
    BUTTON1: 4,
    BUTTON2: 5,
}


class Joypad:
    def __init__(self):
        self._pressed: set = set()

    def press(self, button: str) -> None:
        self._pressed.add(button)

    def release(self, button: str) -> None:
        self._pressed.discard(button)

    def reset(self) -> None:
        self._pressed.clear()

    def get_state(self) -> dict:
        return {'_pressed': set(self._pressed)}

    def set_state(self, s: dict) -> None:
        self._pressed = set(s['_pressed'])

    def port_00(self) -> int:
        """Return the byte read from port 0x00 (START button, active-low)."""
        value = 0xFF
        if START in self._pressed:
            value &= ~0x80
        return value & 0xFF

    def port_c0(self) -> int:
        """Return the byte read from port 0xC0/0xDC (directions + fire, active-low)."""
        value = 0xFF
        for button, bit in _C0_BITS.items():
            if button in self._pressed:
                value &= ~(1 << bit)
        return value & 0xFF
