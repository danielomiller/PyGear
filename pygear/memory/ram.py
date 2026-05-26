"""8 KB main RAM."""


class RAM:
    SIZE = 0x2000  # 8 KB

    def __init__(self):
        self._mem = bytearray(self.SIZE)

    def read(self, offset: int) -> int:
        return self._mem[offset & (self.SIZE - 1)]

    def write(self, offset: int, value: int):
        self._mem[offset & (self.SIZE - 1)] = value & 0xFF

    def reset(self):
        for i in range(self.SIZE):
            self._mem[i] = 0

    def get_state(self) -> dict:
        return {'mem': bytes(self._mem)}

    def set_state(self, s: dict) -> None:
        self._mem[:] = s['mem']
