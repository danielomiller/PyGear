"""Sega Game Gear cartridge — ROM loading and bank access."""

_HEADER_OFFSETS = (0x1FF0, 0x3FF0, 0x7FF0)
_SEGA_MAGIC = b"TMR SEGA"

_ROM_SIZE_TABLE = {
    0x0A: 8 * 1024,
    0x0B: 16 * 1024,
    0x0C: 32 * 1024,
    0x0D: 48 * 1024,
    0x0E: 64 * 1024,
    0x0F: 128 * 1024,
    0x10: 256 * 1024,
    0x11: 512 * 1024,
    0x12: 1024 * 1024,  # 1 MB
}


class Cartridge:
    def __init__(self, path: str):
        with open(path, "rb") as f:
            data = f.read()

        # Strip 512-byte copier header if present
        if len(data) % 1024 == 512:
            data = data[512:]

        self._data = bytearray(data)
        self.size = len(data)
        self.bank_count = max(1, self.size // (16 * 1024))

        self._parse_header()

    # ------------------------------------------------------------------
    def _parse_header(self):
        self.header_valid = False
        self.product_code = 0
        self.version = 0
        self.region = 0
        self.rom_size_byte = 0

        for off in _HEADER_OFFSETS:
            if off + 16 > len(self._data):
                continue
            if self._data[off : off + 8] == _SEGA_MAGIC:
                self.header_valid = True
                # Bytes 8-9: checksum (ignored for emulation)
                self.product_code = (
                    self._data[off + 12]
                    | (self._data[off + 13] << 8)
                    | ((self._data[off + 14] & 0xF0) << 12)
                )
                self.version = self._data[off + 14] & 0x0F
                self.region = (self._data[off + 15] >> 4) & 0x0F
                self.rom_size_byte = self._data[off + 15] & 0x0F
                break

    # ------------------------------------------------------------------
    def read(self, bank: int, offset: int) -> int:
        """Read a byte from the given 16 KB bank at offset (0–0x3FFF)."""
        addr = (bank * 0x4000 + offset) % self.size
        return self._data[addr]

    def read_raw(self, addr: int) -> int:
        """Read a byte from the raw ROM image (physical address)."""
        return self._data[addr % self.size]

    def __len__(self):
        return self.size
