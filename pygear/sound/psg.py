"""SN76489 PSG — register file, write decoder, and attenuation table.

The SN76489 has eight internal registers accessed through a single write port:

  Reg 0  Tone channel 0 frequency  (10-bit period)
  Reg 1  Tone channel 0 volume     (4-bit attenuation)
  Reg 2  Tone channel 1 frequency  (10-bit period)
  Reg 3  Tone channel 1 volume     (4-bit attenuation)
  Reg 4  Tone channel 2 frequency  (10-bit period)
  Reg 5  Tone channel 2 volume     (4-bit attenuation)
  Reg 6  Noise control             (4-bit: bits 1-0=NF, bit 2=type)
  Reg 7  Noise volume              (4-bit attenuation)

Write protocol
--------------
Latch byte (bit 7 = 1):  1 CC T D3 D2 D1 D0
  CC = channel 0-3, T = 0 (frequency) or 1 (volume/noise), D = 4 data bits.
  Selects and partially writes the target register; stores register index.
  For tone registers: D3-D0 → period bits 3-0.
  For volume/noise:   D3-D0 → full 4-bit value.

Data byte (bit 7 = 0):  0 x D9 D8 D7 D6 D5 D4
  Written to whichever register was most recently latched.
  For tone registers: D9-D4 → period bits 9-4 (bits 3-0 preserved from latch).
  For volume/noise:   D3-D0 of the byte → full 4-bit value (upper bits ignored).

Attenuation
-----------
4-bit value 0-15; each step is -2 dB.  Value 15 = silence.
ATTENUATION[n] gives the linear amplitude multiplier (0.0–1.0).

Clock
-----
PSG_CLOCK = 3,579,545 Hz (Z80 master clock).
TICK_RATE = PSG_CLOCK / 16 — rate at which tone counters decrement.
"""

import math

PSG_CLOCK = 3_579_545        # Hz
TICK_RATE = PSG_CLOCK / 16   # ~223,722 Hz

# Linear amplitude for each attenuation step (2 dB per step, 15 = silence)
ATTENUATION: list = [10 ** (-i * 2 / 20) for i in range(15)] + [0.0]


class PSG:
    def __init__(self):
        self._tone_period  = [0, 0, 0]        # 10-bit periods, channels 0-2
        self._volume       = [15, 15, 15, 15] # 4-bit attenuation (default silent)
        self._noise_ctrl   = 0                # 4-bit noise control
        self._latch_reg    = 0                # last latched register index 0-7

        # Synthesis state initialised here; populated fully by render() in Task 2/3
        self._tone_counter = [0.0, 0.0, 0.0]
        self._tone_flip    = [False, False, False]
        self._lfsr         = 0x8000
        self._noise_counter = 0.0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._tone_period   = [0, 0, 0]
        self._volume        = [15, 15, 15, 15]
        self._noise_ctrl    = 0
        self._latch_reg     = 0
        self._tone_counter  = [0.0, 0.0, 0.0]
        self._tone_flip     = [False, False, False]
        self._lfsr          = 0x8000
        self._noise_counter = 0.0

    # ------------------------------------------------------------------
    def write(self, value: int) -> None:
        """Decode and store one byte written to the PSG data port."""
        value &= 0xFF

        if value & 0x80:
            # ---- Latch byte: 1 CC T D3 D2 D1 D0 ----
            reg  = (value >> 4) & 0x07   # 3-bit register index (CC<<1 | T)
            data = value & 0x0F
            self._latch_reg = reg

            if reg == 0 or reg == 2 or reg == 4:   # tone frequency
                ch = reg >> 1
                # Store low 4 bits of the 10-bit period
                self._tone_period[ch] = (self._tone_period[ch] & 0x3F0) | data
            elif reg == 6:                          # noise control
                self._noise_ctrl = data
            else:                                   # volume (regs 1, 3, 5, 7)
                ch = reg >> 1
                self._volume[ch] = data

        else:
            # ---- Data byte: 0 x D9 D8 D7 D6 D5 D4 ----
            reg = self._latch_reg

            if reg == 0 or reg == 2 or reg == 4:   # tone frequency
                ch = reg >> 1
                # Store high 6 bits of the 10-bit period; preserve low 4 bits
                self._tone_period[ch] = ((value & 0x3F) << 4) | (self._tone_period[ch] & 0x0F)
            elif reg == 6:                          # noise control
                self._noise_ctrl = value & 0x0F
            else:                                   # volume
                ch = reg >> 1
                self._volume[ch] = value & 0x0F
