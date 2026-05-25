"""SN76489 PSG — register file, write decoder, attenuation table, and synthesis.

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

Tone synthesis
--------------
Each tone channel runs an internal down-counter initialised to its period.
The counter decrements by ticks_per_sample = TICK_RATE / sample_rate each
audio sample.  When it reaches ≤ 0 it reloads (counter += period) and
the square-wave output flips polarity.  A period of 0 is treated as 1 to
prevent an infinite reload loop.

Output frequency = TICK_RATE / (2 × period) Hz.

render() returns a list of (left, right) float pairs in [-1.0, +1.0].
Each channel contributes ±amp where amp = ATTENUATION[volume]; the
four-channel sum per side is divided by 4.0.  The stereo register
(Game Gear port 0x06) controls which channels reach each side:
  Bit 7: Tone 3 right  Bit 3: Tone 3 left
  Bit 6: Tone 2 right  Bit 2: Tone 2 left
  Bit 5: Tone 1 right  Bit 1: Tone 1 left
  Bit 4: Noise right   Bit 0: Noise left
Default 0xFF enables all channels on both sides (full stereo).

Noise synthesis
---------------
The noise channel uses a 16-bit LFSR (initialised to 0x8000).  It is
clocked by its own down-counter; the noise control register selects the
clock period and the LFSR feedback mode.

Noise control byte (reg 6) bit layout:
  bit 2  type: 0 = periodic, 1 = white noise
  bits 1-0  NF: clock select
    00 → period 0x10  (TICK_RATE / 32)
    01 → period 0x20  (TICK_RATE / 64)
    10 → period 0x40  (TICK_RATE / 128)
    11 → use tone channel 2's period register value

On each LFSR clock:
  output_bit = lfsr & 1
  feedback   = output_bit ^ ((lfsr >> 3) & 1)   if white noise
             = output_bit                         if periodic
  lfsr       = (lfsr >> 1) | (feedback << 15)

Writing to reg 6 resets the LFSR to 0x8000.
The noise output is ±ATTENUATION[vol[3]] according to lfsr bit 0.
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
        self._stereo       = 0xFF             # GG port 0x06: all channels both sides

        # Synthesis state
        self._tone_counter  = [0.0, 0.0, 0.0]
        self._tone_flip     = [False, False, False]
        self._lfsr          = 0x8000
        self._noise_counter = 0.0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._tone_period   = [0, 0, 0]
        self._volume        = [15, 15, 15, 15]
        self._noise_ctrl    = 0
        self._latch_reg     = 0
        self._stereo        = 0xFF
        self._tone_counter  = [0.0, 0.0, 0.0]
        self._tone_flip     = [False, False, False]
        self._lfsr          = 0x8000
        self._noise_counter = 0.0

    # ------------------------------------------------------------------
    def set_stereo(self, value: int) -> None:
        """Write the Game Gear stereo control register (port 0x06).

        Bit layout:
          Bit 7: Tone 3 right  Bit 3: Tone 3 left
          Bit 6: Tone 2 right  Bit 2: Tone 2 left
          Bit 5: Tone 1 right  Bit 1: Tone 1 left
          Bit 4: Noise right   Bit 0: Noise left
        """
        self._stereo = value & 0xFF

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
                self._lfsr = 0x8000               # reset LFSR on any noise-ctrl write
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
                self._lfsr = 0x8000               # reset LFSR on any noise-ctrl write
            else:                                   # volume
                ch = reg >> 1
                self._volume[ch] = value & 0x0F

    # ------------------------------------------------------------------
    # Noise period lookup for NF bits 0-2; NF=3 borrows tone channel 2's period.
    _NOISE_PERIODS = (0x10, 0x20, 0x40)

    def render(self, n_samples: int, sample_rate: float) -> list:
        """Synthesise *n_samples* audio samples at *sample_rate* Hz.

        Returns a list of (left, right) float pairs in [-1.0, +1.0].
        All four channels (three tone + one noise) are mixed per side and
        divided by 4.0.  The stereo register controls channel routing.

        Stereo bit layout (self._stereo):
          Bits 7-4: right enable for channels Tone3, Tone2, Tone1, Noise
          Bits 3-0: left  enable for channels Tone3, Tone2, Tone1, Noise
        Channel order matches the bit layout: ch0=Tone1, ch1=Tone2, ch2=Tone3,
        ch3=Noise.  Left bit indices: Noise=0, Tone1=1, Tone2=2, Tone3=3.
        Right bit indices: Noise=4, Tone1=5, Tone2=6, Tone3=7.
        """
        ticks_per_sample = TICK_RATE / sample_rate
        white_noise = bool(self._noise_ctrl & 0x04)
        nf          = self._noise_ctrl & 0x03
        stereo      = self._stereo

        # Precompute per-channel left/right enable masks
        # Tone ch0=Tone1 (bit1 left, bit5 right), ch1=Tone2 (bit2, bit6),
        # ch2=Tone3 (bit3, bit7); Noise ch3 (bit0, bit4)
        left_en  = [bool(stereo & (1 << (ch + 1))) for ch in range(3)]
        right_en = [bool(stereo & (1 << (ch + 5))) for ch in range(3)]
        noise_left_en  = bool(stereo & 0x01)
        noise_right_en = bool(stereo & 0x10)

        # Cache attribute references — avoids repeated dict lookup in hot loop
        tc   = self._tone_counter
        tp   = self._tone_period
        tf   = self._tone_flip
        vol  = self._volume
        atten = ATTENUATION
        noise_periods = self._NOISE_PERIODS
        noise_counter = self._noise_counter
        lfsr          = self._lfsr
        noise_period  = (tp[2] or 1) if nf == 3 else noise_periods[nf]

        out = []
        for _ in range(n_samples):
            # --- Tone channels ---
            for ch in range(3):
                tc[ch] -= ticks_per_sample
                period = tp[ch] or 1
                while tc[ch] <= 0:
                    tc[ch] += period
                    tf[ch] = not tf[ch]

            # --- Noise channel ---
            noise_counter -= ticks_per_sample
            while noise_counter <= 0:
                noise_counter += noise_period
                out_bit  = lfsr & 1
                feedback = (out_bit ^ ((lfsr >> 3) & 1)) if white_noise else out_bit
                lfsr = (lfsr >> 1) | (feedback << 15)

            # --- Mix left and right independently ---
            mix_l = 0.0
            mix_r = 0.0
            for ch in range(3):
                amp = atten[vol[ch]]
                sig = amp if tf[ch] else -amp
                if left_en[ch]:
                    mix_l += sig
                if right_en[ch]:
                    mix_r += sig

            noise_amp = atten[vol[3]]
            noise_sig = noise_amp if (lfsr & 1) else -noise_amp
            if noise_left_en:
                mix_l += noise_sig
            if noise_right_en:
                mix_r += noise_sig

            out.append((mix_l / 4.0, mix_r / 4.0))

        self._noise_counter = noise_counter
        self._lfsr          = lfsr
        return out
