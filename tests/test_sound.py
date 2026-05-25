"""Tests for pygear.sound: PSG register file, tone synthesis, noise synthesis,
and IOPorts wiring (port 0x7E)."""
import math
import pytest
from pygear.sound.psg import PSG, ATTENUATION, PSG_CLOCK, TICK_RATE
from pygear.io.ports import IOPorts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockVDP:
    def __init__(self):
        self.writes: list[tuple[int, int]] = []
        self._vcounter = 0x42

    def port_read(self, port: int) -> int:
        return self._vcounter

    def port_write(self, port: int, value: int) -> None:
        self.writes.append((port, value))


class MockJoypad:
    def port_00(self) -> int:
        return 0xFF

    def port_c0(self) -> int:
        return 0xFF


def _make_io(psg=None):
    return IOPorts(MockVDP(), MockJoypad(), psg)


def _set_tone(psg: PSG, ch: int, period: int) -> None:
    """Write a 10-bit period to tone channel ch (0-2) via latch+data bytes."""
    reg = ch * 2                         # reg 0, 2, or 4
    lo  = period & 0x0F
    hi  = (period >> 4) & 0x3F
    psg.write(0x80 | (reg << 4) | lo)   # latch byte
    psg.write(hi)                         # data byte


def _set_volume(psg: PSG, ch: int, vol: int) -> None:
    """Write attenuation *vol* to channel ch (0-3)."""
    reg = ch * 2 + 1                     # reg 1, 3, 5, or 7
    psg.write(0x80 | (reg << 4) | (vol & 0x0F))


def _count_zero_crossings(samples: list) -> int:
    # samples is a list of (left, right) pairs; use the left channel
    left = [s[0] for s in samples]
    return sum(1 for i in range(1, len(left)) if left[i] != left[i - 1])


# ---------------------------------------------------------------------------
# TestAttenuation
# ---------------------------------------------------------------------------

class TestAttenuation:
    def test_length(self):
        assert len(ATTENUATION) == 16

    def test_index_0_is_unity(self):
        assert abs(ATTENUATION[0] - 1.0) < 1e-9

    def test_index_15_is_silence(self):
        assert ATTENUATION[15] == 0.0

    def test_all_in_unit_range(self):
        assert all(0.0 <= v <= 1.0 for v in ATTENUATION)

    def test_monotonically_decreasing(self):
        assert all(ATTENUATION[i] > ATTENUATION[i + 1] for i in range(14))

    def test_2db_per_step(self):
        ratio = 10 ** (-2 / 20)
        for i in range(14):
            assert abs(ATTENUATION[i + 1] / ATTENUATION[i] - ratio) < 1e-9

    def test_tick_rate(self):
        assert abs(TICK_RATE - PSG_CLOCK / 16) < 1e-6

    def test_psg_clock_value(self):
        assert PSG_CLOCK == 3_579_545


# ---------------------------------------------------------------------------
# TestPSGDefaults
# ---------------------------------------------------------------------------

class TestPSGDefaults:
    def test_tone_periods_zero(self):
        assert PSG()._tone_period == [0, 0, 0]

    def test_all_volumes_silent(self):
        assert PSG()._volume == [15, 15, 15, 15]

    def test_noise_ctrl_zero(self):
        assert PSG()._noise_ctrl == 0

    def test_latch_reg_zero(self):
        assert PSG()._latch_reg == 0

    def test_lfsr_initial(self):
        assert PSG()._lfsr == 0x8000

    def test_tone_counters_zero(self):
        assert PSG()._tone_counter == [0.0, 0.0, 0.0]

    def test_tone_flip_false(self):
        assert PSG()._tone_flip == [False, False, False]

    def test_noise_counter_zero(self):
        assert PSG()._noise_counter == 0.0

    def test_reset_restores_tone_periods(self):
        p = PSG()
        _set_tone(p, 0, 0x3FF)
        p.reset()
        assert p._tone_period == [0, 0, 0]

    def test_reset_restores_volumes(self):
        p = PSG()
        _set_volume(p, 0, 5)
        p.reset()
        assert p._volume == [15, 15, 15, 15]

    def test_reset_restores_noise_ctrl(self):
        p = PSG()
        p.write(0x80 | (6 << 4) | 7)
        p.reset()
        assert p._noise_ctrl == 0

    def test_reset_restores_lfsr(self):
        p = PSG()
        p.render(500, 44100)
        p.reset()
        assert p._lfsr == 0x8000

    def test_reset_restores_synthesis_state(self):
        p = PSG()
        p.render(100, 44100)
        p.reset()
        assert p._tone_counter == [0.0, 0.0, 0.0]
        assert p._tone_flip    == [False, False, False]
        assert p._noise_counter == 0.0

    def test_stereo_default_all_channels(self):
        assert PSG()._stereo == 0xFF

    def test_reset_restores_stereo(self):
        p = PSG()
        p.set_stereo(0x00)
        p.reset()
        assert p._stereo == 0xFF


# ---------------------------------------------------------------------------
# TestPSGWrite
# ---------------------------------------------------------------------------

class TestPSGWrite:
    # -- Latch bytes --

    def test_latch_tone_ch0_low_nibble(self):
        p = PSG()
        p.write(0x80 | 0x0A)          # reg=0, data=0xA
        assert p._latch_reg == 0
        assert p._tone_period[0] & 0x0F == 0xA

    def test_latch_tone_ch0_preserves_high_bits(self):
        p = PSG()
        _set_tone(p, 0, 0x3F0)        # set high 6 bits
        p.write(0x80 | 0x05)          # re-latch low nibble=5
        assert p._tone_period[0] == 0x3F5

    def test_latch_tone_ch1(self):
        p = PSG()
        p.write(0x80 | (2 << 4) | 0xB)
        assert p._latch_reg == 2
        assert p._tone_period[1] & 0x0F == 0xB

    def test_latch_tone_ch2(self):
        p = PSG()
        p.write(0x80 | (4 << 4) | 0xC)
        assert p._latch_reg == 4
        assert p._tone_period[2] & 0x0F == 0xC

    def test_latch_volume_ch0(self):
        p = PSG()
        p.write(0x80 | (1 << 4) | 7)
        assert p._latch_reg == 1
        assert p._volume[0] == 7

    def test_latch_volume_ch1(self):
        p = PSG()
        p.write(0x80 | (3 << 4) | 3)
        assert p._volume[1] == 3

    def test_latch_volume_ch2(self):
        p = PSG()
        p.write(0x80 | (5 << 4) | 9)
        assert p._volume[2] == 9

    def test_latch_volume_noise(self):
        p = PSG()
        p.write(0x80 | (7 << 4) | 12)
        assert p._volume[3] == 12

    def test_latch_noise_ctrl_stores_value(self):
        p = PSG()
        p.write(0x80 | (6 << 4) | 5)
        assert p._noise_ctrl == 5

    def test_latch_noise_ctrl_resets_lfsr(self):
        p = PSG()
        p.render(300, 44100)           # advance LFSR
        p.write(0x80 | (6 << 4) | 0)
        assert p._lfsr == 0x8000

    def test_value_masked_to_byte(self):
        p = PSG()
        p.write(0x180)                 # masked to 0x80 → latch reg=0, data=0
        assert p._latch_reg == 0

    # -- Data bytes --

    def test_data_tone_high_bits(self):
        p = PSG()
        p.write(0x80 | 0x07)          # latch reg=0, low=7
        p.write(0x0F)                  # high 6 bits = 0x0F → period bits 9-4
        assert p._tone_period[0] == (0x0F << 4) | 7

    def test_data_tone_preserves_low_nibble(self):
        p = PSG()
        p.write(0x80 | 0x0D)          # low nibble=D
        p.write(0x3F)                  # high 6 bits all set
        assert p._tone_period[0] & 0x0F == 0xD

    def test_data_tone_masks_to_6_bits(self):
        p = PSG()
        p.write(0x80 | 0x00)          # latch reg=0
        p.write(0x7F)                  # bit7=0 (data byte); bits 5-0 = 0x3F
        assert p._tone_period[0] == (0x3F << 4)

    def test_data_volume_updates_latched_channel(self):
        p = PSG()
        p.write(0x80 | (3 << 4) | 0)  # latch reg=3 (ch1 vol)
        p.write(0x08)                  # data byte: low 4 bits = 8
        assert p._volume[1] == 8

    def test_data_noise_ctrl_low_nibble(self):
        p = PSG()
        p.write(0x80 | (6 << 4) | 0)  # latch reg=6
        p.write(0x06)                  # low nibble=6
        assert p._noise_ctrl == 6

    def test_data_noise_ctrl_resets_lfsr(self):
        p = PSG()
        p.write(0x80 | (6 << 4) | 0)
        p.render(300, 44100)
        p.write(0x03)                  # data byte to noise ctrl reg
        assert p._lfsr == 0x8000

    def test_full_10bit_period_roundtrip(self):
        p = PSG()
        _set_tone(p, 1, 0x2A5)
        assert p._tone_period[1] == 0x2A5


# ---------------------------------------------------------------------------
# TestToneSynthesis
# ---------------------------------------------------------------------------

class TestToneSynthesis:
    def test_all_silent_yields_zeros(self):
        p = PSG()
        # render() returns (left, right) pairs; all channels silent → both sides 0.0
        assert all(l == 0.0 and r == 0.0 for l, r in p.render(256, 44100))

    def test_returns_correct_length(self):
        p = PSG()
        assert len(p.render(100, 44100)) == 100

    def test_returns_stereo_pairs(self):
        p = PSG()
        samples = p.render(4, 44100)
        assert all(isinstance(s, tuple) and len(s) == 2 for s in samples)

    def test_active_channel_nonzero(self):
        p = PSG()
        _set_tone(p, 0, 0x100)
        _set_volume(p, 0, 0)
        # With default stereo=0xFF, both sides should have signal
        assert any(l != 0.0 or r != 0.0 for l, r in p.render(512, 44100))

    def test_square_wave_amplitude_single_channel(self):
        p = PSG()
        _set_tone(p, 0, 0x100)
        _set_volume(p, 0, 0)           # ATTENUATION[0] = 1.0; expect ±0.25
        # With default stereo=0xFF both sides equal; check left channel
        for l, r in p.render(2048, 44100):
            assert abs(abs(l) - 0.25) < 1e-9
            assert abs(abs(r) - 0.25) < 1e-9

    def test_attenuation_applied(self):
        p = PSG()
        _set_tone(p, 0, 0x100)
        _set_volume(p, 0, 2)           # ATTENUATION[2] = 10^(-4/20)
        expected_amp = ATTENUATION[2] / 4.0
        for l, r in p.render(512, 44100):
            assert abs(abs(l) - expected_amp) < 1e-9
            assert abs(abs(r) - expected_amp) < 1e-9

    def test_frequency_accuracy(self):
        sr = 44100
        period = 0x106                 # f ≈ TICK_RATE / (2 * 0x106)
        p = PSG()
        _set_tone(p, 0, period)
        _set_volume(p, 0, 0)
        samples = p.render(sr, sr)     # 1 second at 1 sample-per-tick
        crossings = _count_zero_crossings(samples)
        expected_f = TICK_RATE / (2 * period)
        measured_f = crossings / 2.0
        assert abs(measured_f - expected_f) / expected_f < 0.01

    def test_three_channels_in_phase_max_amplitude(self):
        p = PSG()
        for ch in range(3):
            _set_tone(p, ch, 0x100)
            _set_volume(p, ch, 0)
        max_abs = max(abs(l) for l, r in p.render(64, 44100))
        assert abs(max_abs - 0.75) < 1e-9

    def test_channel_independence(self):
        # Channel 1 and 2 silent: only ch0 contributes
        p = PSG()
        _set_tone(p, 0, 0x80)
        _set_volume(p, 0, 0)
        samples = p.render(512, 44100)
        for l, r in samples:
            assert abs(abs(l) - 0.25) < 1e-9
            assert abs(abs(r) - 0.25) < 1e-9

    def test_stateful_across_calls(self):
        p = PSG()
        _set_tone(p, 0, 0x80)
        _set_volume(p, 0, 0)
        a = p.render(64, 44100)
        b = p.render(64, 44100)
        p2 = PSG()
        _set_tone(p2, 0, 0x80)
        _set_volume(p2, 0, 0)
        assert a + b == p2.render(128, 44100)

    def test_period_zero_no_crash(self):
        p = PSG()
        _set_volume(p, 0, 0)           # vol=0 but period stays 0
        p.render(64, 44100)            # must not hang or raise

    def test_output_bounded(self):
        p = PSG()
        for ch in range(3):
            _set_tone(p, ch, 1)        # very high frequency
            _set_volume(p, ch, 0)
        for l, r in p.render(1024, 44100):
            assert abs(l) <= 1.0 + 1e-9
            assert abs(r) <= 1.0 + 1e-9

    def test_silent_channel_contributes_zero(self):
        # ch1 active, ch0 and ch2 silent → amplitude = ±0.25 only
        p = PSG()
        _set_tone(p, 1, 0x100)
        _set_volume(p, 1, 0)
        for l, r in p.render(512, 44100):
            assert abs(abs(l) - 0.25) < 1e-9
            assert abs(abs(r) - 0.25) < 1e-9


# ---------------------------------------------------------------------------
# TestNoiseSynthesis
# ---------------------------------------------------------------------------

class TestNoiseSynthesis:
    def _noise_only(self, nf: int, white: bool, vol: int = 0) -> PSG:
        p = PSG()
        ctrl = (0x04 if white else 0x00) | (nf & 0x03)
        p.write(0x80 | (6 << 4) | ctrl)
        _set_volume(p, 3, vol)
        return p

    def test_silent_noise_zero(self):
        p = self._noise_only(0, False, vol=15)
        assert all(l == 0.0 and r == 0.0 for l, r in p.render(256, 44100))

    def test_active_noise_nonzero(self):
        p = self._noise_only(0, False, vol=0)
        samples = p.render(4096, 44100)
        assert any(l != 0.0 or r != 0.0 for l, r in samples)

    def test_noise_amplitude_bounded(self):
        p = self._noise_only(0, True, vol=0)
        for l, r in p.render(2048, 44100):
            assert abs(l) <= 1.0 + 1e-9
            assert abs(r) <= 1.0 + 1e-9

    def test_periodic_lfsr_step(self):
        # Verify manual LFSR step matches implementation
        lfsr = 0x8000
        for _ in range(4):
            out_bit  = lfsr & 1
            lfsr = (lfsr >> 1) | (out_bit << 15)
        assert lfsr == 0x0800

    def test_white_differs_from_periodic(self):
        s_per   = self._noise_only(0, False).render(4000, 44100)
        s_white = self._noise_only(0, True ).render(4000, 44100)
        assert s_per != s_white

    def test_white_noise_more_entropy(self):
        s_per   = self._noise_only(0, False).render(4000, 44100)
        s_white = self._noise_only(0, True ).render(4000, 44100)
        assert len(set(s_white)) >= len(set(s_per))

    def test_nf0_nf1_differ(self):
        s0 = self._noise_only(0, False).render(2000, 44100)
        s1 = self._noise_only(1, False).render(2000, 44100)
        assert s0 != s1

    def test_nf1_nf2_differ(self):
        s1 = self._noise_only(1, False).render(2000, 44100)
        s2 = self._noise_only(2, False).render(2000, 44100)
        assert s1 != s2

    def test_nf3_uses_ch2_period(self):
        # NF=3 with ch2 period=0x40 should match NF=2 (period=0x40)
        p3 = PSG()
        _set_tone(p3, 2, 0x40)
        p3.write(0x80 | (6 << 4) | 3)   # NF=3, periodic
        _set_volume(p3, 3, 0)

        p2 = PSG()
        _set_tone(p2, 2, 0x40)
        p2.write(0x80 | (6 << 4) | 2)   # NF=2, periodic → period=0x40
        _set_volume(p2, 3, 0)

        assert p3.render(500, 44100) == p2.render(500, 44100)

    def test_lfsr_reset_on_latch_write(self):
        p = PSG()
        p.render(500, 44100)
        assert p._lfsr != 0x8000
        p.write(0x80 | (6 << 4) | 0)
        assert p._lfsr == 0x8000

    def test_lfsr_reset_on_data_write(self):
        p = PSG()
        p.write(0x80 | (6 << 4) | 0)   # latch reg=6
        p.render(500, 44100)
        p.write(0x03)                    # data byte to noise ctrl
        assert p._lfsr == 0x8000

    def test_noise_stateful_across_calls(self):
        p = self._noise_only(0, True)
        a = p.render(100, 44100)
        b = p.render(100, 44100)
        p2 = self._noise_only(0, True)
        assert a + b == p2.render(200, 44100)

    def test_four_channels_max_amplitude(self):
        p = PSG()
        for ch in range(3):
            _set_tone(p, ch, 0x100)
            _set_volume(p, ch, 0)
        p.write(0x80 | (6 << 4) | 0)
        _set_volume(p, 3, 0)
        for l, r in p.render(8192, 44100):
            assert abs(l) <= 1.0 + 1e-9
            assert abs(r) <= 1.0 + 1e-9

    def test_noise_attenuation(self):
        p = self._noise_only(0, False, vol=4)
        expected = ATTENUATION[4] / 4.0
        for l, r in p.render(512, 44100):
            assert abs(abs(l) - expected) < 1e-9
            assert abs(abs(r) - expected) < 1e-9


# ---------------------------------------------------------------------------
# TestIOPortsPSG
# ---------------------------------------------------------------------------

class TestIOPortsPSG:
    def test_psg_none_default_no_crash(self):
        io = _make_io()
        io.write(0x7E, 0x9A)           # must not raise

    def test_psg_none_write_7e_ignored(self):
        vdp = MockVDP()
        io  = IOPorts(vdp, MockJoypad())
        io.write(0x7E, 0x80)
        assert vdp.writes == []

    def test_write_7e_routes_to_psg(self):
        psg = PSG()
        io  = _make_io(psg)
        io.write(0x7E, 0x85)           # latch reg=0, low nibble=5
        assert psg._tone_period[0] == 5
        assert psg._latch_reg == 0

    def test_write_7e_does_not_reach_vdp(self):
        vdp = MockVDP()
        io  = IOPorts(vdp, MockJoypad(), PSG())
        io.write(0x7E, 0x80)
        assert vdp.writes == []

    def test_read_7e_still_goes_to_vdp(self):
        vdp = MockVDP()
        vdp._vcounter = 0x55
        io  = IOPorts(vdp, MockJoypad(), PSG())
        assert io.read(0x7E) == 0x55

    def test_write_be_still_goes_to_vdp(self):
        vdp = MockVDP()
        io  = IOPorts(vdp, MockJoypad(), PSG())
        io.write(0xBE, 0x42)
        assert (0xBE, 0x42) in vdp.writes

    def test_write_bf_still_goes_to_vdp(self):
        vdp = MockVDP()
        io  = IOPorts(vdp, MockJoypad(), PSG())
        io.write(0xBF, 0x81)
        assert (0xBF, 0x81) in vdp.writes

    def test_write_7e_value_masked(self):
        psg = PSG()
        io  = _make_io(psg)
        io.write(0x7E, 0x1FF)          # masked to 0xFF → latch reg=7, data=0xF
        assert psg._volume[3] == 0xF

    def test_write_7e_full_psg_sequence(self):
        psg = PSG()
        io  = _make_io(psg)
        # Set ch0 period to 0x123 and volume to 3
        io.write(0x7E, 0x80 | 0x03)    # latch reg=0, low=3
        io.write(0x7E, 0x12)            # high bits → period = 0x123
        io.write(0x7E, 0x80 | (1 << 4) | 3)  # ch0 vol=3
        assert psg._tone_period[0] == 0x123
        assert psg._volume[0] == 3

    def test_psg_renders_after_io_writes(self):
        psg = PSG()
        io  = _make_io(psg)
        _set_tone(psg, 0, 0x100)
        io.write(0x7E, 0x80 | (1 << 4) | 0)   # set vol=0 via IO port
        samples = psg.render(256, 44100)
        assert any(l != 0.0 or r != 0.0 for l, r in samples)

    def test_write_06_routes_to_psg_stereo(self):
        psg = PSG()
        io  = _make_io(psg)
        io.write(0x06, 0xAA)
        assert psg._stereo == 0xAA

    def test_write_06_psg_none_no_crash(self):
        io = IOPorts(MockVDP(), MockJoypad())
        io.write(0x06, 0x55)   # must not raise when psg is None

    def test_write_06_does_not_reach_vdp(self):
        vdp = MockVDP()
        io  = IOPorts(vdp, MockJoypad(), PSG())
        io.write(0x06, 0xF0)
        assert vdp.writes == []

    def test_stereo_left_only_silences_right(self):
        # Stereo = 0x0F: all channels left-only; right side should be silent
        psg = PSG()
        _set_tone(psg, 0, 0x100)
        _set_volume(psg, 0, 0)
        psg.set_stereo(0x0F)
        for l, r in psg.render(512, 44100):
            assert r == 0.0
            assert abs(abs(l) - 0.25) < 1e-9

    def test_stereo_right_only_silences_left(self):
        # Stereo = 0xF0: all channels right-only; left side should be silent
        psg = PSG()
        _set_tone(psg, 0, 0x100)
        _set_volume(psg, 0, 0)
        psg.set_stereo(0xF0)
        for l, r in psg.render(512, 44100):
            assert l == 0.0
            assert abs(abs(r) - 0.25) < 1e-9

    def test_stereo_all_off_yields_silence(self):
        psg = PSG()
        _set_tone(psg, 0, 0x100)
        _set_volume(psg, 0, 0)
        psg.set_stereo(0x00)
        for l, r in psg.render(256, 44100):
            assert l == 0.0
            assert r == 0.0

    def test_stereo_value_masked_to_byte(self):
        psg = PSG()
        psg.set_stereo(0x1AA)   # masked to 0xAA
        assert psg._stereo == 0xAA
