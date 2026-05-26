"""Game Gear console — wires all hardware subsystems and drives frame emulation."""

import pickle

from .cartridge import Cartridge
from .cpu.z80 import Z80
from .vdp.vdp import VDP, CYCLES_PER_LINE, TOTAL_LINES
from .memory.bus import MemoryBus
from .io.ports import IOPorts
from .io.joypad import Joypad
from .sound.psg import PSG


class GameGearConsole:
    SAMPLE_RATE = 44100

    def __init__(self, cart: Cartridge) -> None:
        self.vdp      = VDP()
        self.joypad   = Joypad()
        self.psg      = PSG()
        self.bus      = MemoryBus(cart)
        self.ports    = IOPorts(self.vdp, self.joypad, self.psg)
        self.cpu      = Z80(self.bus, self.ports)
        self._sav_path = cart.sav_path
        self.vdp.attach_cpu(self.cpu)
        self.bus.load_sav(self._sav_path)   # no-op if file absent

    def reset(self) -> None:
        self.bus.reset()
        self.vdp.reset()
        self.joypad.reset()
        self.psg.reset()
        self.cpu.reset()

    def save_sav(self) -> bool:
        """Flush battery-backed cart RAM to disk. Returns True if file was written."""
        return self.bus.save_sav(self._sav_path)

    # ------------------------------------------------------------------
    # Save states
    # ------------------------------------------------------------------

    def _state_path(self, slot: int) -> str:
        base = self._sav_path[:-4]   # strip ".sav"
        return f"{base}_s{slot}.state"

    def save_state(self, slot: int = 0) -> str:
        """Serialize the full console state to disk. Returns the file path."""
        state = {
            'cpu':    self.cpu.get_state(),
            'vdp':    self.vdp.get_state(),
            'psg':    self.psg.get_state(),
            'bus':    self.bus.get_state(),
            'joypad': self.joypad.get_state(),
        }
        path = self._state_path(slot)
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def load_state(self, slot: int = 0) -> bool:
        """Restore console state from disk. Returns True on success."""
        path = self._state_path(slot)
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
        except FileNotFoundError:
            return False
        self.cpu.set_state(state['cpu'])
        self.vdp.set_state(state['vdp'])
        self.psg.set_state(state['psg'])
        self.bus.set_state(state['bus'])
        self.joypad.set_state(state['joypad'])
        return True

    def step_frame(self) -> list:
        """Advance one video frame (262 scanlines) and return audio samples.

        The CPU and VDP are stepped together one scanline at a time so that
        VBlank and line interrupts are delivered between CPU instruction
        boundaries rather than at the end of the whole frame.

        Returns a list of (left, right) float pairs ([-1.0, +1.0] each) at
        SAMPLE_RATE Hz sized for exactly one 60 Hz frame (~735 samples).
        Stereo routing is controlled by the GG stereo register (port 0x06).
        """
        for _ in range(TOTAL_LINES):
            self.cpu.run_cycles(CYCLES_PER_LINE)
            self.vdp.step(CYCLES_PER_LINE)

        n_samples = round(self.SAMPLE_RATE / 60)
        return self.psg.render(n_samples, self.SAMPLE_RATE)
