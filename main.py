"""PyGear — Sega Game Gear emulator entry point.

Usage:
    python main.py roms/game.gg [--scale N]

Default scale is 3 (480×432 window).  Press Escape or close the window to quit.
"""

import argparse
import sys

import numpy as np
import pygame

from pygear.cartridge import Cartridge
from pygear.console import GameGearConsole
from pygear.io.joypad import UP, DOWN, LEFT, RIGHT, BUTTON1, BUTTON2, START
from pygear.vdp.vdp import SCREEN_W, SCREEN_H

DEFAULT_SCALE = 3

KEY_MAP = {
    pygame.K_UP:     UP,
    pygame.K_DOWN:   DOWN,
    pygame.K_LEFT:   LEFT,
    pygame.K_RIGHT:  RIGHT,
    pygame.K_z:      BUTTON1,
    pygame.K_x:      BUTTON2,
    pygame.K_RETURN: START,
}


def frame_to_surface(frame: list, scale: int) -> pygame.Surface:
    """Convert a 144×160 list-of-(R,G,B) frame to a scaled pygame Surface."""
    arr  = np.array(frame, dtype=np.uint8)                  # (144, 160, 3)
    surf = pygame.surfarray.make_surface(arr.transpose(1, 0, 2))  # (160, 144)
    if scale != 1:
        surf = pygame.transform.scale(surf, (SCREEN_W * scale, SCREEN_H * scale))
    return surf


def main() -> None:
    parser = argparse.ArgumentParser(description="PyGear — Sega Game Gear emulator")
    parser.add_argument("rom",   help="path to .gg ROM file")
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE,
                        help="display scale factor (default: 3)")
    args = parser.parse_args()

    if args.scale < 1:
        parser.error("--scale must be at least 1")

    pygame.mixer.pre_init(44100, -16, 1, 512)
    pygame.init()

    scale  = args.scale
    screen = pygame.display.set_mode((SCREEN_W * scale, SCREEN_H * scale))
    pygame.display.set_caption("PyGear")
    clock = pygame.time.Clock()

    try:
        cart = Cartridge(args.rom)
    except FileNotFoundError:
        print(f"ROM not found: {args.rom}", file=sys.stderr)
        pygame.quit()
        sys.exit(1)

    console   = GameGearConsole(cart)
    audio_ch  = pygame.mixer.Channel(0)

    running = True
    while running:
        # --- Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in KEY_MAP:
                    console.joypad.press(KEY_MAP[event.key])
            elif event.type == pygame.KEYUP:
                if event.key in KEY_MAP:
                    console.joypad.release(KEY_MAP[event.key])

        # --- Emulate one frame ---
        audio = console.step_frame()

        # --- Video ---
        if console.vdp.frame_ready:
            console.vdp.frame_ready = False
            surf = frame_to_surface(console.vdp.frame, scale)
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        # --- Audio ---
        arr   = np.clip(np.array(audio) * 32767, -32767, 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(arr.reshape(-1, 1))
        if not audio_ch.get_busy():
            audio_ch.play(sound)

        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
