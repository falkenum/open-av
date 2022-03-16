import cmath
from math import log, log2
import sdl2.ext as sdl
import time
import numpy as np

# Image size (pixels)
WIDTH = 640
HEIGHT = 480

zoom = 1.0


window_size= WIDTH, HEIGHT

window = sdl.Window("hello", window_size)
surface = window.get_surface()
pixels = sdl.pixels2d(surface)
# print(pixels.shape)

    
window.show()
# last_frame_time = time.time()



COLOR_WHITE = 255, 255, 255
COLOR_BLACK = 0, 0, 0

while True:

    for x in range(0, WIDTH):
        for y in range(0, HEIGHT):

            # rgb_color =
            # color = (rgb_color[0] << 16) | (rgb_color[1] << 8) | rgb_color[2] | (255 << 24)
            # pixels[x][y] = color
            pass

    window.refresh()

    # while time.time() - last_frame_time < 0.01:
    #     time.sleep(0.001)
    # last_frame_time = time.time()
    