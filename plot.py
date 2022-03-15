import cmath
from math import log, log2
import sdl2.ext as sdl
import time
import numpy as np
import colorsys

MAX_ITER = 80

def mandelbrot(c):
    z = 0
    n = 0
    while abs(z) <= 2 and n < MAX_ITER:
        z = z*z + c
        n += 1
    return n

def julia(z0):
    c = 0.285 + 0.01j
    z = z0
    n = 0
    while abs(z) <= 2 and n < MAX_ITER:
        z = z*z + c
        n += 1

    if n == MAX_ITER:
        return MAX_ITER
    
    return n + 1 - log(log2(abs(z)))


# Image size (pixels)
WIDTH = 640
HEIGHT = 480

# Plot window
RE_START = -2
RE_END = 1
IM_START = -1
IM_END = 1
zoom = 1.0


window_size= WIDTH, HEIGHT

window = sdl.Window("hello", window_size)
surface = window.get_surface()
pixels = sdl.pixels2d(surface)
# print(pixels.shape)

    
window.show()
# last_frame_time = time.time()


focus_point = 0 + 0j

while True:

    next_focus_point = cmath.inf + cmath.infj
    for x in range(0, WIDTH):
        for y in range(0, HEIGHT):
            # Convert pixel coordinate to complex number
            c = complex((RE_START + (x / WIDTH) * (RE_END - RE_START)) / zoom,
                        (IM_START + (y / HEIGHT) * (IM_END - IM_START)) / zoom)

            c_clip = c + focus_point

            m = mandelbrot(c_clip)
            # m = julia(c_clip)
            hue = m / MAX_ITER
            saturation = 1.0
            value = 1.0 if m < MAX_ITER else 0.0

            # center_to_xy = (x - WIDTH // 2)**2 + (y - HEIGHT // 2)**2
            # center_to_focus_point = (focus_point[0] - WIDTH // 2)**2 + (focus_point[1] - HEIGHT // 2)**2

            if m < MAX_ITER and abs(c_clip) < abs(next_focus_point):
                next_focus_point = c_clip

            rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)
            rgb_color = int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255)

            color = (rgb_color[0] << 16) | (rgb_color[1] << 8) | rgb_color[2] | (255 << 24)
            pixels[x][y] = color

    
    # for i in range(pixels.shape[0]):
    #     for j in range(pixels.shape[1]):
    #         pixels[i][j] = pixels_tmp[i][j]
    window.refresh()
    print("zoom", zoom)
    print("focus_point", focus_point)
    zoom += 1.0
    focus_point = next_focus_point

    # while time.time() - last_frame_time < 0.01:
    #     time.sleep(0.001)
    # last_frame_time = time.time()
    