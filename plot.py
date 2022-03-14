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

# Image size (pixels)
WIDTH = 640
HEIGHT = 480

# Plot window
CENTER_RE = 0.0
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

last_frame_time = time.time()
while True:
    for x in range(0, WIDTH):
        for y in range(0, HEIGHT):
            # Convert pixel coordinate to complex number
            c = complex((RE_START + CENTER_RE + (x / WIDTH) * (RE_END - RE_START)) / zoom,
                        (IM_START + (y / HEIGHT) * (IM_END - IM_START)) / zoom)
            # Compute the number of iterations
            m = mandelbrot(c)
            # The color depends on the number of iterations
            hue = m / MAX_ITER
            saturation = 1.0
            value = 1.0 if m < MAX_ITER else 0

            rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)
            rgb_color = int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255)
            color = (rgb_color[0] << 16) | (rgb_color[1] << 8) | rgb_color[2] | (255 << 24)
            pixels[x][y] = color
    
    zoom += 0.2
    window.refresh()

    # while time.time() - last_frame_time < 0.01:
    #     time.sleep(0.001)
    # last_frame_time = time.time()
    