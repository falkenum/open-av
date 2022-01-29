
import struct
import matplotlib.pyplot as plt
import numpy as np

FFT_SIZE = 1024

with open(".\\res\\sounds\\enter-sandman.bin", mode="rb") as f:
    sample_rate, = struct.unpack(">I", f.read(4))

    stft_mag_db = []
    b = f.read(4)
    while len(b) > 0:
        fft_mag_db = []
        for i in range(FFT_SIZE):
            val, = struct.unpack(">f", b)
            fft_mag_db.append(val)
            b = f.read(4)
        stft_mag_db.append(fft_mag_db)
    
    freq = [i * float(sample_rate) / FFT_SIZE for i in range(FFT_SIZE)]

    plt.yscale("log")
    # print(len(stft_mag_db))
    plt.plot(freq, stft_mag_db[50])
    plt.show()
