
import struct
import matplotlib.pyplot as plt
import numpy as np
import pickle

FFT_SIZE = 1024


with open(".\\res\\sounds\\enter-sandman.pickle", mode="rb") as f:
    data = pickle.load(f)
    print(data)

    freq = [i * float(data["sample_rate"]) / FFT_SIZE for i in range(FFT_SIZE)]

    plt.plot(freq, data["stft_output_db"][3000])
    plt.show()
