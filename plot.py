
import struct
import matplotlib.pyplot as plt
import numpy as np
import pickle


with open(".\\res\\sounds\\enter-sandman.pickle", mode="rb") as f:
    data = pickle.load(f)
    FFT_SIZE = len(data["stft_output_db"][0] * 2)
    # print(data)

    freq = [i * float(data["sample_rate"]) / FFT_SIZE for i in range(int(FFT_SIZE/2))]

    plt.plot(freq, data["stft_output_db"][3000])
    plt.show()
