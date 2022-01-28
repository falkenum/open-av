
import struct

with open(".\\res\\sounds\\enter-sandman.bin", mode="rb") as f:
    input = f.read(4)
    print(input)
    sample_rate = struct.unpack(">I", input)
    print(sample_rate)
