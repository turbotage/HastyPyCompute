
import numpy as np
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt

filter = gaussian(160, 10)

plt.figure()
plt.plot(filter)
plt.show()

fft_filter = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(filter)))

plt.figure()
plt.plot(fft_filter)
plt.show()


