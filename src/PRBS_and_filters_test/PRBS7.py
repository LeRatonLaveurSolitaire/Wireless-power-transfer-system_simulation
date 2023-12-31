import matplotlib.pyplot as plt
import pandas
from scipy.fft import fft, fftfreq
import numpy as np

def extract_bit(number, index):
    return (number & (1 << index)) >> index

PRBS_value = 0b0100000
to_add = extract_bit(PRBS_value, 0)^extract_bit(PRBS_value, 1)
values_list = []

for i in range (130):
    values_list.append(PRBS_value)
    values_list.append(PRBS_value)
    values_list.append(PRBS_value)
    values_list.append(PRBS_value)

    PRBS_value = (PRBS_value  >> 1) & 0x7F
    to_add = extract_bit(PRBS_value, 0)^extract_bit(PRBS_value, 1)
    PRBS_value += (to_add << 7)
    


only_fst_bit = [10 * ((value & 0x01) - 0.5) for value in values_list]


plt.plot([i/4 for i in range(len(only_fst_bit))],only_fst_bit)
plt.show()

# PRBS_series = pandas.Series(only_fst_bit)

# print("computing autocorrlation please wait")

# PRBS_autocorr = [PRBS_series.autocorr(lag = i) for i in range(2**(10))]


# print("computing done.")


# PRBS_period = 1/( 8 *85000)




# print("computing FFT please wait")
# PRBS_spec_dens = fft(np.array(PRBS_autocorr))

# print("computing done.")
# freq = fftfreq(n = len(PRBS_spec_dens), d=PRBS_period)

# plt.title("PRBS Spectral density with 170kHz clock frequency")
# plt.plot(freq, [np.absolute(f) for f in PRBS_spec_dens])
# plt.ylabel("spectral_density (f)")
# plt.xlabel("f")

# bandwidth_start = 70e3  # 75 kHz
# bandwidth_end = 100e3    # 100 kHz

# plt.fill_between([bandwidth_start, bandwidth_end], 0, np.max(PRBS_spec_dens), color='r', alpha=0.3, label='Bandwidth (75 kHz - 95 kHz)')


# plt.show()