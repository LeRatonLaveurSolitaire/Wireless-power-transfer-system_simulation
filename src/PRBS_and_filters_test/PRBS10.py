import matplotlib.pyplot as plt
import pandas
from scipy.fft import fft, fftfreq
import numpy as np
import csv
import emphasis_filters


def extract_bit(number, index):
    return (number & (1 << index)) >> index


PRBS_value = 0b1
values_list = []


# PRBS10

# for i in range (1024 * 2**5):
#     values_list.append(PRBS_value)
#     values_list.append(PRBS_value)
#     values_list.append(PRBS_value)
#     values_list.append(PRBS_value)
#     to_add = extract_bit(PRBS_value, 0)^extract_bit(PRBS_value, 3)
#     PRBS_value = (PRBS_value  >> 1) & 0b1111111111
#     PRBS_value += (to_add <<9)


# PRBS7

for i in range(128 * 2**2):
    values_list.append(PRBS_value)
    values_list.append(PRBS_value)
    values_list.append(PRBS_value)
    values_list.append(PRBS_value)
    values_list.append(PRBS_value)
    values_list.append(PRBS_value)
    values_list.append(PRBS_value)
    values_list.append(PRBS_value)
    values_list.append(PRBS_value)
    values_list.append(PRBS_value)
    values_list.append(PRBS_value)
    values_list.append(PRBS_value)
    values_list.append(PRBS_value)
    values_list.append(PRBS_value)
    values_list.append(PRBS_value)
    values_list.append(PRBS_value)
    to_add = extract_bit(PRBS_value, 0) ^ extract_bit(PRBS_value, 1)
    PRBS_value = (PRBS_value >> 1) & 0b1111111
    PRBS_value += to_add << 6


only_fst_bit = [2 * ((value & 0x01) - 0.5) for value in values_list]

print(values_list)

# plt.plot([i for i in range(len(values_list))],values_list,"x")
# plt.show()

PRBS_series = pandas.Series(only_fst_bit)

print("computing autocorrlation please wait")

# PRBS_autocorr = [PRBS_series.autocorr(lag = i) for i in range(2**(11))]


print("computing done.")


PRBS_period = 1 / (16 * 2 * 85000)


print("computing FFT please wait")
PRBS_abs_values = [np.absolute(value) for value in values_list]
PRBS_spec_dens = fft(np.array(PRBS_abs_values))

print("computing done.")
freq = fftfreq(n=len(PRBS_spec_dens), d=PRBS_period)

PRBS_post_filter = []

with open("PRBS10_spectral_density.csv", "w") as csv_file:
    filewriter = csv.writer(csv_file)
    for i in range(len(PRBS_spec_dens)):
        f = freq[i]
        PRBS_dens = PRBS_spec_dens[i]
        z = np.exp(1j * 2 * np.pi * f * emphasis_filters.T_s)
        post_filter = PRBS_dens * emphasis_filters.pre_emphasis_filter(z)
        PRBS_post_filter.append(post_filter)
        filewriter.writerow([f, PRBS_dens, post_filter])


plt.title("PRBS Spectral density with 170kHz clock frequency")
plt.plot(freq, [np.absolute(f) for f in PRBS_spec_dens], ".", label="PRBS")
plt.plot(
    freq, [np.absolute(f) for f in PRBS_post_filter], ".", label="PRBS post-filter"
)
plt.ylabel("spectral_density (f)")
plt.xlabel("f")
plt.legend()
bandwidth_start = 70e3  # 75 kHz
bandwidth_end = 100e3  # 100 kHz
plt.fill_between(
    [bandwidth_start, bandwidth_end],
    0,
    np.max(PRBS_spec_dens),
    color="r",
    alpha=0.3,
    label="Bandwidth (75 kHz - 95 kHz)",
)

plt.show()

# plt.title("PRBS spectral density after pre-emphasis  filter with 170kHz clock frequency")
# plt.plot(freq, [np.absolute(f) for f in PRBS_post_filter],".",label="PRBS post-filter")
# plt.ylabel("spectral_density (f)")
# plt.xlabel("f")
# plt.legend()
# bandwidth_start = 70e3  # 75 kHz
# bandwidth_end = 100e3    # 100 kHz
# plt.fill_between([bandwidth_start, bandwidth_end], 0, np.max(PRBS_spec_dens), color='r', alpha=0.3, label='Bandwidth (75 kHz - 95 kHz)')

# plt.show()
