import matplotlib.pyplot as plt
import pandas
from scipy.fft import fft, fftfreq
import numpy as np


def pre_emphasis_filter(z:float)->float:
    T_s = 1e-6
    f_z = 42_000
    K = 1
    return K * (1 - np.exp(-2 * np.pi * T_s * f_z) * z ** (-1))


def extract_bit(number:int, index:int)->float:
    return (number & (1 << index)) >> index

def main()->None:
    PRBS_value = 0b10000000
    to_add = extract_bit(PRBS_value, 0) ^ extract_bit(PRBS_value, 1)
    values_list = []

    for i in range(127 * 10):
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
        PRBS_value = (PRBS_value >> 1) & 0x7F
        PRBS_value += to_add << 6


    only_fst_bit = [((value & 0x01) - 0.5) * 2 for value in values_list]


    plt.plot([i / 16 for i in range(len(only_fst_bit))], only_fst_bit)
    plt.show()

    PRBS_series = pandas.Series(only_fst_bit)

    # print("computing autocorrlation please wait")

    # PRBS_autocorr = [PRBS_series.autocorr(lag = i) for i in range(2**(10))]


    # print("computing done.")


    PRBS_period = 1 / (16 * 170_000)


    print("computing FFT please wait")
    PRBS_spec_dens = fft(only_fst_bit)

    print("computing done.")
    freq = fftfreq(n=len(PRBS_spec_dens), d=PRBS_period)

    PRBS_post_freq = []
    for i, fre in enumerate(freq):
        PRBS_post_freq.append(
            PRBS_spec_dens[i] * pre_emphasis_filter(np.exp(1j * 2 * np.pi * fre * 1e-6))
        )


    plt.title("Gain of the pre-emphasis filter")
    plt.plot(
        freq[: len(freq) // 4 * 2 // 3],
        [np.absolute(f) for f in PRBS_post_freq][: len(freq) // 4 * 2 // 3],
    )
    # plt.plot(freq[:len(freq)//4*2//3], [np.absolute(f) for f in PRBS_spec_dens][:len(freq)//4*2//3])
    # plt.plot(freq[:len(freq)//4*2//3], [np.absolute(pre_emphasis_filter(np.exp(1j*2*np.pi*f*1/170_000))) for f in freq][:len(freq)//4*2//3])
    plt.ylabel("spectral_density (f)")
    plt.xlabel("f")

    bandwidth_start = 50e3  # 75 kHz
    bandwidth_end = 144e3  # 100 kHz

    plt.fill_between(
        [bandwidth_start, bandwidth_end],
        0,
        np.max([np.absolute(f) for f in PRBS_post_freq]),
        color="r",
        alpha=0.3,
        label="Bandwidth (75 kHz - 95 kHz)",
    )


    plt.show()

if __name__ == "__main__":
    main()