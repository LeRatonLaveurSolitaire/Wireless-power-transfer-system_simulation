# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:05:32 2023

@author: thoma
"""

import numpy as np
import matplotlib.pyplot as plt

def pre_emphasis_filter(z:float) -> float :
    return np.absolute(K * (1 - np.exp(-2 * np.pi * T_s * f_z) * z ** (-1)))


def de_emphasis_filter(z:float) -> float :
    return np.absolute(1 / (K * (1 - np.exp(-2 * np.pi * T_s * f_z) * z ** (-1))))

def main() -> None:
    nb_f = 1000
    K = 1
    f_z = 42_000
    T_s = 1 / 170_000

    fs = [100 * i for i in range(nb_f)]
    zs = [np.exp(1j * 2 * np.pi * f * T_s) for f in fs]
    pre_filter_output = []
    de_filter_output = []

    for z in zs:
        pre_filter_output.append(pre_emphasis_filter(z))
        de_filter_output.append(de_emphasis_filter(z))

    print(np.exp(-2 * np.pi * T_s * f_z))
    plt.title(f"module of emphasis filters with $f_s$ = {f_z//1000}kHz")
    plt.plot(fs, pre_filter_output, label="pre_emphasis")
    # plt.plot(fs, de_filter_output, label="de_emphasis")
    plt.legend()
    plt.xlabel("frequency (Hz)")
    plt.ylabel("Module")
    plt.show()

if __name__ == "__main__":
    main()