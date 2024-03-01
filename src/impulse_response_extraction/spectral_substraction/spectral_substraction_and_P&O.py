import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import os

path_to_utils = os.path.join(sys.path[0], "..", "..", "..", "utils")
sys.path.insert(0, path_to_utils)

from plot_function import bode_plot
import wpt_system_class as wpt


def open_mat_file(file_name: str = None) -> np.array :
    """Function to open a .mat data file.

    Args:
        file_name (str): file path

    Returns:
        numpy.array: n-D numpy array containing the data
    """

    # Load the .mat file with the h5py library
    mat_file = h5py.File(file_name, "r")

    # Get variables from the .mat file
    variables_hdf5 = mat_file["values"]

    # Convert the HDF5 dataset to a NumPy array
    variables_array = variables_hdf5[()]

    # Close the file
    mat_file.close()

    return variables_array


def fractional_decade_smoothing_impedance(
    impedances: np.array = None,
    frequencies: np.array = None,
    fractional_factor: int = None,
) -> np.array :
    """Fractional decade smoothing.

    Args:
        impedances (array): complex impedance values
        frequencies (array): frequency corresponding to the impedance
        fractional_factor (int): smoothing factor

    Returns:
        array: smoothed impedance after filtering
    """
    smoothed_impedances = []
    for i, freq in enumerate(frequencies):
        lower_bound = freq / fractional_factor
        upper_bound = freq * fractional_factor

        # Find indices within the frequency range for smoothing
        indices = np.where((frequencies >= lower_bound) & (frequencies <= upper_bound))[
            0
        ]

        # Calculate the smoothed impedance within the range
        smoothed_impedance = np.mean(np.absolute(impedances[indices])) * np.exp(
            1j * np.mean(np.angle(impedances[indices]))
        )
        smoothed_impedances.append(smoothed_impedance)

    return np.array(smoothed_impedances)


def extract_noise(
    clean_current: list = None,
    noisy_current: list = None,
    clean_voltage: list = None,
    noisy_voltage: list = None,
) -> list:
    """Noise extraction function using spectrum substraction.
    This method also integrate a phase compensation technique.

    Working principle :
        * Compute FFT of the clean signal
        * Compute FFT of the noisy signal
        * Compute the phase difference between the signals
        * Adjuste the phase of the noisy signal
        * Substract the clean FFT to the noisy FFT in order to only keep the noisy spectrum
        * Compute the IFFT of the noise
        * Return the IFFT of the noise

    Args:
        clean_signal (list): Signal without noise.
        noisy_signal (list): Signal with noise (PRBS is injected).

    Returns:
        list: Noise extracted from the noisy signal.
    """
    # Perform FFT on the signals
    clean_fft_current = np.fft.fft(clean_current)
    noisy_fft_current = np.fft.fft(noisy_current)
    clean_fft_voltage = np.fft.fft(clean_voltage)
    noisy_fft_voltage = np.fft.fft(noisy_voltage)

    # Calculate impedance of both signals

    noisy_impedance = noisy_fft_voltage / noisy_fft_current
    clean_impedance = clean_fft_voltage / clean_fft_current

    system_impedance = noisy_impedance - clean_impedance
    # Remove clean signal spectrum from compensated noisy signal spectrum

    return system_impedance


def half(array: np.array = None) -> np.array :
    array = array[len(array) // 2:]
    return array


def main() -> None :
    """Main function of the script."""

    # Load the variables from the .mat file
    sampling_period = 1e-6
    file_name = "sim_data/sim_values_PRBS10_eperimental_setup_20mm.mat"

    variables = open_mat_file(file_name)

    time, prbs, current, voltage = [], [], [], []

    # split the variables in 3 : time, prbs and current

    for var in variables:
        time.append(var[0])
        prbs.append(var[1])
        current.append(var[2])
        voltage.append(var[3])

    # Trim the edges where no noise is injected, extract current with prbs injection, current without prbs injection

    while prbs[-1] == 0:
        prbs.pop()
        current.pop()
        voltage.pop()

    while prbs[0] == 0:
        prbs.pop(0)

    # extract current when prbs is injected
    current_prbs, voltage_prbs = [], []
    for i in range(len(prbs)):
        current_prbs.append(current[i - len(prbs)])
        voltage_prbs.append(voltage[i - len(prbs)])

    # extract clean current before PRBS injection
    current = current[len(current) - 2 * len(prbs) : len(current) - len(prbs)]
    voltage = voltage[len(voltage) - 2 * len(prbs) : len(voltage) - len(prbs)]

    # sys_impedance = extract_noise(clean_current = current, noisy_current = current_prbs, clean_voltage = voltage, noisy_voltage = voltage_prbs)
    fft_c = np.fft.fft(half(current))
    fft_c_prbs = np.fft.fft(half(current_prbs))
    fft_v = np.fft.fft(half(voltage))
    fft_v_prbs = np.fft.fft(half(voltage_prbs))
    freqs_to_print = np.fft.fftfreq(n=len(fft_c), d=sampling_period)
    freqs_to_print = freqs_to_print[: len(freqs_to_print) // 2]

    # plt.plot(freqs_to_print, 20 * np.log10(np.absolute(fft_c[:len(freqs_to_print)])))
    # plt.plot(freqs_to_print,  20 * np.log10(np.absolute(fft_c_prbs[:len(freqs_to_print)])))
    # plt.xscale("log")
    # plt.grid()
    # plt.show()

    # plt.plot(freqs_to_print, 20 * np.log10(np.absolute(fft_v[:len(freqs_to_print)])))
    # plt.plot(freqs_to_print, 20 * np.log10(np.absolute(fft_v_prbs[:len(freqs_to_print)])))
    # plt.xscale("log")
    # plt.grid()
    # plt.show()
    x = 0
    sys_phase = (fft_v_prbs) / (fft_c_prbs)
    sys_gain = 1 / (fft_c_prbs - fft_c)
    # print(np.where((freqs_to_print > 80_000) & (freqs_to_print < 90_000)))
    # gain_factor = np.absolute(sys_phase[[np.where((freqs_to_print > 80_000) & (freqs_to_print < 90_000) )][0]])/np.absolute(sys_gain[[np.where((freqs_to_print > 80_000) & (freqs_to_print < 90_000) )][0]])
    # print(gain_factor,np.shape(gain_factor))
    sys_gain *= 200  # gain_factor

    sys_impedance = []
    for i in range(len(sys_gain)):
        gain = sys_gain[i]
        phase = sys_phase[i]
        sys_impedance.append(np.absolute(gain) * np.exp(1j * np.angle(phase)))
    sys_impedance = np.array(sys_impedance)

    # sys_impedance = 100 / (fft_c_prbs - fft_c)
    # sys_impedance = 6000 / (
    #     fft_c_prbs - fft_c
    # )  # (fft_v_prbs / fft_c_prbs)# - (fft_v/fft_c)

    # Plot the clean, noisy and extracted noise signals
    # plt.figure(figsize=(10, 6))
    # plt.subplot(3, 1, 1)
    # plt.title("Clean current")
    # plt.plot([i for i in range(len(current))], current)

    # plt.subplot(3, 1, 2)
    # plt.title("Current with PRBS injected")
    # plt.plot([i for i in range(len(current_prbs))], current_prbs)

    # plt.subplot(3, 1, 3)
    # plt.title("Extracted Noise")
    # plt.plot([i for i in range(len(noise))], noise)

    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    # FFT & filtering

    # fft_noise = np.fft.fft(
    #     noise[len(noise) // 4 :]
    # )  # Compute FFT with impulse response truncation for better results
    freqs = np.fft.fftfreq(n=len(sys_impedance), d=sampling_period)

    sys_fft = sys_impedance[
        : len(sys_impedance) // 2
    ]  # //2 to remove negatve frequency
    freqs = freqs[: len(freqs) // 2]  # //2 to remove negatve frequency

    # Cut the high frequency

    # while freqs_noise[-1] > 105000:
    #     freqs_noise = np.delete(freqs_noise,len(freqs_noise)-1)
    #     fft_noise = np.delete(fft_noise,len(fft_noise)-1)

    # Cut the low frequency

    while freqs[0] < 10000:
        freqs = np.delete(freqs, 0)
        sys_fft = np.delete(sys_fft, 0)

    # Applie the fractionnal decade smoothing technique

    smoothed_impedance = fractional_decade_smoothing_impedance(sys_fft, freqs, 1.05)

    # phase_gain = -np.mean(
    #     np.angle(
    #         smoothed_impedance[
    #             np.where((freqs < 95_000) & (freqs > 75_000))
    #         ]
    #     )
    # )

    smoothed_impedance = [
        impedance for impedance in smoothed_impedance  # * np.exp(1j * phase_gain)
    ]

    # Impulse method

    file_name = "sim_data/impulse_values_start-up.mat"

    variables = open_mat_file(file_name)

    time, voltage, current = [], [], []

    for var in variables:
        time.append(var[0])
        voltage.append(var[1])
        current.append(var[2] / 2.5)

    while current[0] == 0:
        voltage.pop(0)
        current.pop(0)

    fft_impulse = np.fft.fft(np.array(current))
    freqs_impulse = np.fft.fftfreq(n=len(fft_impulse), d=1e-6)

    fft_impulse = fft_impulse[
        : len(fft_impulse) // 2
    ]  # //2 to remove negatives frequencies
    freqs_impulse = freqs_impulse[
        : len(freqs_impulse) // 2
    ]  # //2 to remove negatives frequencies

    # Cut the high frequency for ploting

    # while freqs_impulse[-1] > 105000:
    #     freqs_impulse = np.delete(freqs_impulse,len(freqs_impulse)-1)
    #     fft_impulse = np.delete(fft_impulse,len(fft_impulse)-1)

    # Cut the low frequency for ploting

    while freqs_impulse[0] < 10000:
        freqs_impulse = np.delete(freqs_impulse, 0)
        fft_impulse = np.delete(fft_impulse, 0)

    # Model base bode plot

    # Experimental values

    f0 = 85000
    L1 = 24 * 1e-6
    C1 = 1 / ((2 * np.pi * f0) ** 2 * L1)
    R1 = 0.075
    # M = 16.27 * 1e-6 # for a 5mm gap
    M = 5.2345e-06 # for a 20mm gap
    L2 = 24 * 1e-6
    C2 = 1 / ((2 * np.pi * f0) ** 2 * L2)
    R2 = 0.075
    R_l = 1.25


    primary_s = wpt.transmitter(L=L1, C_s=C1, R=R1)
    secondary_s = wpt.reciever(L=L2, C_s=C2, R=R2, R_l=R_l)

    wpt_system = wpt.total_system(
        transmitter=primary_s, reciever=secondary_s, M=M, name="model S-S"
    )

    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    # print(freqs_noise)
    # plot the 3 methods on a Bode diagram

    f_min = freqs[1]
    f_max = max(max(freqs_impulse), max(freqs))
    nb_samples = 20000
    bode_plot(
        systems=[wpt_system],
        f_min=f_min,
        f_max=f_max,
        nb_samples=nb_samples,
        f0=f0,
        samples=[
            #sys_fft,
            np.array(smoothed_impedance),
            #1 / np.array(fft_impulse),
        ],
        samples_frequency=[
            #freqs,
            freqs,
            #freqs_impulse,
        ],
        samples_names=[
            #"raw",
            "estimation",
            #"impulse response",
        ],
        title="Result with PRBS 10, experimental values",
    )


if __name__ == "__main__":
    main()
