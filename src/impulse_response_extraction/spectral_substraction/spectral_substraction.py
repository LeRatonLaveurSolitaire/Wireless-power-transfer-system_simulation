import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import os

path_to_utils = os.path.join(sys.path[0], "..", "..", "..", "utils")
sys.path.insert(0, path_to_utils)

from plot_function import bode_plot
import wpt_system_class as wpt


def open_mat_file(file_name) -> np.array :
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


def fractional_decade_smoothing_impedance(impedances, frequencies, fractional_factor) -> np.array :
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


def extract_noise(clean_signal, noisy_signal) -> (np.array,np.array) :
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
    clean_fft = np.fft.fft(clean_signal)
    noisy_fft = np.fft.fft(noisy_signal)

    # Calculate phase difference between clean and noisy signals
    # phase_difference = np.angle(noisy_fft) - np.angle(clean_fft)

    # Compensate for phase difference in the noisy signal's FFT
    # phase_compensated_noisy_fft = np.abs(noisy_fft) * np.exp(
    #     1j * (np.angle(noisy_fft) - phase_difference)
    # )

    # Remove clean signal spectrum from compensated noisy signal spectrum
    noise_spectrum = noisy_fft - clean_fft

    # Retrieve noise in time domain
    noise_signal_fft = np.fft.ifft(noise_spectrum)
    print(noise_signal_fft)
    # Convert noise signal from frequency domain to time domain
    noise_signal = np.real(noise_signal_fft)

    return noise_signal, noise_spectrum


def main() -> None :
    """Main function of the script."""

    # Load the variables from the .mat file

    file_name = "sim_data/sim_values_row_PRBS10_start-up_values.mat"

    variables = open_mat_file(file_name)

    time, prbs, current = [], [], []

    # split the variables in 3 : time, prbs and current

    for var in variables:
        time.append(var[0])
        prbs.append(var[1])
        current.append(var[2])

    # Trim the edges where no noise is injected, extract current with prbs injection, current without prbs injection

    while prbs[-1] == 0:
        prbs.pop()
        current.pop()

    while prbs[0] == 0:
        prbs.pop(0)

    # extract current when prbs is injected
    current_prbs = []
    for i in range(len(prbs)):
        current_prbs.append(current[i - len(prbs)])

    # extract clean current before PRBS injection
    current = current[len(current) - 2 * len(prbs) : len(current) - len(prbs)]

    noise, noise_spectrum = extract_noise(clean_signal=current[len(current)//2:], noisy_signal=current_prbs[len(current_prbs)//2:])

    # Plot the clean, noisy and extracted noise signals
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.title("Clean current")
    plt.plot([i for i in range(len(current))], current)

    plt.subplot(3, 1, 2)
    plt.title("Current with PRBS injected")
    plt.plot([i for i in range(len(current_prbs))], current_prbs)

    plt.subplot(3, 1, 3)
    plt.title("Extracted Noise")
    plt.plot([i for i in range(len(noise))], noise)

    plt.grid()
    plt.tight_layout()
    plt.show()

    # FFT & filtering

    sampling_period = 1e-6

    fft_noise = np.fft.fft(
        noise[:]
    )  # Compute FFT with impulse response truncation for better results
    freqs = np.fft.fftfreq(n=len(noise_spectrum), d=sampling_period)

    fft_noise = fft_noise[: len(noise_spectrum) // 2]  # //2 to remove negatve frequency
    freqs_noise = freqs[: len(freqs) // 2]  # //2 to remove negatve frequency

    # Cut the high frequency

    # while freqs_noise[-1] > 105000:
    #     freqs_noise = np.delete(freqs_noise,len(freqs_noise)-1)
    #     fft_noise = np.delete(fft_noise,len(fft_noise)-1)

    # Cut the low frequency

    while freqs_noise[0] < 10000:
        freqs_noise = np.delete(freqs_noise, 0)
        fft_noise = np.delete(fft_noise, 0)

    # Applie the fractionnal decade smoothing technique

    smoothed_impedance = fractional_decade_smoothing_impedance(
        fft_noise, freqs_noise, 1.05
    )

    # Adjust static gain and phase

    static_gain = 6000

    # Adjust the phase so the mean between 75000 and 95000 is 0 wich should be the case in the real system

    phase_gain = -np.mean(
        np.angle(
            smoothed_impedance[
                np.where((freqs_noise < 95_000) & (freqs_noise > 75_000))
            ]
        )
    )

    smoothed_impedance = [
        impedance / static_gain * np.exp(1j * phase_gain)
        for impedance in smoothed_impedance
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
    freqs = np.fft.fftfreq(n=len(fft_impulse), d=sampling_period)

    fft_impulse = fft_impulse[
        : len(fft_impulse) // 2
    ]  # //2 to remove negatives frequencies
    freqs_impulse = freqs[: len(freqs) // 2]  # //2 to remove negatives frequencies

    # Cut the high frequency for ploting

    # while freqs_impulse[-1] > 105000:
    #     freqs_impulse = np.delete(freqs_impulse,len(freqs_impulse)-1)
    #     fft_impulse = np.delete(fft_impulse,len(fft_impulse)-1)

    # Cut the low frequency for ploting

    while freqs_impulse[0] < 10000:
        freqs_impulse = np.delete(freqs_impulse, 0)
        fft_impulse = np.delete(fft_impulse, 0)

    # Model base bode plot

    # Start-up values
    f0 = 85000
    L1 = 236 * 1e-6
    C1 = 1 / ((2 * np.pi * f0) ** 2 * L1)
    R1 = 0.7
    M = 11.2 * 1e-6
    L2 = 4.82 * 1e-6
    C2 = 1 / ((2 * np.pi * f0) ** 2 * L2)
    R2 = 0.4
    R_l = 0.7

    # PhD values
    # f0 = 85000
    # L1 = 280.5 * 1e-6
    # C1 = 12.5 * 1e-9
    # R1 = 0.6
    # M = 14.3 * 1e-6
    # L2 = 120 * 1e-6
    # C2 = 29.2 * 1e-9 
    # R2 = 0.4
    # R_l = 3.6

    primary_s = wpt.transmitter(L=L1, C_s=C1, R=R1)
    secondary_s = wpt.reciever(L=L2, C_s=C2, R=R2, R_l=R_l)

    wpt_system = wpt.total_system(
        transmitter=primary_s, reciever=secondary_s, M=M, name="model S-S"
    )

    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    # print(freqs_noise)
    # plot the 3 methods on a Bode diagram

    f_min = freqs_noise[1]
    f_max = max(max(freqs_impulse), max(freqs_noise))
    nb_samples = 20000
    bode_plot(
        systems=[wpt_system],
        f_min=f_min,
        f_max=f_max,
        nb_samples=nb_samples,
        f0=f0,
        samples=[
            #static_gain / np.array(fft_noise),
            1 / np.array(smoothed_impedance),
            1 / np.array(fft_impulse),
        ],
        samples_frequency=[
            #freqs_noise, 
            freqs_noise, 
            freqs_impulse,
            ],
        samples_names=[
            #"raw", 
            "estimation", 
            "impulse response",
            ],
        title="Result with PRBS 10, start-up values",
    )


if __name__ == "__main__":
    main()
