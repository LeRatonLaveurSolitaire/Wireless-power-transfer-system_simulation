import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import os

path_to_utils = os.path.join( sys.path[0],'..', '..', '..', 'utils')
sys.path.insert(0,path_to_utils)

from plot_function import bode_plot
import wpt_system_class as wpt


def open_mat_file(file_name):
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


def fractional_decade_smoothing_impedance(impedances, frequencies, fractional_factor):
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


def extract_noise(clean_signal, noisy_signal):
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
    phase_difference = np.angle(clean_fft) - np.angle(noisy_fft)

    # Compensate for phase difference in the noisy signal's FFT
    phase_compensated_noisy_fft = np.abs(noisy_fft) * np.exp(
        1j * (np.angle(noisy_fft) + phase_difference)
    )

    # Remove clean signal spectrum from compensated noisy signal spectrum
    noise_spectrum = phase_compensated_noisy_fft - clean_fft

    # Retrieve noise in time domain
    noise_signal_fft = np.fft.ifft(noise_spectrum)

    # Convert noise signal from frequency domain to time domain
    noise_signal = np.real(noise_signal_fft)

    return noise_signal

def div(a,b):
    return a/b

def main():
    """Main function of the script."""

    # Load the variables from the .mat file

    file_name = "sim_data/sim_values_row_PRBS7_start_up_values_with_voltage.mat"

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
        current.pop(0)
        voltage.pop(0)


    # FFT & filtering

    sampling_period = 1e-6
    fft_v = np.fft.fft(np.array(voltage[len(voltage)//5:]))
    fft_i = np.fft.fft(np.array(current[len(current)//5:]))


    fft_z = fft_v/fft_i  
    
    freqs = np.fft.fftfreq(n=len(fft_z), d=sampling_period)

    fft_z = fft_z[: len(fft_z) // 2]  # //2 to remove negatve frequency
    freqs_z = freqs[: len(freqs) // 2]  # //2 to remove negatve frequency

    # Cut the high frequency

    # while freqs_noise[-1] > 105000:
    #     freqs_noise = np.delete(freqs_noise,len(freqs_noise)-1)
    #     fft_noise = np.delete(fft_noise,len(fft_noise)-1)

    # Cut the low frequency

    # while freqs_noise[0] < 10000:
    #     freqs_noise = np.delete(freqs_noise, 0)
    #     fft_noise = np.delete(fft_noise, 0)

    # Applie the fractionnal decade smoothing technique

    smoothed_impedance = fractional_decade_smoothing_impedance(
        fft_z, freqs_z, 1.15
    )

    # Adjust static gain and phase

    static_gain = 1
    # adjust the phase so the mean between 75000 and 95000 is 0 wich should be the case in the real system
    #phase_gain = -np.mean(np.angle(smoothed_impedance[np.where((freqs_z  < 95_000) & (freqs_z > 75_000)) ])) 

    # smoothed_impedance = [
    #     i / static_gain * np.exp(1j * phase_gain) for i in smoothed_impedance
    # ]

    # impulse method

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

    # while freqs_impulse[0] < 10000:
    #     freqs_impulse = np.delete(freqs_impulse, 0)
    #     fft_impulse = np.delete(fft_impulse, 0)

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
    # C1 = 12.5 * 1e-9#1 / ((2 * np.pi * f0) ** 2 * L1)
    # R1 = 0.7
    # M = 11.2 * 1e-6
    # L2 = 4.82 * 1e-6
    # C2 = 29.2 * 1e-9#1 / ((2 * np.pi * f0) ** 2 * L2)
    # R2 = 0.4
    # R_l = 3.65

    primary_s = wpt.transmitter(L=L1, C_s=C1, R=R1)
    secondary_s = wpt.reciever(L=L2, C_s=C2, R=R2, R_l=R_l)

    wpt_system = wpt.total_system(
        transmitter=primary_s, reciever=secondary_s, M=M, name="model S-S"
    )

    # plot the 3 methods on a Bode diagram

    f_min = min(freqs_impulse[1], freqs_z[1])
    f_max = max(max(freqs_impulse), max(freqs_z))
    nb_samples = 2000
    bode_plot(
        systems=[wpt_system],
        f_min=f_min,
        f_max=f_max,
        nb_samples=nb_samples,
        f0=f0,
        samples=[fft_z, smoothed_impedance, 1/fft_impulse],#,fft_impulse],
        samples_frequency=[freqs_z, freqs_z, freqs_impulse],#,freqs_impulse],
        samples_names=["row fft", "smoothed & adjusted","Impulse"],#,"impulse"],
        title="Result with PRBS 7, PhD values with P&O identification",
    )


if __name__ == "__main__":
    main()