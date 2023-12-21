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


def X_corr_function(var1, var2):
    """Cross-correlation function.

    Args:
        var1 (list): 1st variable (PRBS in this case)
        var2 (list): 2nd variable (I_err in this case)

    Raises:
        ValueError: In case var1 and var2 don't have the same size.

    Returns:
        list: list of the cross-correlation values.
    """
    
    if len(var1) != len(var2) and var1 % 2 == 0:
        raise ValueError(
            "Error in cross_correlation, the variables don't have the same size."
        )

    PRBS_amplitude = abs(var1[0]) * 4000

    X_corr = []
    N = len(var1) // 2
    for n in range(N):
        g = 0
        for m in range(N,2*N):
            g += 1 / (N) * var1[(N - n + m) % len(var1)] * var2[m]
        X_corr.append(g/(PRBS_amplitude))

    return X_corr

def apply_de_emphasis_filter(array,t_s,f_z):
    """Apply the de-emphsasis filter on "array" after calculating filter coefficient.

    Args:
        array (list): array to filter
        t_s (int): sampling period
        f_z (int): corner frequency

    Returns:
        list: filtered array
    """

    coeff = [] # coefficient of the filter truncated at 0.01 in time domain
    z1 = np.exp(-2 * np.pi * t_s*f_z) # coefficient of the filter in the Z domain

    # computation of the coefficients in time domain
    while (z1 **(len(coeff)) > 0.01):
        coeff.append(z1**(len(coeff)))

    # new value after filtering
    post_filter = []

    # computation of the new value after convolution with the filter
    for i in range(len(array)):
        new_value = 0
        for j in range(len(coeff)):
            if i-j  >=0:
                new_value+= coeff[j] * array[i-j]
        post_filter.append(new_value)

    return post_filter

def apply_pre_emphasis_filter(array,t_s,f_z):
    """Apply the pre-emphsasis filter on "array" after calculating filter coefficient.

    Args:
        array (list): array to filter
        t_s (int): sampling period
        f_z (int): corner frequency

    Returns:
        list: filtered array
    """

    coeff = [] # coefficient of the filter truncated at 0.01 in time domain
    z1 = np.exp(-2 * np.pi * t_s*f_z) # coefficient of the filter in the Z domain

    # computation of the coefficients in time domain
    while (z1 **(len(coeff)) > 0.01):
        coeff.append((len(coeff)==0) - z1**(len(coeff)))

    # new value after filtering
    post_filter = []

    # computation of the new value after convolution with the filter
    for i in range(len(array)):
        new_value = 0
        for j in range(len(coeff)):
            if i-j  >=0:
                new_value+= coeff[j] * array[i-j]
        post_filter.append(new_value)

    return post_filter


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
        indices = np.where((frequencies >= lower_bound) & (frequencies <= upper_bound))[0]
        
        # Calculate the smoothed impedance within the range
        smoothed_impedance = np.mean(np.absolute(impedances[indices]))*np.exp(1j* np.mean(np.angle(impedances[indices])))
        smoothed_impedances.append(smoothed_impedance)
    
    return np.array(smoothed_impedances)

def main():
    """Main function of the script."""

    # Load the variables from the .mat file 

    file_name = "sim_data/sim_values_fs_680000.mat"

    variables = open_mat_file(file_name)

    time, prbs, current = [], [], []

    # split the variables in 3 : time, prbs and current

    for var in variables:
        time.append(var[0])
        prbs.append(var[1])
        current.append(var[2])
    
    # Trim the edges where no noise is injected

    while prbs[0] == 0:
        prbs.pop(0)
        current.pop(0)

    while prbs[-1] == 0:
        prbs.pop()
        current.pop()

    # Cross_correlation method

    #current = apply_pre_emphasis_filter(current,1e-6,42000)

    X_corr = X_corr_function(prbs, current)

    sampling_period = 1/680000
    
    plt.plot([i*sampling_period for i in range(len(X_corr))],X_corr)
    plt.show()

    fft_X_corr = np.fft.fft(np.array(X_corr))
    freqs = np.fft.fftfreq(n=len(fft_X_corr), d=sampling_period)

    fft_X_corr = fft_X_corr[: len(fft_X_corr) // 2]     # //2 to remove negatve frequency
    freqs_X_corr = freqs[: len(freqs) // 2]             # //2 to remove negatve frequency

    # Cut the high frequency

    # while freqs_X_corr[-1] > 105000:
    #     freqs_X_corr = np.delete(freqs_X_corr,len(freqs_X_corr)-1)
    #     fft_X_corr = np.delete(fft_X_corr,len(fft_X_corr)-1)

    # Cut the low frequency

    # while freqs_X_corr[0] < 10000:
    #     freqs_X_corr = np.delete(freqs_X_corr, 0)
    #     fft_X_corr = np.delete(fft_X_corr, 0)

    # Applie the fractionnal decade smoothing technique

    smoothed_impedance = fractional_decade_smoothing_impedance(fft_X_corr,freqs_X_corr,1.07)
    

    # impulse method

    file_name = "sim_data/impulse_values_start-up.mat"

    variables = open_mat_file(file_name)

    time, voltage, current = [], [], []

    for var in variables:
        time.append(var[0])
        voltage.append(var[1])
        current.append(var[2]/2.5)

    while current[0] == 0:
        voltage.pop(0)
        current.pop(0)
    T_s = 1e-6
    fft_impulse = np.fft.fft(np.array(current))
    freqs = np.fft.fftfreq(n=len(fft_impulse), d=T_s)

    fft_impulse = fft_impulse[: len(fft_impulse) // 2]  # //2 to remove negatives frequencies
    freqs_impulse = freqs[: len(freqs) // 2]            # //2 to remove negatives frequencies

    # Cut the high frequency

    # while freqs_impulse[-1] > 105000:
    #     freqs_impulse = np.delete(freqs_impulse,len(freqs_impulse)-1)
    #     fft_impulse = np.delete(fft_impulse,len(fft_impulse)-1)

    # Cut the low frequency

    # while freqs_impulse[0] < 10000:
    #     freqs_impulse = np.delete(freqs_impulse, 0)
    #     fft_impulse = np.delete(fft_impulse, 0)

    # Model base bode plot

    f0 = 85000
    L1 = 236 * 1e-6
    C1 = 1 / ((2 * np.pi * f0) ** 2 * L1)
    R1 = 0.7
    M = 11.2 * 1e-6
    L2 = 4.82 * 1e-6
    C2 = 1 / ((2 * np.pi * f0) ** 2 * L2)
    R2 = 0.4
    R_l = 0.7

    primary_s = wpt.transmitter(L=L1, C_s=C1, R=R1)
    secondary_s = wpt.reciever(L=L2, C_s=C2, R=R2, R_l=R_l)

    wpt_system = wpt.total_system(
        transmitter=primary_s, reciever=secondary_s, M=M, name="model S-S"
    )

    # plot the 3 methods on a Bode diagram
    # freqs_impulse = np.delete(freqs_impulse,0)
    # freqs_X_corr = np.delete(freqs_X_corr,0)
    f_min = min(freqs_impulse[1],freqs_X_corr[1])
    f_max = max(max(freqs_impulse),max(freqs_X_corr))
    nb_samples = 1000
    bode_plot(
        systems=[wpt_system],
        f_min=f_min,
        f_max=f_max,
        nb_samples=nb_samples,
        f0=f0,
        samples=[1/np.array(fft_X_corr), 1/np.array(fft_impulse),1/np.array(smoothed_impedance)],
        samples_frequency=[freqs_X_corr, freqs_impulse,freqs_X_corr],
        samples_names=["X_corr", "impulse","smoothed"],
    )


if __name__ == "__main__":
    main()
