"""Script that open a sim data file, perform a frequency response extraction and parameters identification using a NN."""
import numpy as np
import h5py
import csv
import torch
from torch import tensor
import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

path_to_utils = os.path.join(sys.path[0], "..", "..", "utils")  # add /utils/ to path
sys.path.insert(-1, path_to_utils)

from plot_function import bode_plot
import wpt_system_class as wpt

path_to_models = os.path.join(sys.path[0], "..", "..", "neural_network_experiment")  # add /models. to path
sys.path.insert(-1, path_to_models)

from nn_model_4_128 import NN_model


def delinearise_R_l(R_l: float = 0) -> float :
    """Recover R from the output data of the NN."""
    return 10 ** (R_l * 0.1)


def delinearise_M(M: float = 0) -> float :
    """Recover M from the output data of the NN."""
    L1 = 24e-6
    L2 = 24e-6
    return 10 ** ((M * 0.1)) * (0.1 * (L1 * L2) ** 0.5)


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


def extract_impedance(
    clean_current: list = None,
    noisy_current: list = None,
    clean_voltage: list = None,
    noisy_voltage: list = None,
) -> np.array :
    """Extract the system impedance using the combination of spectral substraction (Gain) and simple impedance computation (Phase).

    Args:
        clean_current (list): clean current. Defaults to None.
        noisy_current (list): currrent with PRBS injection. Defaults to None.
        clean_voltage (list): clean voltage. Defaults to None.
        noisy_voltage (list): voltage with PRBS injection. Defaults to None.
    """
    clean_fft_current = np.fft.fft(clean_current)
    noisy_fft_current = np.fft.fft(noisy_current)
    clean_fft_voltage = np.fft.fft(clean_voltage)
    noisy_fft_voltage = np.fft.fft(noisy_voltage)

    #for i in range(20):
    #    print(f"{clean_fft_current[i]}\t{noisy_fft_current[i]}\t{noisy_fft_voltage[i]}")

    gain_fft = 300 / (noisy_fft_current - clean_fft_current)
    phase_fft = noisy_fft_voltage / noisy_fft_current

    system_impedance = np.array(
        [
            np.absolute(gain_fft[i]) * np.exp(1j * np.angle(phase_fft[i]))
            for i in range(min([len(phase_fft), len(gain_fft)]))
        ]
    )

    return system_impedance


def nn_input_tensor(
    sys_impedance: list = None,
    sys_frequencies: list = None,
    L1: float = None,
    C1: float = None,
    R1: float = None,
) -> tensor :
    """Create the input_tensor from the computed impedance

    Args:
        sys_impedance (list): Impedance of the system. Defaults to None.
        sys_frequencies (list): Frequencies of the impedance. Defaults to None.
        L1 (float): Primary coil inductance in Henri. Defaults to None.
        R1 (float): Primary coil ESR in ohm. Defaults to None.
        C1 (float): Primary side series compensation capacitor in Farad. Defaults to None.
    Returns:
        tensor: input tensor for the neural network
    """

    wanted_frequencies = np.geomspace(50000, 144500, num=15, dtype=np.int64)

    output_tensor = []

    for i, frequency in enumerate(wanted_frequencies):
        index = np.argmin(
            np.array(
                [abs(sys_frequency - frequency) for sys_frequency in sys_frequencies]
            )
        )
        Z_estim = sys_impedance[index]
        # print(f"{index}\t{np.absolute(Z_estim)}\t{np.angle(Z_estim)}")
        Z1 = (
            1j * 2 * np.pi * frequency * L1 + R1 + 1 / (2 * np.pi * frequency * 1j * C1)
        )
        Z2 = Z_estim - Z1
        output_tensor.append(np.absolute(Z2))
        output_tensor.append(np.angle(Z2))

    return tensor(output_tensor, dtype=torch.float32)


def trim(list_in: list = None) -> list :
    return list_in[len(list_in) // 2 :]


def pretty_print(
    real_r: float = None,
    real_m: float = None,
    estim_r: float = None,
    estim_m: float = None,
) -> None :
    print("|" + "-" * 37 + "|")
    print("| Parameter | Real value | Estimation |")
    print("|" + "-" * 11 + "|" + "-" * 12 + "|" + "-" * 12 + "|")
    print(
        "|"
        + f"{'R_l (Ohm)':^11}"
        + "|"
        + f"{real_r:^12.3f}"
        + "|"
        + f"{estim_r:^12.3f}"
        + "|"
    )
    print("|" + "-" * 11 + "|" + "-" * 12 + "|" + "-" * 12 + "|")
    print(
        "|"
        + f"{'M (µH)':^11}"
        + "|"
        + f"{real_m:^12.2f}"
        + "|"
        + f"{estim_m:^12.2f}"
        + "|"
    )
    print("|" + "-" * 37 + "|" + "\n")


def main() -> None :
    """Main function of the script."""

    model_path = "neural_network_experiment\models_4_128\most_accurate_model.pt"
    data_path = "exp_data\data.csv"

    sampling_period = 1e-6

    # System known parameters

    L1 = 24e-6
    C1 = 1 / ((2 * np.pi * 85_000) ** 2 * L1)
    R1 = 0.075


    time, prbs, current, voltage = [], [], [], []

    # split the variables in 4 : time, prbs, current and voltage

    with open(data_path,newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter = ',')
        for row in csv_reader:
            time.append(float(row[0]))
            voltage.append(-int(row[1]))
            current.append(float(row[2]))
            prbs.append(float(row[3]))

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

    # for i in range(15):
    #     print(
    #         f"{trim(current)[i]:f}\t{trim(current_prbs)[i]:f}\t{trim(voltage_prbs)[i]:f}"
    #     )

    # print(
    #     f"len c : {len(trim(current))},\nlen v : {len(voltage)},\nlen nc : {len(current_prbs)},\nlen nv : {len(voltage_prbs)},\nlen prbs : {len(prbs)}"
    # )

    sys_impedance = extract_impedance(
        clean_current=trim(current),
        noisy_current=trim(current_prbs),
        clean_voltage=trim(voltage),
        noisy_voltage=trim(voltage_prbs),
    )

    sys_frequencies = np.fft.fftfreq(
        n=len(sys_impedance),
        d=sampling_period,
    )

    smooth_sys_impedance = fractional_decade_smoothing_impedance(
        impedances=sys_impedance,
        frequencies=sys_frequencies,
        fractional_factor=1.03,
    )

    input_tensor = nn_input_tensor(
        sys_impedance=smooth_sys_impedance,
        sys_frequencies=sys_frequencies,
        L1=L1,
        R1=R1,
        C1=C1,
    )

    # c_tensor = tensor(
    #     [
    #         80.005348,  0.188792,
    #         26.228678, -0.181752,
    #         101.336761,-0.638122,
    #         58.914665, -0.489149,
    #         42.030663, -0.286746,
    #         31.316027,  0.202536,
    #         42.960331,  0.271206,
    #         23.067545, -0.033072,
    #         47.115223, -0.440104,
    #         47.046467, -0.380599,
    #         38.042652, -0.545289,
    #         39.938625, -0.633409,
    #         34.425003, -0.966398,
    #         39.917068, -0.821997,
    #         35.868595, -1.259970,
    #     ]
    # )

    # print("simulation tensor")
    # for i in range(len(input_tensor) // 2):
    #     print(
    #         f"{input_tensor[2*i]:.3f} \t {input_tensor[2*i +1]:.3f}"  # \t {c_tensor[2*i]:.3f} \t {c_tensor[2*i +1]:.3f}"
    #     )

    # Load the neural network from the model object and the weights data

    neural_network = NN_model(input_size=30, output_size=2)
    checkpoint = torch.load(model_path)
    neural_network.load_state_dict(checkpoint)

    # Compute NN inferance

    with torch.inference_mode():
        output_tensor = neural_network(input_tensor)

    R = delinearise_R_l(output_tensor[0].item())
    M = delinearise_M(output_tensor[1].item())

# Potential test input tensor :

# 31.691679000854492, -2.73964786529541, 
# 29.640941619873047, -2.8143234252929688,
# 27.79124641418457, -2.909667730331421, 
# 26.136383056640625, -3.0375630855560303, 
# 24.68477439880371, 3.058429002761841, 
# 23.570587158203125, 2.7373595237731934, 
# 24.213733673095703, 2.00589919090271, 
# 35.29467010498047, 0.0, 
# 30.790233612060547, -1.724277138710022, 
# 27.17706298828125, -2.3004798889160156, 
# 26.042028427124023, -2.5823445320129395, 
# 25.85675048828125, -2.7643332481384277, 
# 26.204023361206055, -2.8997228145599365, 
# 26.93599510192871, -3.0080463886260986, 
# 27.986780166625977, -3.0980052947998047]   

# component values :

# M : 16.41e-6
# R_l : 1.18

    print("\nsimulation results :\n")
    pretty_print(
        real_r=1.25,
        # real_m=16.2, # for a 5mm gap
        real_m=5.23, # for a 20mm gap
        estim_r=R,
        estim_m=M * 1e6,
    )


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

    f_min = 25_000
    f_max = 250_000
    nb_samples = 20000
    bode_plot(
        systems=[wpt_system],
        f_min=f_min,
        f_max=f_max,
        nb_samples=nb_samples,
        f0=f0,
        samples=[
            #sys_fft,
            np.array(smooth_sys_impedance),
            #1 / np.array(fft_impulse),
        ],
        samples_frequency=[
            #freqs,
            sys_frequencies,
            #freqs_impulse,
        ],
        samples_names=[
            #"raw",
            "estimation",
            #"impulse response",
        ],
        title="Result with PRBS 10, experimental values",
    )

    # with torch.inference_mode():
    #     output_tensor = neural_network(c_tensor)

    # R = delinearise_R_l(output_tensor[0].item())
    # M = delinearise_M(output_tensor[1].item())

    # print("implementation results :\n")
    # pretty_print(
    #     real_r=0.7,
    #     real_m=11.2,
    #     estim_r=R,
    #     estim_m=M * 1e6,
    # )


if __name__ == "__main__":
    main()
