""" Identifiaction of the wpt system from its equivalent impedence"""

import numpy as np
import matplotlib.pyplot as plt
from utils.plot_function import bode_plot
from utils.identification_functions import *  # mse_cost_function, update_param_s, cost_function_s
import utils.wpt_system_class as wpt


def main() -> None:
    """Main funcion of the script."""
    # Transmiter from wich we are certain of the parameters

    transmitter_s = wpt.transmitter(
        topology="S", L=86 * 1e-6, C_s=41 * 1e-9, R=0.35, height=0.75, width=0.1, N=9
    )

    # Recievers from wich we don't know the parameters

    reciever_s = wpt.reciever(
        topology="S",
        L=84 * 1e-6,
        R=0.3,
        C_s=42 * 1e-9,
        R_l=2,
        height=0.2,
        width=0.25,
        N=12,
    )

    reciever_p = wpt.reciever(
        topology="P",
        L=84 * 1e-6,
        R=0.3,
        C_p=42 * 1e-9,
        R_l=2,
        height=0.2,
        width=0.25,
        N=12,
    )

    # Real system from wich we observe the behavior
    f0 = 85_000

    real_system = wpt.total_system(
        transmitter=transmitter_s,
        reciever=reciever_p,
        name="Real sys (S-P)",
        M=7 * 1e-6,
    )

    # Estimated recievers and systems

    estimated_reciever_s = wpt.reciever(topology="S", L=0, R=0.1, C_s=1e-9, R_l=0)
    estimated_reciever_p = wpt.reciever(topology="P", L=0, R=0.1, C_p=1e-9, R_l=0)

    estimated_system_s = wpt.total_system(
        transmitter=transmitter_s,
        reciever=estimated_reciever_s,
        M=0,
        name="S-S estimation",
    )
    estimated_system_p = wpt.total_system(
        transmitter=transmitter_s,
        reciever=estimated_reciever_p,
        M=0,
        name="S-P estimation",
    )
    # Creation of the parameter vector, the value of C2 and R2 are fixed and won't be oprimised

    param_vect = [0.1, 1e-12, 1e-6, 1 * 1e-9, 0.3]  # R_l, M²,  L2, C2, R2

    # Reading of the impedance at different value with gaussian noise

    tested_frequency = [f for f in range(75_000, 96_000, 1_000)]
    impedance_at_freq = []
    for f in tested_frequency:
        mu = 0
        sigma = 0.3
        noise = np.random.normal(mu, sigma) + 1j * np.random.normal(mu, sigma)

        impedance_at_freq.append(real_system.impedance(f) + noise)

    # Optimisation of the parameters
    cost_p = mse_cost_function(
        impedance_at_freq, tested_frequency, param_vect, estimated_system_p
    )

    # cost_p = cost_function_p(
    #     impedance_at_freq, tested_frequency, param_vect, estimated_system_p
    # )

    costs = []
    i = 0
    while i < 30:  # cost_s > 1:
        i += 1
        cost_p = mse_cost_function(
            impedance_at_freq, tested_frequency, param_vect, estimated_system_p
        )

        update_param_p(
            impedance_at_freq, tested_frequency, param_vect, estimated_system_p
        )
        print(
            f"Iter {i}, cost : {cost_p:6.3G}, param_vect : R_l = {param_vect[0]:.2G}, M = {np.real(param_vect[1]**0.5) * 1e6 :.2G} µH, L2 = {param_vect[2] * 1e6 :.2G} µH, C2 = {param_vect[3] * 1e9:.2G} nF"  # , R_l : {param_vect[0]:6.3f}, M : {param_vect[1]**0.5 * 1e6 :6.0f} µH, L2 : {param_vect[2] * 1e6 :.2f} µH"
        )

    print(
        f"Estimated parameters : R_l : {param_vect[0]:.2G}, M : {np.real(param_vect[1]**0.5)* 1e6 :.2G} µH, L2 : {param_vect[2] * 1e6 :.2G} µH, C2 = {param_vect[3] * 1e9:.2G} nF"
    )

    print(f"Estimated parameters : R_l : 2, M : 7 µH, L2 : 84 µH, C2 = 42 nF")
    print(
        f"cost for estimated system : {mse_cost_function(impedance_at_freq, tested_frequency, param_vect, estimated_system_p):.3f}"
    )
    print(
        f"cost for real system : {mse_cost_function(impedance_at_freq, tested_frequency, param_vect, real_system):.3f}"
    )
    # ----------------------------Plot part---------------------------#

    systems = [real_system, estimated_system_p]

    f_min, f_max, nb_samples = 70_000, 105_000, 5000

    # Plot the result
    bode_plot(
        systems=systems,
        f_min=f_min,
        f_max=f_max,
        nb_samples=nb_samples,
        f0=f0,
        samples=impedance_at_freq,
        samples_frequency=tested_frequency,
    )


if __name__ == "__main__":
    main()
