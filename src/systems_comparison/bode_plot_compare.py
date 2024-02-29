"""A simple script that plot the bode plot for different topologies in order to compare their caracteristics. """

import numpy as np
import matplotlib.pyplot as plt
from plot_function import bode_plot
import wpt_system_class as wpt


def main() -> None:
    # Componants value

    f0 = 85000
    w0 = 2 * np.pi * f0
    L1 = 200 * 1e-6
    L2 = 200 * 1e-6
    M = 40 * 1e-6
    C1 = 1 / (L1 * w0**2)
    C2 = 1 / (L2 * w0**2)

    R1 = 0.3
    R2 = 0.3
    R_ls = 10.55
    R_lp = 1081
    R_llcc = 10.55

    C3 = L2 * C2 / (L1 + M**4 / (L1 * L2 * C2 * R_ls**2)) * 0.85
    C1_sp = 1.7726 * 1e-8
    print(f'{C3 = }')
    # Declaration of the systems

    transmitter_s = wpt.transmitter(topology="S", L=L1, R=R1, C_s=C1)
    transmitter_p = wpt.transmitter(topology="P", L=L1, R=R1, C_p=C1)
    transmitter_sp = wpt.transmitter(
        topology="SP", L=L1, R=R1, C_p=C3 * 0.85, C_s=C1_sp
    )
    transmitter_lcc = wpt.transmitter(
        topology="LCC", L=L1, R=R1, C_p=C1, C_s=C2, L_s=L2
    )

    reciever_s = wpt.reciever(topology="S", L=L2, R=R2, C_s=C2, R_l=R_ls)
    reciever_p = wpt.reciever(topology="P", L=L2, R=R2, C_p=C2, R_l=R_lp)
    reciever_lcc = wpt.reciever(
        topology="LCC", L=L2, R=R2, C_s=C2, C_p=C1, R_l=R_llcc, L_s=L1
    )

    system_s_s = wpt.total_system(
        transmitter=transmitter_s, reciever=reciever_s, M=M, name="S-S"
    )
    system_s_p = wpt.total_system(
        transmitter=transmitter_s, reciever=reciever_p, M=M, name="S-P"
    )
    system_s = wpt.total_system(
        transmitter=transmitter_s, reciever=reciever_s, M=0, name="S-"
    )

    system_p_p = wpt.total_system(
        transmitter=transmitter_p, reciever=reciever_s, M=M, name="P-S"
    )
    system_p_s = wpt.total_system(
        transmitter=transmitter_p, reciever=reciever_p, M=M, name="P-P"
    )
    system_p = wpt.total_system(
        transmitter=transmitter_p, reciever=reciever_s, M=0, name="P-"
    )

    system_sp_s = wpt.total_system(
        transmitter=transmitter_sp, reciever=reciever_s, M=M, name="SP-S"
    )
    system_sp_p = wpt.total_system(
        transmitter=transmitter_sp, reciever=reciever_p, M=M, name="SP-P"
    )
    system_sp = wpt.total_system(
        transmitter=transmitter_sp, reciever=reciever_s, M=0, name="SP-"
    )

    system_lcc_lcc = wpt.total_system(
        transmitter=transmitter_lcc, reciever=reciever_lcc, M=M, name="LCC-LCC"
    )
    system_lcc_s = wpt.total_system(
        transmitter=transmitter_lcc, reciever=reciever_s, M=M, name="LCC-S"
    )
    system_lcc = wpt.total_system(
        transmitter=transmitter_lcc, reciever=reciever_lcc, M=0, name="LCC-"
    )

    systems_s = [system_s_s, system_s_p, system_s]
    systems_p = [system_p_s, system_p_p, system_p]
    systems_sp = [system_sp_s, system_sp_p, system_sp]
    systems_lcc = [system_lcc_lcc, system_lcc_s, system_lcc]

    # systems_all = [
    #     system_s_s,
    #     system_s_p,
    #     system_s,
    #     system_p_s,
    #     system_p_p,
    #     system_p,
    #     system_sp_s,
    #     system_sp_p,
    #     system_sp,
    #
    # ]
    # Plot of the systems bode plots

    f_min, f_max, nb_samples = 3_000, 2500_000, 1000000
    
    #f_min, f_max, nb_samples = 70_000, 105_000, 10000
    
    bode_plot(
        systems=systems_s,
        f_min=f_min,
        f_max=f_max,
        nb_samples=nb_samples,
        f0=f0,
        title="Bode plot of systems impedance with Series topology on transmitter side",
    )

    bode_plot(
        systems=systems_p,
        f_min=f_min,
        f_max=f_max,
        nb_samples=nb_samples,
        f0=f0,
        title="Bode plot of systems impedance with Parallel topology on transmitter side",
    )

    bode_plot(
        systems=systems_sp,
        f_min=f_min,
        f_max=f_max,
        nb_samples=nb_samples,
        f0=f0,
        title="Bode plot of systems impedance with SP topology on transmitter side",
    )

    bode_plot(
        systems=systems_lcc,
        f_min=f_min,
        f_max=f_max,
        nb_samples=nb_samples,
        f0=f0,
        title="Bode plot of systems impedance with LCC topology on transmitter side",
    )

    # bode_plot(
    #     systems=systems_all,
    #     f_min=f_min,
    #     f_max=f_max,
    #     nb_samples=nb_samples,
    #     f0=f0,
    #     title="Bode plot of systems impedance with Series-Parallel topology on transmitter side"
    #     # samples=impedance_at_freq,
    #     # samples_frequency=tested_frequency,
    # )


if __name__ == "__main__":
    main()
