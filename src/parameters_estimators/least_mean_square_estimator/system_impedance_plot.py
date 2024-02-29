"""Simulation script that create WPTs, compute their impedance at different frequency and plot it on a Bode diagram."""

import numpy as np
import matplotlib.pyplot as plt
import utils.wpt_system_class as wpt
from utils.plot_function import bode_plot


def main() -> None :
    """Main funcion of the script."""
    # System creation, data from "Wireless Power Transfer Structure Design for Electric Vehicle in Charge While Driving"

    transmitter_s = wpt.transmitter(
        topology="S", L=86 * 1e-6, C_s=41 * 1e-9, R=0.35, height=0.75, width=0.1, N=9
    )
    transmitter_sp = wpt.transmitter(
        topology="SP",
        L=86 * 1e-6,
        C_s=33 * 1e-9,
        C_p=8.2 * 1e-9,
        R=0.35,
        height=0.75,
        width=0.1,
        N=9,
    )

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

    f0 = 85_000  # resonance frequency

    M = ((7 - 0.35) * (2 + 0.3)) ** 0.5 / (
        f0 * 2 * np.pi
    )  # ideal M = sqrt(Z_t * R_l) / w0 as showned in Vincenzo's Ph.D p8, max M = sqrt(L1 * L2)

    system_1 = wpt.total_system(
        transmitter=transmitter_s,
        reciever=reciever_s,
        M=M,
        M_from_geo=False,
        h=0.1,
        name="S-S",
    )
    system_2 = wpt.total_system(
        transmitter=transmitter_s,
        reciever=reciever_p,
        M=M,
        M_from_geo=False,
        h=0.1,
        name="S-P",
    )
    system_3 = wpt.total_system(
        transmitter=transmitter_sp,
        reciever=reciever_s,
        M=M,
        M_from_geo=False,
        h=0.1,
        name="SP-S",
    )
    system_4 = wpt.total_system(
        transmitter=transmitter_sp,
        reciever=reciever_p,
        M=M,
        M_from_geo=False,
        h=0.1,
        name="SP-P",
    )

    systems = [system_1, system_2]  # , system_3, system_4]

    for system in systems:
        print(f"System topology : {system.topology}, M = {system.M * 1e6:.1f} µH")
        print(f"Impedance at {f0/1000:.1f}kHz : {system.impedance(f0):.2f} Ohm\n")

    # System impedance plot

    # Frequency bounds and nb of samples

    f_min, f_max, nb_samples = 10_000, 1_000_000, 5000

    # Plot declaration
    bode_plot(systems=systems, f_min=f_min, f_max=f_max, nb_samples=nb_samples, f0=f0)


if __name__ == "__main__":
    main()
