""" Plot function to plot."""

import numpy as np
import matplotlib.pyplot as plt


def bode_plot(
    systems: list = None,
    f_min: float = 1e5,
    f_max: float = 1e6,
    nb_samples: int = None,
    f0: int = None,
    samples: list = None,
    samples_frequency: list = None,
):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle("Bode plot of the system impedance")

    # vertical line for f0
    if f0 != None:
        ax1.axvline(x=f0, color="black", linestyle=":")
        ax2.axvline(x=f0, color="black", linestyle=":")

    if systems != None:
        for system in systems:
            gain, phase, freqs = system.bode_plot_impedance(f_min, f_max, nb_samples)
            ax1.plot(freqs, gain, label=system.name)
            ax2.plot(freqs, phase, label=system.name)

    # Plot the mesured impedance

    if samples != None and samples_frequency != None:
        ax1.plot(
            samples_frequency,
            [20 * np.log10(np.absolute(impedance)) for impedance in samples],
            "rx",
            label="samples",
        )

        ax2.plot(
            samples_frequency,
            [np.angle(impedance) * 180 / np.pi for impedance in samples],
            "rx",
            label="samples",
        )

    # f0 label
    if f0 != None:
        y_bot1 = ax1.get_ylim()[0]
        ax1.text(f0, y_bot1 - 4, r"$f0$", color="black", ha="center", va="center")

        y_bot2 = ax2.get_ylim()[0]
        ax2.text(f0, y_bot2 * 1.1, r"$f0$", color="black", ha="center", va="center")

    # label and legend

    ax1.set_ylabel("Gain (dB)")
    ax1.set_xscale("log")
    ax1.legend()

    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylabel("Phase (deg)")
    ax2.set_xscale("log")
    ax2.legend()

    plt.show()
