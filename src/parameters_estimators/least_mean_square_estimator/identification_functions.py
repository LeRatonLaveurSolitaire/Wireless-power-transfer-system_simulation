import numpy as np
import utils.wpt_system_class as wpt


def mse_cost_function(
    samples: list = None,
    frequencys: list = None,
    param_vect: tuple = None,  # R_l, M²,  L2, C2, R2
    estimated_system: wpt.total_system = None,
) -> float:
    """MSE loss cost function for a wpt system.

    Compare the samples with the impedance of the system

    Args:
        samples (list): list of the mesured impedances. Defaults to None.
        frequencys (list): list of the frequency at wich the samples have been mesured. Defaults to None.
        estimated_system (wpt.total_system): totam_system object of the etimated system. The trasmitter must
            have the value of the transmitter from the real system. Defaults to None.
    Returns:
            float: Cost computed by the cost funcion.
    """

    cost = 0

    # for more info, look at "Multiple Parameters Estimation Based on Transmitter Side Information in Wireless Power Transfer System"

    for i, sample in enumerate(samples):
        re_part = np.real(sample - estimated_system.impedance(frequencys[i]))
        im_part = np.imag(sample - estimated_system.impedance(frequencys[i]))

        cost += (
            1 / len(samples) * (re_part**2 + im_part**2)
        )  # np.absolute((wi * M) ** 2 + D * (R_l + 1j * wi * L + B)) ** 2

    return cost


def cost_function_s(
    samples: list = None,
    frequencys: list = None,
    param_vect: tuple = None,  # R_l, M²,  L2, C2, R2
    estimated_system: wpt.total_system = None,
) -> float:
    """Cost function for a S-S wpt system.

    Args:
        samples (list): list of the mesured impedances. Defaults to None.
        frequencys (list): list of the frequency at wich the samples have been mesured. Defaults to None.
        estimated_system (wpt.total_system): totam_system object of the etimated system. The trasmitter must
            have the value of the transmitter from the real system. Defaults to None.
    Returns:
            float: Cost computed by the cost funcion.
    """

    cost = 0

    if estimated_system.transmitter.topology != "S":
        raise ValueError(
            f"invalid topology of transmitter : {estimated_system.transmitter.topology}, must be 'S'"
        )
    if estimated_system.reciever.topology != "S":
        raise ValueError(
            f"invalid topology of reciever : {estimated_system.reciever.topology}, must be 'S'"
        )

    # for more info, look at "Multiple Parameters Estimation Based on Transmitter Side Information in Wireless Power Transfer System"

    for i, sample in enumerate(samples):
        wi = 2 * np.pi * frequencys[i]

        R_l = estimated_system.reciever.R_l
        L = estimated_system.reciever.L
        R = estimated_system.reciever.R
        C = estimated_system.reciever.C_s
        M = estimated_system.M

        A = estimated_system.transmitter.R + 1j * (
            wi * estimated_system.transmitter.L
            - 1 / (wi * estimated_system.transmitter.C_s)
        )
        B = R - 1j * 1 / (C)
        D = sample - A

        re_part = (
            np.real(D) * (R_l + R) - np.imag(D) * (L * wi - 1 / (C * wi)) + wi**2
        )
        im_part = np.imag(D) * (R_l + R) + np.real(D) * (L * wi - 1 / (C * wi))

        # ( 1 / len(samples) * (re_part**2 + im_part**2))
        cost += (
            1
            / len(samples)
            * np.absolute((wi * M) ** 2 + D * (R_l + 1j * wi * L + B)) ** 2
        )

    return cost


def update_param_s(
    samples: list = None,
    frequencys: list = None,
    param_vect: tuple = None,  # R_l, M²,  L2, C2, R2
    estimated_system: wpt.total_system = None,
) -> None:
    """Compute the partial derivative of the cost function for estimating the gradient of R_l, M² and L2 and update the param_vect based.


    Args:
        samples (list): list of the mesured impedances. Defaults to None.
        frequencys (list): list of the frequency at wich the samples have been mesured. Defaults to None.
        param_vect (tuple): parameter of the estimated system with the following shape : (R_l, M²,  L2, C2, R2).
            Defaults to None.
        estimated_system (wpt.total_system): totam_system object of the etimated system. The trasmitter must
            have the value of the transmitter from the real system. Defaults to None.

    """

    partial_derivative = [0, 0, 0]  # R_l, M², L2
    k = 10
    update_rate = [k, k, k]

    for i, sample in enumerate(samples):
        wi = 2 * np.pi * frequencys[i]
        transmitter_impedance = (
            estimated_system.transmitter.R
            + 1j * wi * estimated_system.transmitter.L
            - 1j * 1 / (wi * estimated_system.transmitter.C_s)
        )
        B = estimated_system.reciever.R - 1j * 1 / (estimated_system.reciever.C_s)
        D = sample - transmitter_impedance
        R_l = param_vect[2]
        L = estimated_system.reciever.L
        M2 = param_vect[1]
        C = estimated_system.reciever.C_s

        # partial_derivative[0] += (
        #     1
        #     / (len(samples))
        #     * (
        #         (wi**2 * M2 + np.real(D) * R_l - np.imag(D) * wi * L + np.real(D * B))
        #         * np.real(D)
        #         + (np.imag(D) * R_l + np.real(D) * wi * L + np.imag(D * B)) * np.imag(D)
        #     )
        # )
        # partial_derivative[1] += (
        #     1
        #     / (len(samples))
        #     * (wi**2 * M2 + np.real(D) * R_l - np.imag(D) * wi * L + np.real(D * B))
        #     * wi**2
        # )
        # partial_derivative[2] += (
        #     1
        #     / (len(samples))
        #     * (
        #         (wi**2 * M2 + np.real(D) * R_l - np.imag(D) * wi * L + np.real(D * B))
        #         * (-np.imag(D) * wi)
        #         + (np.imag(D) * R_l + np.real(D) * wi * L + np.imag(D * B))
        #         * np.real(D)
        #         * wi
        #     )
        # )

        partial_derivative[0] += np.real(
            transmitter_impedance
            + wi**2 * M2 / (R_l + 1j * wi * L + 1 / (1j * wi * C))
            - sample
        ) * np.real(C * wi**3 / (C * wi * (R_l + i * L * wi) - 1j)) + np.imag(
            transmitter_impedance
            + wi**2 * M2 / (R_l + 1j * wi * L + 1 / (1j * wi * C))
            - sample
        ) * np.imag(
            C * wi**3 / (C * wi * (R_l + i * L * wi) - 1j)
        )

        partial_derivative[1] += np.real(
            transmitter_impedance
            + wi**2 * M2 / (R_l + 1j * wi * L + 1 / (1j * wi * C))
            - sample
        ) * np.real(
            C**2 * wi**4 * M2 / (-1 + C * wi**2 * (L * wi - 1j * R_l)) ** 2
        ) + np.imag(
            transmitter_impedance
            + wi**2 * M2 / (R_l + 1j * wi * L + 1 / (1j * wi * C))
            - sample
        ) * np.imag(
            C**2 * wi**4 * M2 / (-1 + C * wi**2 * (L * wi - 1j * R_l)) ** 2
        )

        partial_derivative[2] += np.real(
            transmitter_impedance
            + wi**2 * M2 / (R_l + 1j * wi * L + 1 / (1j * wi * C))
            - sample
        ) * np.real(
            -1j * C**2 * M2 * wi**5 / (C * wi * (R_l + 1j * L * wi) - 1j) ** 2
        ) + np.imag(
            transmitter_impedance
            + wi**2 * M2 / (R_l + 1j * wi * L + 1 / (1j * wi * C))
            - sample
        ) * np.imag(
            -1j * C**2 * M2 * wi**5 / (C * wi * (R_l + 1j * L * wi) - 1j) ** 2
        )

    update_values = [0, 0, 0]
    for i in range(3):
        param_vect[i] -= np.real(update_rate[i] * partial_derivative[i])

    estimated_system.M = np.real(param_vect[1] ** (0.5))
    estimated_system.reciever.R_l = param_vect[0]
    estimated_system.reciever.L = param_vect[2]
    estimated_system.reciever.C_s = param_vect[3]
    estimated_system.reciever.R = param_vect[4]


def cost_function_p(
    samples: list = None,
    frequencys: list = None,
    param_vect: tuple = None,  # R_l, M²,  L2, C2, R2
    estimated_system: wpt.total_system = None,
) -> float:
    """Cost function for a S-P wpt system.

    Args:
        samples (list): list of the mesured impedances. Defaults to None.
        frequencys (list): list of the frequency at wich the samples have been mesured. Defaults to None.
        estimated_system (wpt.total_system): totam_system object of the etimated system. The trasmitter must
            have the value of the transmitter from the real system. Defaults to None.
    Returns:
            float: Cost computed by the cost funcion.
    """

    cost = 0

    if estimated_system.transmitter.topology != "S":
        raise ValueError(
            f"invalid topology of transmitter : {estimated_system.transmitter.topology}, must be 'S'"
        )
    if estimated_system.reciever.topology != "P":
        raise ValueError(
            f"invalid topology of reciever : {estimated_system.reciever.topology}, must be 'P'"
        )

    # for more info, look at "Multiple Parameters Estimation Based on Transmitter Side Information in Wireless Power Transfer System"

    for i, sample in enumerate(samples):
        wi = 2 * np.pi * frequencys[i]

        R_l = estimated_system.reciever.R_l
        L = estimated_system.reciever.L
        R = estimated_system.reciever.R
        C = estimated_system.reciever.C_p
        M = estimated_system.M

        A = estimated_system.transmitter.R + 1j * (
            wi * estimated_system.transmitter.L
            - 1 / (wi * estimated_system.transmitter.C_s)
        )
        B = R - 1j * 1 / (C)
        D = sample - A

        re_part = wi**2 * M**2 + np.real(D) * R_l * (1 - wi**2 * L * C)
        im_part = wi**3 * M**2 * R_l * C + np.imag(D) * R_l * (1 - wi**2 * L * C)

        cost += (
            1 / len(samples) * (re_part**2 + im_part**2)
        )  # np.absolute((wi * M) ** 2 + D * (R_l + 1j * wi * L + B)) ** 2

    return cost


def update_param_p(
    samples: list = None,
    frequencys: list = None,
    param_vect: tuple = None,  # R_l, M²,  L2, C2, R2
    estimated_system: wpt.total_system = None,
) -> None:
    """Compute the partial derivative of the cost function for estimating the gradient of R_l, M² and L2 and update the param_vect based.


    Args:
        samples (list): list of the mesured impedances. Defaults to None.
        frequencys (list): list of the frequency at wich the samples have been mesured. Defaults to None.
        param_vect (tuple): parameter of the estimated system with the following shape : (R_l, M²,  L2, C2, R2).
            Defaults to None.
        estimated_system (wpt.total_system): totam_system object of the etimated system. The trasmitter must
            have the value of the transmitter from the real system. Defaults to None.

    """

    partial_derivative = [0, 0, 0, 0]  # R_l, M², L2, C2
    k = 1e-23
    update_rate = [k * 1e21, k, k * 1e19, k * 1e12]

    for i, sample in enumerate(samples):
        wi = 2 * np.pi * frequencys[i]
        transmitter_impedance = (
            estimated_system.transmitter.R
            + 1j * wi * estimated_system.transmitter.L
            - 1j * 1 / (wi * estimated_system.transmitter.C_s)
        )

        D = sample - transmitter_impedance
        R_l = param_vect[2]
        L = estimated_system.reciever.L
        M2 = param_vect[1]
        C = estimated_system.reciever.C_p

        real_part = wi**2 * M2 + np.real(D) * R_l * (1 - wi**2 * L * C)
        imag_part = wi**3 * M2 * R_l * C + np.imag(D) * R_l * (1 - wi**2 * L * C)

        partial_derivative[0] += (
            1
            / (len(samples))
            * (
                real_part * np.real(D) * (1 - wi**2 * L * C)
                + imag_part * (wi**3 * M2 * C + np.real(D) * (1 - wi**2 * L * C))
            )
        )
        partial_derivative[1] += (
            1 / (len(samples)) * (real_part * wi**2 + imag_part * wi**2 * R_l * C)
        )
        partial_derivative[2] += (
            1
            / (len(samples))
            * (-R_l * C * wi**2 * (np.real(D) * real_part + np.imag(D) * real_part))
        )

        partial_derivative[3] += (
            1
            / (len(samples))
            * (
                real_part * (-L * wi**2 * R_l * np.real(D))
                + (imag_part * (wi**3 * M2 * R_l - np.imag(D) * R_l * wi**2 * L))
            )
        )
    update = []
    for i in range(len(partial_derivative)):
        param_vect[i] -= np.real(update_rate[i] * partial_derivative[i])
        update.append(np.real(-update_rate[i] * partial_derivative[i]))

    estimated_system.M = np.real(param_vect[1] ** (0.5)) * 1e-6
    estimated_system.reciever.R_l = param_vect[0]
    estimated_system.reciever.L = param_vect[2] * 1e-6
    estimated_system.reciever.C_p = param_vect[3]
    estimated_system.reciever.R = param_vect[4]
