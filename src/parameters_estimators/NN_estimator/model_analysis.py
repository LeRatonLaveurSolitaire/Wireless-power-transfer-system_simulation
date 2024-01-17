import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.utils.data import dataloader
import time


from nn_model import NN_model
from dataset_class import CustomDataset


def delinearise_R_l(R_l: float = 0):
    return 10 ** ((0.15 * R_l) + 0.5)


def delinearise_M(M: float = 0):
    L1 = 236e-6
    L2 = 4.82e-6
    return 10 ** ((0.1 * M)) * (0.1 * (L1 * L2) ** 0.5)


def delinearise_f2(f2: float = 0):
    return 500 * f2 + 85000


def pretty_print(real: list = None, estim: list = None):
    real_r = delinearise_R_l(real[0])
    real_m = delinearise_M(real[1]) * 10**6
    # real_f = delinearise_f2(real[2])
    estim_r = delinearise_R_l(estim[0])
    estim_m = delinearise_M(estim[1]) * 10**6
    # estim_f = delinearise_f2(estim[2])

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
    # print("|" + "-" * 11 + "|" + "-" * 12 + "|" + "-" * 12 + "|")
    # print(
    #    "|"
    #    + f"{'f_2 (Hz)':^11}"
    #    + "|"
    #    + f"{real_f:^12.0f}"
    #    + "|"
    #    + f"{estim_f:^12.0f}"
    #    + "|"
    # )
    print("|" + "-" * 37 + "|" + "\n")


def main():
    dataset_path = "src/parameters_estimators/NN_estimator/dataset.pkl"
    model_path = "src/parameters_estimators/NN_estimator/models/most_accurate_model.pt"
    threshold = 1e-6 # compute inaccuacy only for M  > threshold

    print("Loading dataset...")
    dataset = CustomDataset()
    dataset.load(dataset_path)
    print("Dataset sucessfully loaded !\n")

    print("Loading model...")
    data = dataset[0]
    model = NN_model(input_size=len(data[0]), output_size=len(data[1]))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    print("Model sucessfully loaded !\n")

    print("Estimation exemple : \n")
    real = dataset[43][1].tolist()
    estim = model(dataset[43][0]).tolist()
    pretty_print(real=real, estim=estim)

    real = dataset[2689][1].tolist()
    estim = model(dataset[2689][0]).tolist()
    pretty_print(real=real, estim=estim)

    real = dataset[0][1].tolist()
    estim = model(dataset[0][0]).tolist()
    pretty_print(real=real, estim=estim)

    inaccuracy = []

    # for i, data in enumerate(dataset):
    #     if delinearise_M(data[1][1]) > threshold:
    #         output_tensor = data[1]
    #         estimation = model(data[0])

    #         inn = abs(output_tensor[0] - estimation[0]) / abs(output_tensor[0]) * 100
    #         inaccuracy.append(inn)
    #         inn = abs(output_tensor[1] - estimation[1]) / abs(output_tensor[1]) * 100
    #         inaccuracy.append(inn)

    # print(
    #     f"Average inaccuracy for data over the threshold : {sum(inaccuracy)/len(inaccuracy):4.2f}\n"
    # )


if __name__ == "__main__":
    main()
