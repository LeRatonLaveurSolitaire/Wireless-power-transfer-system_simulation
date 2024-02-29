import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.utils.data import dataloader
import time

import nn_model_4_128
import nn_model_5_128
import nn_model_5_256

import sys
import os

path_to_dataset_class = os.path.join(sys.path[0], "..", "src", "parameters_estimators", "NN_estimator")
sys.path.insert(0, path_to_dataset_class)

from dataset_class import CustomDataset


def delinearise_R_l(R_l: float = 0) -> float :
    return 10 ** (R_l * 0.1)


def delinearise_M(M: float = 0) -> float :
    L1 = 24e-6
    L2 = 24e-6
    return 10 ** ((0.1 * M)) * (0.1 * (L1 * L2) ** 0.5)


def delinearise_f2(f2: float = 0) -> float :
    return 500 * f2 + 85000


def pretty_print(real: list = None, estim1: list = None, estim2: list = None, estim3: list = None) -> None :
    real_r = delinearise_R_l(real[0])
    real_m = delinearise_M(real[1]) * 10**6
    # real_f = delinearise_f2(real[2])
    estim1_r = delinearise_R_l(estim1[0])
    estim1_m = delinearise_M(estim1[1]) * 10**6
    # estim_f = delinearise_f2(estim[2])
    estim2_r = delinearise_R_l(estim2[0])
    estim2_m = delinearise_M(estim2[1]) * 10**6

    estim3_r = delinearise_R_l(estim3[0])
    estim3_m = delinearise_M(estim3[1]) * 10**6


    print("|" + "-" * (12*4 + 15) + "|")
    print("| Parameter | Real value |  Model  1  |  Model  2  |  Model  3  |")
    print("|" + "-" * 11 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|"+ "-" * 12 + "|")
    print(
        "|"
        + f"{'R_l (Ohm)':^11}"
        + "|"
        + f"{real_r:^12.3f}"
        + "|"
        + f"{estim1_r:^12.3f}"
        + "|"
        + f"{estim2_r:^12.3f}"
        + "|"
        + f"{estim3_r:^12.3f}"
        + "|"
    )
    print("|" + "-" * 11 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|"+ "-" * 12 + "|")
    print(
        "|"
        + f"{'M (µH)':^11}"
        + "|"
        + f"{real_m:^12.2f}"
        + "|"
        + f"{estim1_m:^12.2f}"
        + "|"
        + f"{estim2_m:^12.2f}"
        + "|"
        + f"{estim3_m:^12.2f}"
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
    print("|" + "-" * (12*4 + 15) + "|" + "\n")


def main() -> None:
    dataset_path = "neural_network_experiment\dataset_experimental.pkl"
    model1_path = "neural_network_experiment\models-exp-4_128\most_accurate_model.pt"
    model2_path = "neural_network_experiment\models-exp-5_128\most_accurate_model.pt"
    model3_path = "neural_network_experiment\models-exp-5_256\most_accurate_model.pt"
    threshold = 0  # compute inaccuacy only for M  > threshold

    print("Loading dataset...")
    dataset = CustomDataset()
    dataset.load(dataset_path)
    print("Dataset sucessfully loaded !\n")

    print("Loading model 1...")
    data = dataset[0]
    model1 = nn_model_4_128.NN_model(input_size=len(data[0]), output_size=len(data[1]))
    checkpoint = torch.load(model1_path)
    model1.load_state_dict(checkpoint)
    print("Model 1 sucessfully loaded !\n")
    
    print("Loading model 2...")
    data = dataset[0]
    model2 = nn_model_5_128.NN_model(input_size=len(data[0]), output_size=len(data[1]))
    checkpoint = torch.load(model2_path)
    model2.load_state_dict(checkpoint)
    print("Model 2 sucessfully loaded !\n")

    print("Loading model 3...")
    data = dataset[0]
    model3 = nn_model_5_256.NN_model(input_size=len(data[0]), output_size=len(data[1]))
    checkpoint = torch.load(model3_path)
    model3.load_state_dict(checkpoint)
    print("Model 3 sucessfully loaded !\n")

    print("Estimation exemple : \n")

    with torch.inference_mode():
        real = dataset[43][1].tolist()
        estim1 = model1(dataset[43][0]).tolist()
        estim2 = model2(dataset[43][0]).tolist()
        estim3 = model3(dataset[43][0]).tolist()

    pretty_print(real=real, estim1=estim1, estim2 = estim2, estim3 = estim3)

    with torch.inference_mode():
        real = dataset[2689][1].tolist()
        estim1 = model1(dataset[2689][0]).tolist()
        estim2 = model2(dataset[2689][0]).tolist()
        estim3 = model3(dataset[2689][0]).tolist()

    pretty_print(real=real, estim1=estim1, estim2 = estim2, estim3 = estim3)

    with torch.inference_mode():
        real = dataset[0][1].tolist()
        estim1 = model1(dataset[0][0]).tolist()
        estim2 = model2(dataset[0][0]).tolist()
        estim3 = model3(dataset[0][0]).tolist()

    pretty_print(real=real, estim1=estim1, estim2 = estim2, estim3 = estim3)

    inaccuracy_1 = []
    inaccuracy_2 = []
    inaccuracy_3 = []

    for i, data in enumerate(dataset):
        if delinearise_M(data[1][1]) > threshold:
            output_tensor = data[1]
            with torch.inference_mode():
                estimation_1 = model1(data[0])
                estimation_2 = model2(data[0])
                estimation_3 = model3(data[0])
            if (delinearise_M(data[1][1]) > 16e-6) and (delinearise_M(data[1][1])  < 16.5e-6) and delinearise_R_l(data[1][0]) > 1 and delinearise_R_l(data[1][0]) < 1.5:
                print(f"Potential test input tensor :")
                print(data[0].tolist())
                print(f"component values : \nM : {delinearise_M(data[1][1])} \nR_l : {delinearise_R_l(data[1][0])}")

            inn1 = abs((output_tensor[0] - estimation_1[0]) / output_tensor[0])
            inaccuracy_1.append(inn1)
            inn1 = abs((output_tensor[1] - estimation_1[1]) / output_tensor[1])
            inaccuracy_1.append(inn1)

            inn2 = abs((output_tensor[0] - estimation_2[0]) / output_tensor[0])
            inaccuracy_2.append(inn2)
            inn1 = abs((output_tensor[1] - estimation_2[1]) / output_tensor[1])
            inaccuracy_2.append(inn2)

            inn3 = abs((output_tensor[0] - estimation_3[0]) / output_tensor[0])
            inaccuracy_3.append(inn3)
            inn3 = abs((output_tensor[1] - estimation_3[1]) / output_tensor[1])
            inaccuracy_3.append(inn3)

    print(
        f"Inaccuracy for model 1 : {sum(inaccuracy_1)/len(inaccuracy_1):4.2%}\n"
    )
    print(
        f"Inaccuracy for model 2 : {sum(inaccuracy_2)/len(inaccuracy_2):4.2%}\n"
    )
    print(
        f"Inaccuracy for model 3 : {sum(inaccuracy_3)/len(inaccuracy_3):4.2%}\n"
    )

# Estimation exemple :

# /---------------------------------------------------------------\
# | Parameter | Real value |   Model 1  |   Model 2  |   Model 3  |
# |-----------|------------|------------|------------|------------|
# | R_l (Ohm) |   0.214    |   0.448    |   0.419    |   0.371    |
# |-----------|------------|------------|------------|------------|
# |  M (µH)   |    0.27    |    0.35    |    0.35    |    0.34    |
# \---------------------------------------------------------------/

# /---------------------------------------------------------------\
# | Parameter | Real value |   Model 1  |   Model 2  |   Model 3  |
# |-----------|------------|------------|------------|------------|
# | R_l (Ohm) |   5.353    |   5.350    |   5.217    |   5.275    |
# |-----------|------------|------------|------------|------------|
# |  M (µH)   |    7.10    |    7.51    |    7.15    |    7.01    |
# \---------------------------------------------------------------/

# /---------------------------------------------------------------\
# | Parameter | Real value |   Model 1  |   Model 2  |   Model 3  |
# |-----------|------------|------------|------------|------------|
# | R_l (Ohm) |   4.251    |   3.893    |   3.888    |   4.112    |
# |-----------|------------|------------|------------|------------|
# |  M (µH)   |   11.28    |   10.75    |   10.98    |   11.10    |
# \---------------------------------------------------------------/

# Inaccuracy for model 1 : 36.63 %
# Inaccuracy for model 2 : 40.19 %
# Inaccuracy for model 3 : 20.81 %

if __name__ == "__main__":
    main()
