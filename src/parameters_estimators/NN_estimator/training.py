"""Training script for the neural network."""

import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.utils.data import dataloader
from torch.utils.tensorboard import SummaryWriter
import time


from nn_model import NN_model
from dataset_class import CustomDataset


def delinearise_R_l(R_l: float = 0):
    return 10 ** ((0.2 * R_l) + 1)


def delinearise_M(M: float = 0):
    return 10 ** ((0.15 * M) - 5.5)


def delinearise_f2(f2: float = 0):
    return 500 * f2 + 85000


def pretty_print(real: list = None, estim: list = None):
    real_r = delinearise_R_l(real[0])
    real_m = delinearise_M(real[1]) * 10**6
    real_f = delinearise_f2(real[2])
    estim_r = delinearise_R_l(estim[0])
    estim_m = delinearise_M(estim[1]) * 10**6
    estim_f = delinearise_f2(estim[2])

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
    print("|" + "-" * 11 + "|" + "-" * 12 + "|" + "-" * 12 + "|")
    print(
        "|"
        + f"{'f_2 (Hz)':^11}"
        + "|"
        + f"{real_f:^12.0f}"
        + "|"
        + f"{estim_f:^12.0f}"
        + "|"
    )
    print("|" + "-" * 37 + "|" + "\n")


def main():
    """Main function of the script."""

    # Load dataset

    dataet_path = "src/parameters_estimators/NN_estimator/dataset.pkl"

    print("Loading dataset...")
    dataset = CustomDataset()
    dataset.load(dataet_path)
    print("Dataset sucessfully loaded !")

    # Config tensorboard

    writer = SummaryWriter()

    # Improved performances if a GPU is available

    device = torch.device("cuda" if cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    # Hyperparameters

    num_epochs = 150
    batch_size = 512
    learning_rate = 0.001

    # Split dataset into training/test datasets

    train_percentage = 0.8

    dataset_size = len(dataset)
    train_dataset_size = int(train_percentage * dataset_size)
    test_dataset_size = dataset_size - train_dataset_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_dataset_size, test_dataset_size]
    )

    train_dataset_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataset_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    # Define model

    input_tensor, output_tensor = dataset[0]

    input_size = len(input_tensor)
    output_size = len(output_tensor)

    model = NN_model(input_size=input_size, output_size=output_size).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # load a model in case of error during training a

    # model_path = ""
    # checkpoint = torch.load(file_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    # Training

    print("Start training...")

    start_time = time.time()
    min_inaccuracy = float("inf")

    for epoch in range(num_epochs):
        for i, (input_tensors, output_tensors) in enumerate(train_dataset_loader):
            # Transfer tensors to device (GPU or CPU)

            input_tensors = input_tensors.to(device)
            output_tensors = output_tensors.to(device)

            # Forward pass

            estimations = model(input_tensors)
            loss = criterion(estimations, output_tensors)
            writer.add_scalar("Loss/train", loss, epoch)

            # Backward pass

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Test model accuracy on the test dataset

        with torch.no_grad():
            total_inaccuracy_per_cent = 0
            nbr_samples = 0

            for input_tensors, output_tensors in test_dataset_loader:
                input_tensors = input_tensors.to(device)
                output_tensors = output_tensors.to(device)

                outputs = model(input_tensors)

                for i in range(len(output_tensors)):
                    for j in range(len(output_tensors[i])):
                        total_inaccuracy_per_cent += (
                            abs((output_tensors[i][j] - outputs[i][j]))
                            / (abs(output_tensors[i][j]))
                            * 100
                        ).item()
                nbr_samples += len(input_tensors)

        avg_inaccuracy_per_cent = total_inaccuracy_per_cent / nbr_samples
        print(f"Inaccuracy :{avg_inaccuracy_per_cent:.3}%")
        writer.add_scalar("Accuracy/train", avg_inaccuracy_per_cent, epoch)

        # Save the model if it is more precise than the previous one

        if avg_inaccuracy_per_cent < min_inaccuracy:
            min_inaccuracy = avg_inaccuracy_per_cent
            torch.save(
                model.state_dict(),
                "src/parameters_estimators/NN_estimator/models/most_accurate_model.pt",
            )

        if epoch % 10 == 0:
            # Print time since beggining of training, number of epoch and loss

            sec = int(int(time.time() - start_time) % 60)
            minutes = int(int(time.time() - start_time) / 60) % 60
            h = int(int(int(time.time() - start_time) / 60) / 60)
            print(
                f"Time {h:3d}:{minutes:02d}:{sec:02d} - Epoch {epoch:03d} / {num_epochs} - Loss = {loss.item():07.1f} "
            )
            print("Estimation exemple : ")
            with torch.no_grad():
                real = dataset[42][1].tolist()
                estim = model(dataset[42][0]).tolist()
                pretty_print(real=real, estim=estim)

                real = dataset[43][1].tolist()
                estim = model(dataset[43][0]).tolist()
                pretty_print(real=real, estim=estim)

                real = dataset[44][1].tolist()
                estim = model(dataset[44][0]).tolist()
                pretty_print(real=real, estim=estim)
            # Save trained model during training

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                f"src/parameters_estimators/NN_estimator/models/model_at_epoch_{epoch}.pt",
            )

    writer.flush()
    writer.close()

    # Test final accuracy on the test dataset

    with torch.no_grad():
        total_inaccuracy_per_cent = 0
        nbr_samples = 0

        for input_tensors, output_tensors in test_dataset_loader:
            input_tensors = input_tensors.to(device)
            output_tensors = output_tensors.to(device)

            outputs = model(input_tensors)

            for i in range(len(output_tensors)):
                for j in range(len(output_tensors[i])):
                    total_inaccuracy_per_cent += (
                        abs((output_tensors[i][j] - outputs[i][j]))
                        / (len(output_tensors[i]) * output_tensors[i][j])
                        * 100
                    ).item()
            nbr_samples += len(input_tensors)

    avg_inaccuracy_per_cent = total_inaccuracy_per_cent / nbr_samples
    print(f"Final Inaccuracy :{avg_inaccuracy_per_cent:.5}%")

    # Save final model

    model_path = "src/parameters_estimators/NN_estimator/models/final_model.pt"
    torch.save(model.state_dict(), model_path)

    # Print some estimation

    real = dataset[42][1].tolist()
    estim = model(dataset[42][0]).tolist()
    pretty_print(real=real, estim=estim)

    real = dataset[43][1].tolist()
    estim = model(dataset[43][0]).tolist()
    pretty_print(real=real, estim=estim)

    real = dataset[44][1].tolist()
    estim = model(dataset[44][0]).tolist()
    pretty_print(real=real, estim=estim)


if __name__ == "__main__":
    main()
