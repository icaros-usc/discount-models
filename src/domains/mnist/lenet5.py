"""Trains a LeNet-5 classifier for MNIST.

Adapted from "Training a Classifier," a tutorial in the PyTorch 60-minute blitz:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

Usage:
    # Saves classifier to mnist_classifier.pth
    python -m src.domains.mnist.lenet5

    # Evaluates an existing network.
    python train_mnist_classifier.py --eval-network FILE.pth
"""
import fire
import torch
import torchvision
from torch import nn


class LeNet5(nn.Module):
    """LeNet5 classifier."""

    MEAN_TRANSFORM = 0.1307
    STD_TRANSFORM = 0.3081

    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 6, (5, 5), stride=1,
                      padding=0),  # (1,28,28) -> (6,24,24)
            nn.MaxPool2d(2),  # (6,24,24) -> (6,12,12)
            nn.ReLU(),
            nn.Conv2d(6, 16, (5, 5), stride=1,
                      padding=0),  # (6,12,12) -> (16,8,8)
            nn.MaxPool2d(2),  # (16,8,8) -> (16,4,4)
            nn.ReLU(),
            nn.Flatten(),  # (16,4,4) -> (256,)
            nn.Linear(256, 120),  # (256,) -> (120,)
            nn.ReLU(),
            nn.Linear(120, 84),  # (120,) -> (84,)
            nn.ReLU(),
            nn.Linear(84, 10),  # (84,) -> (10,)
            nn.LogSoftmax(dim=1),  # (10,) log probabilities
        )

    def forward(self, x):
        return self.main((x - self.MEAN_TRANSFORM) / self.STD_TRANSFORM)


def fit(net, epochs, trainloader, device):
    """Trains net for the given number of epochs."""
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters())

    for epoch in range(epochs):
        print(f"=== Epoch {epoch + 1} ===")
        total_loss = 0.0

        # Iterate through batches in the shuffled training dataset.
        for batch_i, data in enumerate(trainloader):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (batch_i + 1) % 100 == 0:
                print(f"Batch {batch_i + 1:5d}: {total_loss}")
                total_loss = 0.0


def evaluate(net, loader, device):
    """Evaluates the network's accuracy on the images in the dataloader."""
    correct_per_num = [0 for _ in range(10)]
    total_per_num = [0 for _ in range(10)]

    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs.to("cpu"), 1)
            correct = (predicted == labels).squeeze()
            for label, c in zip(labels, correct):
                correct_per_num[label] += c.item()
                total_per_num[label] += 1

    for i in range(10):
        print(f"Class {i}: {correct_per_num[i] / total_per_num[i]:5.3f}"
              f" ({correct_per_num[i]} / {total_per_num[i]})")
    print(f"TOTAL  : {sum(correct_per_num) / sum(total_per_num):5.3f}"
          f" ({sum(correct_per_num)} / {sum(total_per_num)})")


def main(
    train_batch_size: int = 64,
    test_batch_size: int = 1000,
    eval_network: str = None,
):
    """Trains and saves the classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Transform each image by turning it into a tensor and then
    # normalizing the values.
    mnist_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((LeNet5.MEAN_TRANSFORM,),
                                         (LeNet5.STD_TRANSFORM,))
    ])
    trainloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=mnist_transforms),
                                              batch_size=train_batch_size,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        './data', train=False, transform=mnist_transforms),
                                             batch_size=test_batch_size,
                                             shuffle=False)

    lenet5 = LeNet5().to(device)

    if eval_network is not None:
        print("===== Loading existing network for evaluation =====")
        print("Filename:", eval_network)
        lenet5.load_state_dict(torch.load(eval_network, map_location=device))
    else:
        print("===== Fitting Network =====")
        fit(lenet5, 2, trainloader, device)

    print("===== Evaluation =====")
    print("=== Training Set Evaluation ===")
    evaluate(lenet5, trainloader, device)
    print("=== Test Set Evaluation ===")
    evaluate(lenet5, testloader, device)

    print("===== Saving Network to mnist_classifier.pth =====")
    torch.save(lenet5.state_dict(), "mnist_classifier.pth")


if __name__ == "__main__":
    fire.Fire(main)
