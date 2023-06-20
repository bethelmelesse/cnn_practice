from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose


# Download traning and test data from open datasets --- MNIST
def mnist():
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor()
    )

    num_classes = 10
    return training_data, test_data, num_classes