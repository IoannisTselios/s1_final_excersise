import os
import pytest
from tests import _PATH_DATA
from data import mnist

# Assuming MNIST class takes a data folder path as an argument
mnist_data_path = os.path.join(_PATH_DATA, "data/")  # Adjust the folder name as per your structure

def test_data_loading():
    # Initialize the MNIST dataset with the data path
    mnist_train, mnist_test = mnist()

    # Test for correct number of samples
    assert len(mnist_train) == 25000  # Adjust the value based on your specific case
    assert len(mnist_test) == 5000

    # Test for correct data shape
    for data, label in mnist_train:
        assert data.shape == (1, 28, 28) or data.shape == (784,)  # Adjust based on your data format

    # Test for representation of all labels
    unique_labels = set(label.item() for _, label in mnist_train)
    print(unique_labels)
    assert len(unique_labels) == 10  # Assuming MNIST has 10 classes

# Add more tests if needed
