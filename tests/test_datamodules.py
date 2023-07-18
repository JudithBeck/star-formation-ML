from pathlib import Path

import pytest
import torch

from src.data.mnist_datamodule import MNISTDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size):
    data_dir = "data/"

    dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


#This test function, test_mnist_datamodule(batch_size), is responsible for testing the functionality and correctness of the MNISTDataModule class in the src.data.mnist_datamodule module. Here's what the test does:

#The test is parametrized with two different batch sizes: [32, 128]. This means the test will be run twice, once with a batch size of 32 and once with a batch size of 128.

#Inside the test function:

#It creates an instance of MNISTDataModule with the specified data_dir and the batch size from the parametrization.
#Calls prepare_data() on the data module, which prepares the data (e.g., downloads the MNIST dataset if not available).
#Asserts that the data_train, data_val, and data_test attributes of the data module are not set (i.e., they should be None).
#Verifies that the MNIST dataset directory and its raw directory exist under the specified data_dir.
#Calls setup() on the data module to set up the train, validation, and test datasets and data loaders.
#Asserts that the data_train, data_val, and data_test attributes are now set (i.e., they are not None).
#Asserts that the train_dataloader(), val_dataloader(), and test_dataloader() methods return valid data loaders (not None).
#Verifies that the total number of data points (samples) across all datasets (train, val, test) is 70,000 (the total number of images in the MNIST dataset).
#Checks the structure of a batch returned by the train dataloader by calling next(iter(dm.train_dataloader())). It verifies the batch size, the data types of x and y tensors, and their shapes.
#The test ensures that the MNISTDataModule class can correctly prepare, set up, and provide data loaders for the MNIST dataset. It also verifies that the batch size, data types, and shapes are correct for the returned batches. By running the test with different batch sizes, it checks the data module's functionality with varying batch sizes.