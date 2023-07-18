from typing import Any, Dict, Optional, Tuple

import torch
import os
import numpy as np
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from sklearn.preprocessing import MinMaxScaler
import joblib


class StarsDataModule(LightningDataModule):
    """
    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """


    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.1, 0.2),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:

            # load dataset
            spectra = np.loadtxt(os.path.join(self.hparams.data_dir, 'dataset_plusMass_smallRadius.txt'))
            labels = np.loadtxt(os.path.join(self.hparams.data_dir, 'parameterset_plusMass_smallRadius.txt'))

            NOISE = 0.75
            spectra_noise = np.random.normal(spectra, scale=NOISE)

            # only keep first 5 columns of labels
            labels = labels[:, :5]

            # log transform third label
            labels[:, 2] = np.log10(labels[:, 2])
            labels[:, 4] = np.log10(labels[:, 4])

            # scale labels (MinMaxScaler between 0 and 1)
            scaler = MinMaxScaler()
            labels = scaler.fit_transform(labels)

            # save scaler for later use
            joblib.dump(scaler, os.path.join(self.hparams.data_dir, 'scaler.gz'))

            # convert to torch tensors
            spectra_noise = torch.from_numpy(spectra_noise).float()
            labels = torch.from_numpy(labels).float()

            # create dataset
            dataset = StarsDataset(spectra_noise, labels)

            # split dataset
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

            np.save(os.path.join(cfg.paths.output_dir, "NOISE.npy"), NOISE)
            np.save(os.path.join(cfg.paths.output_dir, "spectra.npy"), spectra)


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


class StarsDataset(Dataset):
    def __init__(
            self,
            spectra_noise,
            labels,
    ):
        super().__init__()

        self.spectra_noise = spectra_noise
        self.labels = labels
        self.num_samples = len(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        x = self.spectra_noise[index]
        y = self.labels[index]
        return x, y


if __name__ == "__main__":
    _ = StarsDataModule()



#In the folder 'components' inside 'data' within the 'src' directory, you have two Python classes:

#StarsDataModule: This class is a PyTorch Lightning DataModule, which is used for data handling in PyTorch Lightning. It implements methods for preparing, setting up, and providing data loaders for training, validation, and testing. Here's a brief overview of its methods:

#prepare_data: This method is used for downloading data if needed. It doesn't assign state (self.x = y).
#setup: This method is used to load and split datasets (train, validation, and test) when the DataModule is called with trainer.fit() or trainer.test().
#train_dataloader: This method returns a DataLoader for the training dataset.
#val_dataloader: This method returns a DataLoader for the validation dataset.
#test_dataloader: This method returns a DataLoader for the test dataset.
#predict_dataloader: This method returns a DataLoader for the test dataset (used for prediction).
#teardown: This method is used for cleanup after fit or test.
#state_dict: This method returns extra things to save to the checkpoint.
#load_state_dict: This method specifies actions when loading a checkpoint.
#StarsDataset: This class is a custom Dataset class, which inherits from PyTorch's Dataset. It serves as a data container for the spectra and labels used in the StarsDataModule. The __len__ method returns the number of samples, and the __getitem__ method retrieves a sample by its index.

#The provided script (__name__ == "__main__") instantiates the StarsDataModule class, but it doesn't perform any specific operations in the current context.

#Please note that the given code only shows the data handling part of a machine learning pipeline. Other components like model training, validation, and testing would be part of a more comprehensive training script outside of this code snippet.