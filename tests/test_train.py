import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.train import train
from tests.helpers.run_if import RunIf


def test_train_fast_dev_run(cfg_train):
    """Run for 1 train, val and test step."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
    train(cfg_train)


@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu(cfg_train):
    """Run for 1 train, val and test step on GPU."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
    train(cfg_train)


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_gpu_amp(cfg_train):
    """Train 1 epoch on GPU with mixed-precision."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.precision = 16
    train(cfg_train)


@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_train):
    """Train 1 epoch with validation loop twice per epoch."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.val_check_interval = 0.5
    train(cfg_train)


@pytest.mark.slow
def test_train_ddp_sim(cfg_train):
    """Simulate DDP (Distributed Data Parallel) on 2 CPU processes."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 2
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 2
        cfg_train.trainer.strategy = "ddp_spawn"
    train(cfg_train)


@pytest.mark.slow
def test_train_resume(tmp_path, cfg_train):
    """Run 1 epoch, finish, and resume for another epoch."""
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1

    HydraConfig().set_config(cfg_train)
    metric_dict_1, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "last.ckpt" in files
    assert "epoch_000.ckpt" in files

    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_train.trainer.max_epochs = 2

    metric_dict_2, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "epoch_001.ckpt" in files
    assert "epoch_002.ckpt" not in files

    assert metric_dict_1["train/acc"] < metric_dict_2["train/acc"]
    assert metric_dict_1["val/acc"] < metric_dict_2["val/acc"]


#These test functions focus on different scenarios during the training process. Here's a summary of each test:

#test_train_fast_dev_run(cfg_train): This test runs one training, validation, and test step using fast_dev_run=True on the CPU. The train() function is called with the specified configuration.

#test_train_fast_dev_run_gpu(cfg_train): This test is similar to the previous one but runs the process on a GPU (if available).

#test_train_epoch_gpu_amp(cfg_train): This test trains one epoch on the GPU using mixed-precision (16-bit precision). The train() function is called with the specified configuration.

#test_train_epoch_double_val_loop(cfg_train): This test trains one epoch with the validation loop executed twice per epoch. The train() function is called with the specified configuration.

#test_train_ddp_sim(cfg_train): This test simulates Distributed Data Parallel (DDP) on two CPU processes. The training process runs for two epochs using DDP spawn strategy. The train() function is called with the specified configuration.

#test_train_resume(tmp_path, cfg_train): This test runs one epoch, saves the checkpoint, and resumes the training for another epoch. The train() function is called twice with different configurations for resuming.

#These tests cover various training scenarios, such as fast development runs, running on GPUs with mixed-precision, running DDP simulations, and resuming training from a checkpoint. They ensure that the training process works correctly under different conditions and produces the expected results. The RunIf decorator is used to conditionally skip tests that require certain resources (e.g., GPUs) that may not be available in the testing environment.