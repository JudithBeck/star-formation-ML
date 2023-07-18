import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.eval import evaluate
from src.train import train


@pytest.mark.slow
def test_train_eval(tmp_path, cfg_train, cfg_eval):
    """Train for 1 epoch with `train.py` and evaluate with `eval.py`"""
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.test = True

    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")

    HydraConfig().set_config(cfg_eval)
    test_metric_dict, _ = evaluate(cfg_eval)

    assert test_metric_dict["test/acc"] > 0.0
    assert abs(train_metric_dict["test/acc"].item() - test_metric_dict["test/acc"].item()) < 0.001



#This test function, test_train_eval(tmp_path, cfg_train, cfg_eval), is responsible for performing an end-to-end test of the training and evaluation processes using the train.py and eval.py scripts in the src module. Here's what the test does:

#The test is marked as @pytest.mark.slow, indicating that it may take a longer time to execute.

#Inside the test function:

#It asserts that the temporary directory path (tmp_path) is the same as the output directory specified in both cfg_train and cfg_eval configurations.
#It modifies the cfg_train by opening its dictionary and setting the max_epochs to 1 (to train for only one epoch) and enables testing (test = True).
#It sets the modified cfg_train as the current Hydra configuration using HydraConfig().set_config(cfg_train).
#Calls the train() function from src.train with the modified cfg_train configuration.
#Captures the returned train_metric_dict and ignores the second returned object (_).
#Asserts that a file named "last.ckpt" is present in the "checkpoints" directory under the temporary path.
#The test continues with the evaluation step:

#It modifies the cfg_eval by opening its dictionary and setting the ckpt_path to the path of the checkpoint generated during training (i.e., the path to "last.ckpt" in the checkpoints directory).
#It sets the modified cfg_eval as the current Hydra configuration using HydraConfig().set_config(cfg_eval).
#Calls the evaluate() function from src.eval with the modified cfg_eval configuration.
#Captures the returned test_metric_dict and ignores the second returned object (_).
#The test performs assertions on the evaluation results:

#It asserts that the accuracy (test/acc) in the test_metric_dict is greater than 0.0, indicating that the evaluation was successful.
#It calculates the absolute difference between the test accuracy (test/acc) from the training phase (train_metric_dict["test/acc"]) and the evaluation phase (test_metric_dict["test/acc"]). The test ensures that the absolute difference is less than 0.001, indicating that the accuracy values are almost the same in the training and evaluation phases.
#In summary, this test function verifies that the training and evaluation processes work as expected, and it ensures that the checkpoint generated during training is successfully used for evaluation. It also checks that the evaluation results are consistent with the training results, specifically in terms of accuracy#.