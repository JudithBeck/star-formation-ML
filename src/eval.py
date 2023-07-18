from typing import List, Tuple

import hydra
import pyrootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import os
import numpy as np
import torch
import joblib

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    # trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    y_pred = []
    predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    for batch in predictions:
        y_pred.append(batch)
    y_pred = np.array(torch.cat(y_pred, dim=0).detach())

    # a negative mass makes no sense, so we set negative values in the fifth column to 0 (min value of scaler)
    y_pred[:, 4] = np.where(y_pred[:, 4] < 0, 0, y_pred[:, 4])

    # collect input and true data
    x_test = []
    y_test = []
    test_dataloader = datamodule.test_dataloader()
    for batch in test_dataloader:
        x, y = batch 
        x_test.append(x) 
        y_test.append(y) 
    x_test = np.array(torch.cat(x_test, dim=0))
    y_test = np.array(torch.cat(y_test, dim=0))

    # load scaler and inverse transform
    scaler = joblib.load(os.path.join(cfg.paths.data_dir, "scaler.gz"))
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test)

    # inverse log10 transform of 3rd and 5th column
    y_pred_rescaled[:, 2] = 10**y_pred_rescaled[:, 2]
    y_pred_rescaled[:, 4] = 10**y_pred_rescaled[:, 4]

    # save data
    np.save(os.path.join(cfg.paths.output_dir, "x_test.npy"), x_test)
    np.save(os.path.join(cfg.paths.output_dir, "y_test.npy"), y_test)
    np.save(os.path.join(cfg.paths.output_dir, "y_test_rescaled.npy"), y_test_rescaled)
    np.save(os.path.join(cfg.paths.output_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(cfg.paths.output_dir, "y_pred_rescaled.npy"), y_pred_rescaled)
    
    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
