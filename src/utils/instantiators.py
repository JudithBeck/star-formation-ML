from typing import List

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""

    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""

    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


#instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]: This function takes a configuration dictionary callbacks_cfg as input and returns a list of instantiated callbacks. It iterates through the callbacks_cfg dictionary and checks if each item is a DictConfig with the key "target" present. If so, it instantiates the corresponding callback object using Hydra's instantiate function and adds it to the callbacks list. The list of instantiated callbacks is then returned.

#instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]: This function takes a configuration dictionary logger_cfg as input and returns a list of instantiated loggers. It iterates through the logger_cfg dictionary and checks if each item is a DictConfig with the key "target" present. If so, it instantiates the corresponding logger object using Hydra's instantiate function and adds it to the logger list. The list of instantiated loggers is then returned.

#The DictConfig type is used to represent configuration dictionaries that are parsed and managed by Hydra. The pylogger module is used to get a Python logger, which is likely used for logging messages during the instantiation process.

#These utility functions can be utilized to create and manage callbacks and loggers based on the specified configurations, enabling flexible and modular logging and callback functionality in the broader context of the machine learning pipeline.