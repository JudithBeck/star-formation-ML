from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


#In the 'utils' folder inside 'src', there is a Python function named log_hyperparameters. This function is used to control which parts of the configuration are saved by the Lightning loggers during the training process. Additionally, it saves the number of model parameters.

#Here's an explanation of the function:

#@rank_zero_only: This decorator indicates that the function should only be executed on the process with rank 0 in a distributed training setup. This ensures that hyperparameter logging is performed only once, avoiding redundant logging across multiple processes.

#log_hyperparameters(object_dict: dict) -> None: This function takes a dictionary object_dict as input and performs the hyperparameter logging. The dictionary is expected to contain the following keys:

#"cfg": The configuration object (expected to be an OmegaConf config).
#"model": The model object.
#"trainer": The Lightning trainer object.
#The function starts by creating an empty dictionary hparams to store the hyperparameters.

#It then extracts the relevant parts from the configuration (cfg), model, and trainer objects.

#The total number of model parameters, trainable parameters, and non-trainable parameters are computed and added to the hparams dictionary under the key "model/params/total", "model/params/trainable", and "model/params/non_trainable", respectively.

#The other relevant parts of the configuration, such as data, trainer settings, callbacks, extras, task_name, tags, checkpoint path, and seed are also added to the hparams dictionary.

#Finally, the hparams dictionary is sent to all loggers associated with the trainer using logger.log_hyperparams(hparams). This ensures that the hyperparameters are logged by all the configured loggers.

#The purpose of this function is to provide a centralized way to log hyperparameters to all configured loggers during the training process, making it easier to track and analyze the settings and model details used for each experiment. The @rank_zero_only decorator ensures that the logging is performed only once, preventing redundant logs in distributed training scenarios.