"""This file prepares config fixtures for other tests."""

import pyrootutils
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=[])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(pyrootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.01
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="eval.yaml", return_hydra_config=True, overrides=["ckpt_path=."])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(pyrootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


# this is called by each test which uses `cfg_train` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_train(cfg_train_global, tmp_path) -> DictConfig:
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


# this is called by each test which uses `cfg_eval` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global, tmp_path) -> DictConfig:
    cfg = cfg_eval_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()



#The code you provided contains fixtures for preparing config objects to be used in other tests. These fixtures help set up and modify configurations based on Hydra and OmegaConf to be used for testing purposes. Here's a breakdown of the code:

#The code uses pytest fixtures, which are functions that are automatically run before and after tests.

#The cfg_train_global fixture is responsible for generating the configuration object for training (cfg_train) that is shared across all test functions in the package.

#The cfg_eval_global fixture is responsible for generating the configuration object for evaluation (cfg_eval) that is shared across all test functions in the package.

#Both cfg_train_global and cfg_eval_global use the initialize function from Hydra to set up the Hydra framework for composing configurations from YAML files.

#The compose function is used to load the YAML configuration files train.yaml and eval.yaml respectively, and return the composed configuration objects as cfg.

#The configuration objects cfg_train_global and cfg_eval_global are then modified by using open_dict to set default values for certain configuration options that are shared across all tests.

#Two additional fixtures, cfg_train and cfg_eval, are defined to generate specific configuration objects for individual test functions. These fixtures use cfg_train_global and cfg_eval_global, respectively, as the starting point and then modify them further to set up temporary output and log directories for each test function.

#The cfg_train fixture is used for tests related to the training functionality, and the cfg_eval fixture is used for tests related to the evaluation functionality.

#The yield statement inside the fixtures indicates the point where the actual test is executed. The test function that uses the fixture receives the modified configuration object (cfg) and can use it during the test.

#After the test function has completed, the GlobalHydra instance is cleared to ensure that configurations do not leak between test functions.

#In summary, these fixtures provide a convenient way to generate and modify configuration objects for different test scenarios, ensuring that each test has its own temporary directories for outputs and logs, while still sharing some common configuration settings across all tests.