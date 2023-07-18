import pytest

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = "src/train.py"
overrides = ["logger=[]"]


@RunIf(sh=True)
@pytest.mark.slow
def test_experiments(tmp_path):
    """Test running all available experiment configs with fast_dev_run=True."""
    command = [
        startfile,
        "-m",
        "experiment=glob(*)",
        "hydra.sweep.dir=" + str(tmp_path),
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(tmp_path):
    """Test default hydra sweep."""
    command = [
        startfile,
        "-m",
        "hydra.sweep.dir=" + str(tmp_path),
        "model.optimizer.lr=0.005,0.01",
        "++trainer.fast_dev_run=true",
    ] + overrides

    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep_ddp_sim(tmp_path):
    """Test default hydra sweep with ddp sim."""
    command = [
        startfile,
        "-m",
        "hydra.sweep.dir=" + str(tmp_path),
        "trainer=ddp_sim",
        "trainer.max_epochs=3",
        "+trainer.limit_train_batches=0.01",
        "+trainer.limit_val_batches=0.1",
        "+trainer.limit_test_batches=0.1",
        "model.optimizer.lr=0.005,0.01,0.02",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_optuna_sweep(tmp_path):
    """Test optuna sweep."""
    command = [
        startfile,
        "-m",
        "hparams_search=mnist_optuna",
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_trials=10",
        "hydra.sweeper.sampler.n_startup_trials=5",
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)


@RunIf(wandb=True, sh=True)
@pytest.mark.slow
def test_optuna_sweep_ddp_sim_wandb(tmp_path):
    """Test optuna sweep with wandb and ddp sim."""
    command = [
        startfile,
        "-m",
        "hparams_search=mnist_optuna",
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_trials=5",
        "trainer=ddp_sim",
        "trainer.max_epochs=3",
        "+trainer.limit_train_batches=0.01",
        "+trainer.limit_val_batches=0.1",
        "+trainer.limit_test_batches=0.1",
        "logger=wandb",
    ]
    run_sh_command(command)


#These test functions use the RunIf decorator from the tests.helpers.run_if module to conditionally run certain test cases based on specific requirements. Here's a summary of each test function:

#test_experiments(tmp_path): This test runs all available experiment configurations with fast_dev_run=True. It uses Hydra to sweep over all possible configurations defined in the configs/experiments directory. The run_sh_command() function is called to execute the training process with the given command.

#test_hydra_sweep(tmp_path): This test performs a default Hydra sweep with two different learning rates for the optimizer. It uses the hydra.sweep.dir configuration to specify the output directory for the sweep results. The run_sh_command() function is called to execute the training process with the given command.

#test_hydra_sweep_ddp_sim(tmp_path): This test is similar to the previous one but uses Distributed Data Parallel (DDP) simulation for training. It also sets some additional configuration options for the DDP trainer. The run_sh_command() function is called to execute the training process with the given command.

#test_optuna_sweep(tmp_path): This test performs an Optuna sweep using the mnist_optuna configuration. It specifies the number of trials and startup trials for the Optuna sweeper. The run_sh_command() function is called to execute the training process with the given command.

#test_optuna_sweep_ddp_sim_wandb(tmp_path): This test is similar to the previous one but also uses Wandb as the logger and DDP simulation for training. The run_sh_command() function is called to execute the training process with the given command.

#All these test functions are decorated with @RunIf(sh=True) or @RunIf(wandb=True, sh=True), which means they will only be executed if the required conditions are met (i.e., the sh or wandb packages are available in the environment, and the tests are not skipped). The conditions are specified in the RunIf decorator using the sh and wandb flags, which check the availability of the sh and wandb packages, respectively.

#These tests are useful for evaluating different configurations and training options to ensure that the training process works as expected and produces meaningful results. They help ensure that the training and hyperparameter optimization processes are functioning correctly.