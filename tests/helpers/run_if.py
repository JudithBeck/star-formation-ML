"""Adapted from:

https://github.com/PyTorchLightning/pytorch-lightning/blob/master/tests/helpers/runif.py
"""

import sys
from typing import Optional

import pytest
import torch
from packaging.version import Version
from pkg_resources import get_distribution

from tests.helpers.package_available import (
    _COMET_AVAILABLE,
    _DEEPSPEED_AVAILABLE,
    _FAIRSCALE_AVAILABLE,
    _IS_WINDOWS,
    _MLFLOW_AVAILABLE,
    _NEPTUNE_AVAILABLE,
    _SH_AVAILABLE,
    _TPU_AVAILABLE,
    _WANDB_AVAILABLE,
)


class RunIf:
    """RunIf wrapper for conditional skipping of tests.

    Fully compatible with `@pytest.mark`.

    Example:

        @RunIf(min_torch="1.8")
        @pytest.mark.parametrize("arg1", [1.0, 2.0])
        def test_wrapper(arg1):
            assert arg1 > 0
    """

    def __new__(
        self,
        min_gpus: int = 0,
        min_torch: Optional[str] = None,
        max_torch: Optional[str] = None,
        min_python: Optional[str] = None,
        skip_windows: bool = False,
        sh: bool = False,
        tpu: bool = False,
        fairscale: bool = False,
        deepspeed: bool = False,
        wandb: bool = False,
        neptune: bool = False,
        comet: bool = False,
        mlflow: bool = False,
        **kwargs,
    ):
        """
        Args:
            min_gpus: min number of GPUs required to run test
            min_torch: minimum pytorch version to run test
            max_torch: maximum pytorch version to run test
            min_python: minimum python version required to run test
            skip_windows: skip test for Windows platform
            tpu: if TPU is available
            sh: if `sh` module is required to run the test
            fairscale: if `fairscale` module is required to run the test
            deepspeed: if `deepspeed` module is required to run the test
            wandb: if `wandb` module is required to run the test
            neptune: if `neptune` module is required to run the test
            comet: if `comet` module is required to run the test
            mlflow: if `mlflow` module is required to run the test
            kwargs: native pytest.mark.skipif keyword arguments
        """
        conditions = []
        reasons = []

        if min_gpus:
            conditions.append(torch.cuda.device_count() < min_gpus)
            reasons.append(f"GPUs>={min_gpus}")

        if min_torch:
            torch_version = get_distribution("torch").version
            conditions.append(Version(torch_version) < Version(min_torch))
            reasons.append(f"torch>={min_torch}")

        if max_torch:
            torch_version = get_distribution("torch").version
            conditions.append(Version(torch_version) >= Version(max_torch))
            reasons.append(f"torch<{max_torch}")

        if min_python:
            py_version = (
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            )
            conditions.append(Version(py_version) < Version(min_python))
            reasons.append(f"python>={min_python}")

        if skip_windows:
            conditions.append(_IS_WINDOWS)
            reasons.append("does not run on Windows")

        if tpu:
            conditions.append(not _TPU_AVAILABLE)
            reasons.append("TPU")

        if sh:
            conditions.append(not _SH_AVAILABLE)
            reasons.append("sh")

        if fairscale:
            conditions.append(not _FAIRSCALE_AVAILABLE)
            reasons.append("fairscale")

        if deepspeed:
            conditions.append(not _DEEPSPEED_AVAILABLE)
            reasons.append("deepspeed")

        if wandb:
            conditions.append(not _WANDB_AVAILABLE)
            reasons.append("wandb")

        if neptune:
            conditions.append(not _NEPTUNE_AVAILABLE)
            reasons.append("neptune")

        if comet:
            conditions.append(not _COMET_AVAILABLE)
            reasons.append("comet")

        if mlflow:
            conditions.append(not _MLFLOW_AVAILABLE)
            reasons.append("mlflow")

        reasons = [rs for cond, rs in zip(conditions, reasons) if cond]
        return pytest.mark.skipif(
            condition=any(conditions),
            reason=f"Requires: [{' + '.join(reasons)}]",
            **kwargs,
        )



#The code you provided defines a RunIf class, which is a custom wrapper for conditional skipping of tests in pytest. The RunIf class allows you to skip test functions based on various conditions, such as minimum GPU count, minimum PyTorch version, operating system, package availability, and more.

#Here's an overview of the RunIf class and its usage:

#The __new__ method is overridden to create instances of the RunIf class. This method is called when you use the @RunIf decorator on a test function.

#The __new__ method takes several keyword arguments that represent the conditions for skipping the test. For example:

#min_gpus: Minimum number of GPUs required to run the test.
#min_torch: Minimum PyTorch version required to run the test.
#max_torch: Maximum PyTorch version required to run the test.
#min_python: Minimum Python version required to run the test.
#skip_windows: If True, the test will be skipped on Windows platform.
#tpu, sh, fairscale, deepspeed, wandb, neptune, comet, mlflow: If True, the corresponding package is required to run the test.
#The method checks the specified conditions and stores the reasons for skipping the test in the reasons list.

#The RunIf class is designed to be fully compatible with @pytest.mark, which allows it to be used as a decorator alongside other pytest.mark decorators for test functions.

#When you decorate a test function with @RunIf, it will skip the test if any of the specified conditions are met. The test will be marked with a reason indicating why it was skipped based on the conditions that failed.

#For example, the following code shows how you can use the RunIf class to skip a test based on the PyTorch version:

#python
#Copy code
#@RunIf(min_torch="1.8")
#def test_wrapper():
#    assert True
#In this example, the test function test_wrapper will only run if the installed PyTorch version is greater than or equal to 1.8. If the condition is not met (i.e., PyTorch version is less than 1.8), the test will be skipped with the reason "Requires: torch>=1.8".

#The RunIf class provides a flexible way to conditionally skip tests based on various requirements, making it easier to handle different testing scenarios across different environments and configurations.