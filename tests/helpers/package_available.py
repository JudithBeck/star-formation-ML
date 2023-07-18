import platform

import pkg_resources
from lightning.fabric.accelerators import TPUAccelerator


def _package_available(package_name: str) -> bool:
    """Check if a package is available in your environment."""
    try:
        return pkg_resources.require(package_name) is not None
    except pkg_resources.DistributionNotFound:
        return False


_TPU_AVAILABLE = TPUAccelerator.is_available()

_IS_WINDOWS = platform.system() == "Windows"

_SH_AVAILABLE = not _IS_WINDOWS and _package_available("sh")

_DEEPSPEED_AVAILABLE = not _IS_WINDOWS and _package_available("deepspeed")
_FAIRSCALE_AVAILABLE = not _IS_WINDOWS and _package_available("fairscale")

_WANDB_AVAILABLE = _package_available("wandb")
_NEPTUNE_AVAILABLE = _package_available("neptune")
_COMET_AVAILABLE = _package_available("comet_ml")
_MLFLOW_AVAILABLE = _package_available("mlflow")


#In this 'helpers' module, several utility functions are defined to check the availability of certain packages and hardware accelerators. Let's go through each function:

#_package_available(package_name: str) -> bool: This function checks if a given package is available in the current environment. It uses pkg_resources.require to determine if the package is installed and accessible. If the package is not found (raises pkg_resources.DistributionNotFound), it returns False; otherwise, it returns True.

#_TPU_AVAILABLE: This variable is set to the availability of the TPUAccelerator. It uses the is_available method from the TPUAccelerator class to check if TPU acceleration is available.

#_IS_WINDOWS: This variable is set to True if the platform is Windows; otherwise, it is set to False. It uses the platform.system() method to get the current platform.

#_SH_AVAILABLE: This variable is set to True if the 'sh' package is available in the environment. It checks package availability using the _package_available function and verifies that the platform is not Windows.

#_DEEPSPEED_AVAILABLE: This variable is set to True if the 'deepspeed' package is available in the environment. It checks package availability using the _package_available function and verifies that the platform is not Windows.

#_FAIRSCALE_AVAILABLE: This variable is set to True if the 'fairscale' package is available in the environment. It checks package availability using the _package_available function and verifies that the platform is not Windows.

#_WANDB_AVAILABLE: This variable is set to True if the 'wandb' package is available in the environment. It checks package availability using the _package_available function.

#_NEPTUNE_AVAILABLE: This variable is set to True if the 'neptune' package is available in the environment. It checks package availability using the _package_available function.

#_COMET_AVAILABLE: This variable is set to True if the 'comet_ml' package is available in the environment. It checks package availability using the _package_available function.

#_MLFLOW_AVAILABLE: This variable is set to True if the 'mlflow' package is available in the environment. It checks package availability using the _package_available function.

#These utility functions can be used to check the availability of certain packages and hardware accelerators, which can help conditionally execute certain parts of the code based on the environment and available resources. For example, certain code blocks may require 'deepspeed' for distributed training or 'wandb' for logging, and these functions can be used to check if these packages are available before using them.