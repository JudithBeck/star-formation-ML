import logging

from lightning.pytorch.utilities import rank_zero_only


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


#In the 'utils' folder inside 'src', there is a Python function named get_pylogger. This function is used to initialize a multi-GPU-friendly Python command-line logger. The function customizes the logger to handle multi-GPU setups properly.

#Here's an explanation of the function:

#get_pylogger(name=__name__) -> logging.Logger: This function takes an optional argument name, which is set to __name__ (the name of the current module) by default. It returns a customized logger instance of the logging.Logger class.

#The function initializes a logger using logging.getLogger(name). The logger is given the name specified by the name argument, which is the name of the module where the logger is used.

#The function defines a tuple logging_levels containing different logging levels such as 'debug', 'info', 'warning', 'error', 'exception', 'fatal', and 'critical'.

#It iterates through the logging_levels and sets each logging level for the logger instance. It applies the rank_zero_only decorator from the 'lightning.pytorch.utilities' module to each logging level. The rank_zero_only decorator ensures that the logger will be executed only on the process with rank 0 in a distributed training setup. This is important in multi-GPU setups to avoid duplicated logs from each GPU process.

#The customized logger with the rank_zero_only decorator applied to each logging level is returned.

#The purpose of this function is to create a logger that can handle logging messages in multi-GPU training setups without duplicating logs across multiple GPUs. By using the rank_zero_only decorator, logs will only be printed once on the rank 0 process, preventing redundant log messages from each GPU process. This helps to maintain cleaner and more manageable logs during distributed training.