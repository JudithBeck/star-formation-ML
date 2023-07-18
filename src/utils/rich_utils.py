from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        save_to_file (bool, optional): Whether to export config to the hydra output folder.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in config."""

    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        log.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)





#In the 'utils' folder inside 'src', there are two Python functions: print_config_tree and enforce_tags. These functions are used to print the configuration tree using the Rich library and prompt the user to input tags, respectively.

#print_config_tree(cfg: DictConfig, print_order: Sequence[str], resolve: bool, save_to_file: bool) -> None: This function takes a configuration dictionary cfg (expected to be an OmegaConf config) as input and prints its content using the Rich library's tree structure. It allows customizing the order in which config components are printed through the print_order parameter. The resolve parameter determines whether to resolve reference fields of the DictConfig. The save_to_file parameter determines whether to export the config tree to the hydra output folder.

#The function starts by creating a tree using Rich's rich.tree.Tree class.
#It builds a queue of fields to add to the tree based on the print_order.
#Then, it adds all the other fields from the configuration not specified in the print_order to the queue.
#For each field in the queue, it adds a branch to the tree and adds the content of the corresponding DictConfig as YAML syntax to the branch.
#The function then prints the config tree using rich.print(tree).
#If save_to_file is True, it saves the config tree to a file named "config_tree.log" in the hydra output directory.
#enforce_tags(cfg: DictConfig, save_to_file: bool) -> None: This function takes a configuration dictionary cfg (expected to be an OmegaConf config) as input and prompts the user to input tags from the command line if no tags are provided in the config. The save_to_file parameter determines whether to save the tags to a file.

#If the configuration does not have any tags (cfg.get("tags") returns None or an empty list), it prompts the user to input a list of comma-separated tags from the command line.
#The entered tags are split and trimmed, and then the tags are assigned to the cfg.tags field using with open_dict(cfg):.
#If save_to_file is True, it saves the tags to a file named "tags.log" in the hydra output directory.
#These utility functions enhance the Hydra-based configuration management by providing a more visually appealing way to print the configuration tree and ensuring the presence of user-defined tags in the configuration for better experiment organization and identification.