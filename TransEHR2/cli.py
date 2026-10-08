"""The few things every entry point in ``scripts/`` needs.

These lived in ``run_experiment.py`` and the other scripts imported them from
it, which made one CLI a library for the others: importing
``dump_finetuned_predictions`` pulled in the whole experiment driver, and the
import only resolved because both files sat in the same directory. They are
package code, so they live in the package.
"""

import os
import re
import torch

from typing import List, Optional


# The finetuning stage's name on disk, under ``{MODEL_DIR}/{experiment}/
# {fold}/``. The MIMIC fork ran one directory per prediction task; there is
# one task now, and it is less a task name than a description of the model.
TASK = 'competing_risks'

# The fold hyperparameter tuning runs on, held out of every reported result.
# Selecting hyperparameters on a fold and then reporting that fold's score
# spends the cross-validation on the choice it is meant to be independent
# of, so IBDdataprep's split.py cuts one extra fold and the entry points
# below drop this one by default. Naming a fold explicitly on the command
# line overrides that, which is how the sweep itself reaches fold0.
TUNING_FOLD = 'fold0'


def get_fold_names(data_dir: str,
                   exclude: Optional[List[str]] = None) -> List[str]:
    """Fold subdirectories of ``data_dir``, sorted.

    Args:
        data_dir (str): Directory holding the fold subdirectories.
        exclude (List[str], optional): Fold names to leave out.

    Returns:
        List[str]: The fold directory names, ascending.
    """
    exclude = exclude or []
    fold_names = [
        item for item in os.listdir(data_dir)
        if item not in exclude and re.match(r'fold\d+', item)
        and os.path.isdir(os.path.join(data_dir, item))
    ]
    fold_names.sort()
    return fold_names


def resolve_device(requested: Optional[str]) -> torch.device:
    """The one device a process trains on.

    An explicit ``--device`` wins; otherwise CUDA if there is any, then
    Apple's MPS, then the CPU. Dispatching folds across cards is the
    caller's job, usually through CUDA_VISIBLE_DEVICES, which makes the
    chosen card ``cuda:0`` inside each process.

    Args:
        requested (str, optional): The ``--device`` argument, or None.

    Returns:
        torch.device: The device to put the model and every batch on.
    """
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
