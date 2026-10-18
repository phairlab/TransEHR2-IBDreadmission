"""Machinery for the hyperparameter sweep: what to run, and what ran.

Two stages, with command-line entry points under ``scripts/``:

    generate_tuning_configs.py   spec -> one experiment config per trial,
                                 plus the manifest the array scripts index
    tuning_trial.py              manifest + array index -> a trial's name
                                 or config path, for a SLURM job
    report_tuning_results.py     manifest -> progress and the ranking
    select_tuned_cell.py         manifest -> the winning cell, as a record

The trials themselves are ordinary single-GPU ``fit_transehr2_deephit.py``
jobs. Nothing here trains anything; it decides what to run and reads back
what ran.

Ported from the MIMIC-IV fork, less its ``reporting`` module, which
rendered manuscript tables through a JMIR formatting package this repo
does not carry. Selection lives in ``results``; presentation is plain
text until there are results worth tabulating.
"""
