"""Rendering a grid value and a metric, shared by the two entry points.

Lifted from the MIMIC fork's ``reporting`` module, which is otherwise not
carried here: the rest of it typeset manuscript tables through a JMIR
formatting package, and a sweep needs to print a ranking rather than
typeset one. These two are the parts the decision record depends on, so
they live where both readers can reach them rather than in either one.
"""

import math

from typing import Any, Optional


def format_grid_value(value: Any) -> str:
    """Render a hyperparameter value for a row label.

    Args:
        value: The value.

    Returns:
        Its ``%g`` form for floats, so 2e-05 stays 2e-05 rather than
        becoming 0.00002, and ``null`` for None -- which in this campaign
        is a real cell, the flat learning-rate schedule, not a gap.
    """
    if value is None:
        return 'null'
    if isinstance(value, float):
        return f'{value:g}'
    return str(value)


def format_metric(value: Optional[float]) -> str:
    """Render a metric for a table cell.

    Args:
        value: The metric, or None.

    Returns:
        Four decimal places for anything in a readable range, scientific
        notation for very small values, and an em dash for a missing one.
    """
    if value is None:
        return '—'
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if not math.isfinite(value):
        # A diverged cell reaches here. Say 'nan' rather than formatting it
        # to four places, which reads like a number that was measured.
        return str(value)
    if value != 0 and abs(value) < 1e-3:
        return f'{value:.3e}'
    return f'{value:.4f}'
