# Copyright (c) Microsoft Corporation. All rights reserved.

from __future__ import annotations

import matplotlib.pyplot as plt

from tgp.plot import one, two, zbp

__all__ = ["set_mpl_rc", "one", "two", "zbp"]


def set_mpl_rc(
    small_size: int = 13,
    medium_size: int = 13,
    bigger_size: int = 15,
) -> None:
    """Set the matplotlib rc parameters."""
    plt.rc("font", size=small_size)  # controls default text sizes
    plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=medium_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=small_size)  # legend fontsize
    plt.rc("figure", titlesize=bigger_size)  # fontsize of the figure title
