# Copyright (c) Microsoft Corporation. All rights reserved.
import functools

import xarray as xr

from tgp import frequency_correction, one, plot, prepare, two
from tgp._version import __version__

__all__ = [
    "__version__",
    "one",
    "plot",
    "prepare",
    "two",
    "frequency_correction",
]

try:
    from tgp import data  # noqa: F401

    __all__.append("data")
except (ImportError, ModuleNotFoundError):
    pass


def _bind(f):
    @functools.wraps(f)
    def _wrapped(*args, **kwargs):
        self, *_args = args
        ds = self._obj
        return f(ds, *_args, **kwargs)

    return _wrapped


@xr.register_dataset_accessor("one")
class One:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    analyze = _bind(one.analyze)
    get_zoomin_ranges = _bind(one.get_zoomin_ranges)
    plot_1w = _bind(plot.one.plot_1w)
    plot_2w = _bind(plot.one.plot_2w)
    plot_2w_at = _bind(plot.one.plot_2w_at)
    plot_2w_interactive = _bind(plot.one.plot_2w_interactive)
    plot_2w_th = _bind(plot.one.plot_2w_th)
    plot_2w_th_avg = _bind(plot.one.plot_2w_th_avg)
    plot_3w = _bind(plot.one.plot_3w)
    plot_3w_at = _bind(plot.one.plot_3w_at)
    plot_3w_interactive = _bind(plot.one.plot_3w_interactive)
    plot_3w_tat = _bind(plot.one.plot_3w_tat)
    plot_3w_th = _bind(plot.one.plot_3w_th)
    plot_analysis_interactive = _bind(plot.one.plot_analysis_interactive)
    plot_clusters = _bind(plot.one.plot_clusters)
    plot_gapped = _bind(plot.one.plot_gapped)
    plot_set_gapped_interactive = _bind(plot.one.plot_set_gapped_interactive)
    plot_stage_1 = _bind(plot.one.plot_stage_1)
    plot_stage_1_gapped_clusters = _bind(plot.one.plot_stage_1_gapped_clusters)
    plot_zbp = _bind(plot.one.plot_zbp)
    plot_zoomed_clusters = _bind(plot.one.plot_zoomed_clusters)
    set_2w_th = _bind(one.set_2w_th)
    set_3w_tat = _bind(one.set_3w_tat)
    set_3w_th = _bind(one.set_3w_th)
    set_clusters = _bind(one.set_clusters)
    set_gapped = _bind(one.set_gapped)


@xr.register_dataset_accessor("two")
class Two:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    plot_clusters_zbp = _bind(plot.zbp.plot_clusters_zbp)
    plot_gapped_zbp = _bind(plot.zbp.plot_gapped_zbp)
    plot_gapped_zbp_interactive = _bind(plot.zbp.plot_gapped_zbp_interactive)
    plot_joined_probability = _bind(plot.zbp.plot_joined_probability)
    plot_left_right_zbp_probability = _bind(plot.zbp.plot_left_right_zbp_probability)
    plot_probability_and_clusters = _bind(plot.zbp.plot_probability_and_clusters)
    plot_region_of_interest_2 = _bind(plot.zbp.plot_region_of_interest_2)
    plot_results_interactive = _bind(plot.zbp.plot_results_interactive)
    set_gap_threshold = _bind(two.set_gap_threshold)
    set_zbp_gap = _bind(two.set_zbp_gap)
    set_clusters = _bind(two.set_clusters)
    set_score = _bind(two.set_score)
    cluster_and_score = _bind(two.cluster_and_score)
