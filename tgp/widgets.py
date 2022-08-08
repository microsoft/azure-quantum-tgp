# Copyright (c) Microsoft Corporation. All rights reserved.

from __future__ import annotations

import math

from IPython.display import display
import ipywidgets as wid
from ipywidgets import (
    Checkbox,
    FloatLogSlider,
    FloatSlider,
    Output,
    SelectionSlider,
    interactive,
)
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import tgp
from tgp import plot
from tgp.one import analyze, set_gapped


def plot_2w(ds: xr.Dataset, vmax_start) -> None:
    def _plot(cutter_pair_index, vmax):
        plot.one.plot_2w_at(ds, cutter_pair_index, vmax)

    cutter_pair_index = np.arange(0, len(ds.L_3w.cutter_pair_index))
    widget = interactive(
        _plot,
        cutter_pair_index=SelectionSlider(
            options=cutter_pair_index,
            value=0,
            continuous_update=False,
        ),
        vmax=FloatSlider(
            min=min(ds.L_2w_nl.min(), ds.R_2w_nl.min()),
            max=max(ds.L_2w_nl.max(), ds.R_2w_nl.max()),
            value=vmax_start,
            continuous_update=False,
        ),
    )
    display(widget)


def plot_3w(ds: xr.Dataset, lim_start) -> None:
    def _plot(cutter_pair_index, lim):
        plot.one.plot_3w_at(ds, cutter_pair_index, lim)

    cutter_pair_index = np.arange(0, len(ds.L_3w.cutter_pair_index))
    widget = interactive(
        _plot,
        cutter_pair_index=SelectionSlider(
            options=cutter_pair_index,
            value=0,
            continuous_update=False,
        ),
        lim=FloatSlider(
            min=min(ds.L_3w.min(), ds.R_3w.min()),
            max=max(ds.L_3w.max(), ds.R_3w.max()),
            value=lim_start,
            continuous_update=False,
        ),
    )
    display(widget)


def plot_set_gapped(ds: xr.Dataset, th_2w_p) -> interactive:
    def _plot(th_2w_p):
        set_gapped(ds, th_2w_p)
        plot.one.plot_gapped(ds)

    display(
        interactive(
            _plot, th_2w_p=FloatSlider(min=0.0, max=1.0, step=0.01, value=th_2w_p)
        )
    )


def plot_analysis(ds: xr.Dataset, n_clusters: int = 11) -> None:
    first_time = True

    def _plot_and_analyze(
        th_2w_p,
        th_3w,
        th_3w_tat,
        _plot_gapped,
        _plot_clusters,
        _plot_zbp,
        _plot_stage_1,
        _plot_stage_1_gapped_clusters,
    ):
        thresholds = ds.attrs.get("thresholds", {}).copy()
        thresholds.update({"th_2w_p": th_2w_p, "th_3w": th_3w, "th_3w_tat": th_3w_tat})
        nonlocal first_time
        analyze(ds, thresholds, force=first_time)
        first_time = False
        if _plot_gapped:
            plot.one.plot_gapped(ds)
        if _plot_clusters:
            plot.one.plot_clusters(ds)
        if _plot_zbp:
            plot.one.plot_zbp(ds)
        if _plot_stage_1:
            plot.one.plot_stage_1(ds, n_clusters)
        if _plot_stage_1_gapped_clusters:
            plot.one.plot_stage_1_gapped_clusters(ds, n_clusters)

    th_old = ds.attrs.setdefault("thresholds", {})

    # Calculate limits of '3w' slider
    _max = np.maximum(ds.L_3w, ds.R_3w)
    min_3w = -_max.values.min()
    # Not terribly smart, pick mean of negative maxima as initial guess
    th_3w_start = np.nanmean(_max.where(_max > 0, drop=True))
    slider_style = dict(
        continuous_update=False,
        layout={"width": "500px"},
        style={"description_width": "initial"},
    )
    widget = interactive(
        _plot_and_analyze,
        th_2w_p=FloatSlider(
            min=0,
            max=1,
            step=0.01,
            value=th_old.get("th_2w_p", 0.5),
            description="th_2w_p: fraction of cutter values, sets the gap",
            **slider_style,
        ),
        th_3w=FloatLogSlider(
            min=0,
            max=math.log10(min_3w),
            value=th_old.get("th_3w", math.log10(th_3w_start)),
            description="th_3w: abs value of 3ω conductance, sets the ZBPs",
            **slider_style,
        ),
        th_3w_tat=FloatSlider(
            min=0,
            max=1,
            step=0.01,
            value=th_old.get("th_3w_tat", 0.5),
            description="th_3w_tat: fraction of cutter values, sets the ZBPs",
            **slider_style,
        ),
        _plot_gapped=Checkbox(value=True, description="plot_gapped"),
        _plot_clusters=Checkbox(value=True, description="plot_clusters"),
        _plot_zbp=Checkbox(value=True, description="plot_zbp"),
        _plot_stage_1=Checkbox(value=True, description="plot_stage_1"),
        _plot_stage_1_gapped_clusters=Checkbox(
            value=True, description="plot_stage_1_gapped_clusters"
        ),
    )
    avoid_widgets_output_scroll()
    display(widget)


def avoid_widgets_output_scroll():
    style = """
        <style>
            .jupyter-widgets-output-area .output_scroll {
                height: unset !important;
                border-radius: unset !important;
                -webkit-box-shadow: unset !important;
                box-shadow: unset !important;
            }
            .jupyter-widgets-output-area  {
                height: auto !important;
            }
        </style>
        """
    display(wid.HTML(style))


def plot_gap_extraction_interactive(ds_left, ds_right) -> None:
    out = Output()

    def _plot(
        cutter_value, plunger_value, plot_filtered_conductance, with_2d_plot
    ) -> mpl.figure.Figure:
        with out:
            out.clear_output(wait=True)
            plot.two.plot_gap_extraction(
                ds_left,
                ds_right,
                cutter_value,
                plunger_value,
                plot_filtered_conductance,
            )
            if with_2d_plot:
                plot.two.plot_extracted_gap(
                    ds_left, ds_right, cutter_value, plunger_hline=plunger_value
                )
            plt.show()

    widget = interactive(
        _plot,
        cutter_value=list(ds_left.cutter_pair_index.values),
        plunger_value=list(ds_left.V.values),
        plot_filtered_conductance=[True, False],
        with_2d_plot=[True, False],
    )
    return display(widget, out)


def stage_2_interactive(
    ds_left: xr.Dataset,
    ds_right: xr.Dataset,
    norm: mpl.colors.Normalize | None = None,
    unit=None,
) -> interactive:
    left = ds_left
    right = ds_right
    ps = left.coords["V"].values
    cutters = left.coords["cutter_pair_index"].values
    label = unit or ""

    def _plot(p, cutter_pair_index):
        left_sel = left.sel({"V": p, "cutter_pair_index": cutter_pair_index}).squeeze()
        right_sel = right.sel(
            {"V": p, "cutter_pair_index": cutter_pair_index}
        ).squeeze()
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 5))
        left_sel["g_ll"].squeeze().transpose("left_bias", "B").plot.imshow(
            ax=axs[0, 0],
            add_colorbar=True,
            cbar_kwargs=dict(label=label),
            cmap="viridis",
            vmin=0,
        )
        axs[0, 0].set_title("$G_{ll}$")
        right_sel["g_rr"].squeeze().transpose("right_bias", "B").plot.imshow(
            ax=axs[0, 1],
            add_colorbar=True,
            cbar_kwargs=dict(label=label),
            cmap="viridis",
            vmin=0,
        )
        axs[0, 1].set_title("$G_{rr}$")

        left_sel["g_rl"].squeeze().transpose("left_bias", "B").plot.imshow(
            ax=axs[1, 0], norm=norm, add_colorbar=True, cbar_kwargs=dict(label=label)
        )
        axs[1, 0].set_title("$G_{rl}$")

        right_sel["g_lr"].squeeze().transpose("right_bias", "B").plot.imshow(
            ax=axs[1, 1], norm=norm, add_colorbar=True, cbar_kwargs=dict(label=label)
        )
        axs[1, 1].set_title("$G_{lr}$")

        plt.tight_layout()
        plt.show()
        return fig

    return interactive(
        _plot,
        p=SelectionSlider(options=list(ps)),
        cutter_pair_index=SelectionSlider(options=list(cutters)),
    )


def plot_gapped_zbp(zbp_ds) -> None:
    out = Output()

    def _plot(cutter_value, threshold_low, threshold_high) -> mpl.figure.Figure:
        with out:
            tgp.two.set_gap_threshold(zbp_ds, threshold_low, threshold_high)
            out.clear_output(wait=True)
            plot.zbp.plot_gapped_zbp(zbp_ds, cutter_value)
            plt.show()

    gaps = sorted(np.unique(zbp_ds["gap"].values.flat))
    widget = interactive(
        _plot,
        cutter_value=np.atleast_1d(zbp_ds.cutter_pair_index.values.tolist()),
        threshold_low=gaps,
        threshold_high=[None] + gaps,
    )
    display(widget, out)
