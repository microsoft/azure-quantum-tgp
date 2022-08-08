# Copyright (c) Microsoft Corporation. All rights reserved.

from __future__ import annotations

import functools
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sized

from IPython.display import display
import ipywidgets as wid
from ipywidgets import (
    Checkbox,
    FloatLogSlider,
    FloatSlider,
    SelectionSlider,
    interactive,
)
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from tgp import plot
from tgp.common import ds_requires, expand_clusters
from tgp.one import analyze, get_thresholds, set_gapped

PLOT_KW = dict(x="B", y="V", shading="auto")
_STYLE = dict(layout={"width": "500px"}, style={"description_width": "initial"})
_SLIDER_STYLE = dict(_STYLE, continuous_update=False)


def _check_formattable_string(fname):
    if "{}" not in fname:
        raise ValueError(
            "This function saves multiple plots and requires a formattable"
            " string (with '{}' in it), e.g., 'plot_{}.png'."
        )


@ds_requires(dims=("B", "V", "cutter_pair_index"))
def _plot_grid(
    ds: xr.Dataset,
    vmax: float | None,
    keys: Iterable[str],
    titles: Iterable[str],
    ncols: int = 6,
    vmin: float | None = 0,
    cmap: str = "viridis",
    plot_kwargs: dict[str, Any] | None = None,
    minimal: bool = False,
    fname: str | None = None,
) -> None:
    kw = dict(PLOT_KW, **(plot_kwargs or {}))
    if minimal:
        kw["add_colorbar"] = False
        kw["aspect"] = 1.8

    for key, title in zip(keys, titles):
        pl = ds[key].plot.pcolormesh(
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            col_wrap=ncols,
            col="cutter_pair_index",
            **kw,
        )
        pl.fig.suptitle(title, y=1.02)
        for ax in pl.axes.ravel():
            ax.set_axis_off()
            if minimal:
                ax.set_title("")
        if minimal:
            pl.fig.tight_layout()
        if fname is not None:
            _check_formattable_string(fname)
            pl.fig.savefig(fname.format(key), dpi=200)
    plt.show()


plot_1w = functools.partial(
    _plot_grid,
    keys=("L_1w", "R_1w"),
    titles=(r"$1\omega$ left local", r"$1\omega$ right local"),
    cmap="inferno",
)

plot_2w = functools.partial(
    _plot_grid,
    cmap="cividis",
    keys=("L_2w_nl", "R_2w_nl"),
    titles=(r"$2\omega$ left non-local", r"$2\omega$ right non-local"),
)

plot_2w_th = functools.partial(
    _plot_grid,
    vmax=None,
    cmap="cividis",
    keys=("L_2w_nl_t", "R_2w_nl_t", "LR_2w_nl_t"),
    titles=(r"L_2w_nl_t", r"R_2w_nl_t", r"L_2w_nl_t * R_2w_nl_t"),
)

plot_3w = functools.partial(
    _plot_grid,
    cmap="RdBu_r",
    keys=("L_3w", "R_3w"),
    titles=(r"$3\omega$ left local", r"$3\omega$ right local"),
)

plot_3w_th = functools.partial(
    _plot_grid,
    vmax=None,
    keys=("L_3w_t", "R_3w_t", "LR_3w_t"),
    titles=(
        r"ZBP on left ($=3\omega$ left thresholded)",
        r"ZBP on right ($=3\omega$ right thresholded)",
        r"ZBP on both ends",
    ),
)


@ds_requires(dims=("B", "V"))
def _plot_multiple(
    ds: xr.Dataset,
    sel: dict[str, float] | None,
    vmax: float | None,
    keys: Sized[str],
    labels: Sized[str] | None,
    vmin: float | None = 0,
    max_cols: int = 2,
    cmap: str | list[str] = "cividis",
    title: str | None = None,
    returns: bool = False,
    fname: str | Path | None = None,
) -> mpl.axes.Axes | None:
    ncols = min(max_cols, len(keys))
    nrows = math.ceil(len(keys) / ncols)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 8))
    axs = np.atleast_1d(axs).flatten()
    if labels is None:
        labels = keys
    if not isinstance(cmap, list):
        cmap = [cmap] * len(keys)
    for key, label, cm, ax in zip(keys, labels, cmap, axs):
        da = ds[key]
        if sel is not None:
            da = da.sel(**sel)
        da.attrs["long_name"] = label
        da.plot.pcolormesh(vmax=vmax, vmin=vmin, ax=ax, label=label, cmap=cm, **PLOT_KW)
    for ax in axs[len(keys) :]:
        ax.remove()
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    if returns:
        return axs
    if fname is not None:
        fig.savefig(fname, dpi=200)
    plt.show()


@ds_requires(variables=("L_2w_nl", "R_2w_nl"), dims=("B", "V", "cutter_pair_index"))
def plot_2w_at(
    ds: xr.Dataset,
    cutter_pair_index: int,
    vmax: float | None,
    *,
    fname: str | Path | None = None,
) -> None:
    """Plot 2w at a given cutter pair index.

    Parameters
    ----------
    ds
        Stage 1 `xarray.Dataset`.
    cutter_pair_index
        Cutter pair index to plot.
    vmax
        Maximum value to plot.
    fname
        File name to save plot to. If None, no plot is saved.
    """
    _plot_multiple(
        ds,
        sel={"cutter_pair_index": cutter_pair_index},
        vmax=vmax,
        keys=["L_2w_nl", "R_2w_nl"],
        labels=[r"$2\omega$ left non-local", r"$2\omega$ right non-local"],
        fname=fname,
    )


@ds_requires(variables=("L_2w_nl_ta", "R_2w_nl_ta"), dims=("B", "V"))
def plot_2w_th_avg(
    ds: xr.Dataset, vmax: float | None = None, *, fname: str | Path | None = None
) -> None:
    """Plot 2w thresholded average.

    Parameters
    ----------
    ds
        Stage 1 `xarray.Dataset`.
    vmax
        Maximum value to plot.
    fname
        File name to save plot to. If None, no plot is saved.
    """
    _plot_multiple(
        ds,
        sel=None,
        vmax=vmax,
        keys=["L_2w_nl_ta", "R_2w_nl_ta"],
        labels=[
            r"$2\omega$ left non-local averaged",
            r"$2\omega$ right non-local averaged",
        ],
        title=r"2$\omega$ non-local thresholded and averaged",
        fname=fname,
    )


@ds_requires(variables=("L_3w", "R_3w"), dims=("B", "V", "cutter_pair_index"))
def plot_3w_at(
    ds: xr.Dataset,
    cutter_pair_index: int,
    lim: float | None,
    *,
    fname: str | Path | None = None,
) -> None:
    """Plot 3w at a given cutter pair index.

    Parameters
    ----------
    ds
        Stage 1 `xarray.Dataset`.
    cutter_pair_index
        Cutter pair index to plot.
    lim
        vmin and vmax to use in the plot.
    fname
        File name to save plot to. If None, no plot is saved.
    """
    _plot_multiple(
        ds.real,
        sel={"cutter_pair_index": cutter_pair_index},
        vmin=-lim,
        vmax=lim,
        cmap="RdBu_r",
        keys=["L_3w", "R_3w"],
        labels=[r"$3\omega$ left local", r"$3\omega$ right local"],
        fname=fname,
    )


@ds_requires(variables=("L_gapped", "R_gapped", "gapped"), dims=("B", "V"))
def plot_gapped(ds: xr.Dataset, *, fname: str | Path | None = None) -> None:
    """Plot gapped arrays on the left and right, and the joined array.

    Parameters
    ----------
    ds
        Stage 1 `xarray.Dataset`.
    fname
        File name to save plot to. If None, no plot is saved.
    """
    _plot_multiple(
        ds,
        sel=None,
        vmax=1,
        cmap=["viridis", "viridis", "viridis_r"],
        keys=["L_gapped", "R_gapped", "gapped"],
        labels=[
            r"Averaged thresholded $2\omega$ left non-local",
            r"Averaged thresholded $2\omega$ right non-local",
            "gapped",
        ],
        fname=fname,
    )


@ds_requires(variables=("L_3w_tat", "R_3w_tat", "LR_3w_tat"), dims=("B", "V"))
def plot_3w_tat(ds: xr.Dataset, *, fname: str | Path | None = None) -> None:
    """Plot 3w thresholded-averaged-tresholded conductance.

    Parameters
    ----------
    ds
        Stage 1 `xarray.Dataset`.
    fname
        File name to save plot to. If None, no plot is saved.
    """
    th_3w_tat = ds.attrs["set_3w_tat.th_3w_tat"]
    _plot_multiple(
        ds,
        sel=None,
        vmin=0,
        vmax=1,
        cmap="viridis",
        keys=["L_3w_tat", "R_3w_tat", "LR_3w_tat"],
        labels=[
            f"Left ZBP probability > {th_3w_tat}",
            f"Right ZBP probability > {th_3w_tat}",
            f"Left ZBP probability > {th_3w_tat}  &  right ZBP probability > {th_3w_tat}",
        ],
        fname=fname,
    )


@ds_requires(variables=("L_3w_ta", "R_3w_ta"), dims=("B", "V"))
def plot_zbp(ds: xr.Dataset, *, fname: str | Path | None = None) -> None:
    """Plot the ZBP probability on the left and right.

    Parameters
    ----------
    ds
        Stage 1 `xarray.Dataset`.
    fname
        File name to save plot to. If None, no plot is saved.
    """
    num_cutters = len(ds.cutter_pair_index)
    _plot_multiple(
        ds,
        sel=None,
        vmin=0,
        vmax=1,
        keys=["L_3w_ta", "R_3w_ta"],
        labels=["Left ZBP probability", "Right ZBP probability"],
        title=f"ZBP probability averaged over {num_cutters} cutters",
        cmap="viridis",
        fname=fname,
    )


@ds_requires(variables=("clusters",), dims=("B", "V"))
def plot_clusters(ds: xr.Dataset, *, fname: str | Path | None = None) -> None:
    """Plot the clusters.

    Parameters
    ----------
    ds
        Stage 1 `xarray.Dataset`.
    fname
        File name to save plot to. If None, no plot is saved.
    """
    n = ds.clusters.max().item()
    cmap = plt.get_cmap("turbo", n)
    cmap.set_bad(alpha=0)
    levels = np.arange(1, n + 1)
    qm = ds.clusters.where(ds.clusters >= 1).plot.pcolormesh(
        cmap=cmap,
        add_colorbar=False,
        vmin=0.5,
        vmax=n + 0.5,
        **PLOT_KW,
    )
    qm.figure.colorbar(qm, ticks=levels).set_label("ZBP clusters")
    if fname is not None:
        qm.figure.savefig(fname, dpi=200)


@ds_requires(variables=("gapped", "clusters"), dims=("B", "V"))
def plot_stage_1(
    ds: xr.Dataset,
    n_clusters: int = sys.maxsize,
    *,
    fname: str | Path | None = None,
) -> None:
    """Plot clusters and gapped arrays.

    Parameters
    ----------
    ds
        Stage 1 `xarray.Dataset`.
    n_clusters
        Maximum number of clusters to plot.
    fname
        File name to save plot to. If None, no plot is saved.
    """
    gapless = 1.0 - ds.gapped
    fig, ax = plt.subplots(figsize=(5, 5))
    cmap = mpl.colors.ListedColormap(["tab:blue", "w"])
    pl = gapless.plot.pcolormesh(
        **PLOT_KW, vmin=-0.5, vmax=1.5, cmap=cmap, add_colorbar=False, ax=ax
    )
    cbar = fig.colorbar(pl, ticks=[0, 1], fraction=0.08, aspect=10, pad=0.04)
    cbar.set_label("")
    cbar.set_ticklabels(["Gapped", "Gapless"])
    plt.setp(
        cbar.ax.get_yticklabels(),
        rotation=90,
        ha="center",
        va="top",
        rotation_mode="anchor",
    )
    n_clusters = min(n_clusters, len(expand_clusters(ds.clusters)))
    for i, c in enumerate(
        ["tab:red", "tab:orange", "tab:olive", "tab:pink", "tab:green"][:n_clusters]
    ):
        cmap = mpl.colors.ListedColormap([c])
        cmap.set_bad(alpha=0)
        cl = expand_clusters(ds.clusters)[i]
        if cl.sum() == 0:
            print(f"Cluster {i} has no cluster.")
            continue
        cl.where(cl >= 1).plot.pcolormesh(
            ax=ax, cmap=cmap, **PLOT_KW, add_colorbar=False
        )
    ax.set_title("")
    plt.show()
    if fname is not None:
        fig.savefig(fname)


@ds_requires(variables=("gapped", "clusters"), dims=("B", "V"))
def plot_stage_1_gapped_clusters(
    ds: xr.Dataset,
    n_clusters: int = float("inf"),
    figsize: tuple[float, float] = (4, 8),
    only_plot: bool = False,
    with_colorbar: bool = True,
    *,
    fname: str | Path | None = None,
) -> mpl.axes.Axes | None:
    """Plot the gapped ZBP clusters.

    Parameters
    ----------
    ds
        Stage 1 `xarray.Dataset`.
    n_clusters
        Number of clusters to plot.
    figsize
        Figure size.
    only_plot
        Whether to only create the figure and axes without saving it.
    with_colorbar
        Whether to add a colorbar.
    fname
        File name to save plot to. If None, no plot is saved.

    Returns
    -------
    mpl.axes.Axes | None
        Figure axes or None.
    """
    gapless = 1.0 - ds.gapped

    fig, ax = plt.subplots(figsize=figsize)
    cmap = mpl.colors.ListedColormap(["tab:blue", "w"])
    pl = gapless.plot.pcolormesh(
        ax=ax, cmap=cmap, vmin=-0.5, vmax=1.5, add_colorbar=False, **PLOT_KW
    )
    plots = [pl]
    cmap = mpl.colors.ListedColormap(["black", "tab:orange"])
    cmap.set_bad(alpha=0)
    data_clusters = expand_clusters(ds.clusters)
    n_clusters = min(n_clusters, len(data_clusters))
    for i in range(n_clusters):
        cluster = data_clusters[i]
        cl_gapless = cluster * (1.0 - gapless)
        cl_gapped = cluster * gapless
        cl = 1.0 * cl_gapless + 2.0 * cl_gapped
        cl_pl = cl.where(cl >= 1).plot.pcolormesh(
            ax=ax,
            vmin=0.5,
            vmax=2.5,
            cmap=cmap,
            add_colorbar=False,
            **PLOT_KW,
        )
        plots.append(cl_pl)

    if with_colorbar:
        pos = ax.get_position()
        for im, where, loc, ticks in zip(
            plots,
            ["Outside", "Inside"],
            [pos.y0, pos.y1 - 0.35],
            [[0, 1], [1, 2]],
        ):
            cax = fig.add_axes([0.94, loc, 0.05, 0.35])
            cbar = fig.colorbar(im, cax=cax, ticks=ticks)
            cbar.set_label(f"{where} ZBP cluster")
            cbar.set_ticklabels(["Gapped", "Gapless"])
            plt.setp(
                cbar.ax.get_yticklabels(),
                rotation=90,
                ha="center",
                va="top",
                rotation_mode="anchor",
            )
    ax.set_title("")
    if only_plot:
        return ax
    plt.show()
    if fname is not None:
        fig.savefig(fname)


@ds_requires(variables=("gapped", "clusters"), dims=("B", "V"))
def plot_zoomed_clusters(
    ds: xr.Dataset,
    zoomin_ranges: list[dict[str, tuple[float, float]]],
    n_clusters: int = float("inf"),
    with_colorbar: bool = False,
    *,
    fname: str | None = None,
) -> None:
    """Plot zoomed clusters.

    Parameters
    ----------
    ds
        Stage 1 `xarray.Dataset`.
    zoomin_ranges
        List of dictionary with keys "B" and "V" and values (min, max) for each dimension.
    n_clusters
        Number of clusters to plot. By default all clusters are plotted.
    with_colorbar
        Whether to add a colorbar to the plot.
    fname
        File name to save plot to. If None, no plot is saved.
    """
    for i, zoomin_range in enumerate(zoomin_ranges, start=1):
        ax = plot_stage_1_gapped_clusters(
            ds,
            n_clusters,
            figsize=(4, 3),
            only_plot=True,
            with_colorbar=with_colorbar,
        )
        ax.set_xlim(zoomin_range["B"])
        ax.set_ylim(zoomin_range["V"])
        ax.set_title(f"Cluster #{i}")
        plt.show()
        if fname is not None:
            _check_formattable_string(fname)
            fig = ax.get_figure()
            fig.savefig(fname.format(i))


@ds_requires(variables=("L_2w_nl", "R_2w_nl"), dims=("B", "V", "cutter_pair_index"))
def plot_2w_interactive(ds: xr.Dataset, vmax_start: float) -> None:
    """Plot 2w at a given cutter pair index interactively.

    Parameters
    ----------
    ds
        Stage 1 `xarray.Dataset`.
    vmax_start
        Initial value for the slider.
    """

    def _plot(cutter_pair_index, vmax):
        plot.one.plot_2w_at(ds, cutter_pair_index, vmax)

    cutter_pair_index = np.arange(0, len(ds.L_3w.cutter_pair_index))
    widget = interactive(
        _plot,
        cutter_pair_index=SelectionSlider(
            options=cutter_pair_index, value=0, **_SLIDER_STYLE
        ),
        vmax=FloatSlider(
            min=min(ds.L_2w_nl.min(), ds.R_2w_nl.min()),
            max=max(ds.L_2w_nl.max(), ds.R_2w_nl.max()),
            value=vmax_start,
            **_SLIDER_STYLE,
        ),
    )
    display(widget)


@ds_requires(variables=("L_3w", "R_3w"), dims=("B", "V", "cutter_pair_index"))
def plot_3w_interactive(ds: xr.Dataset, lim_start: float) -> None:
    """Plot 3w at a given cutter pair index interactively.

    Parameters
    ----------
    ds
        Stage 1 `xarray.Dataset`.
    lim_start
        Initial value for the slider.
    """

    def _plot(cutter_pair_index, lim):
        plot.one.plot_3w_at(ds, cutter_pair_index, lim)

    cutter_pair_index = np.arange(0, len(ds.L_3w.cutter_pair_index))
    widget = interactive(
        _plot,
        cutter_pair_index=SelectionSlider(
            options=cutter_pair_index, value=0, **_SLIDER_STYLE
        ),
        lim=FloatSlider(
            description="Set upper and lower plotting limits",
            min=0,
            max=max(
                [abs(ds.L_3w.min()), abs(ds.R_3w.min()), ds.L_3w.max(), ds.R_3w.max()]
            ),
            value=lim_start,
            **_SLIDER_STYLE,
        ),
    )
    display(widget)


@ds_requires(variables=("L_2w_nl_ta", "R_2w_nl_ta"), dims=("B", "V"))
def plot_set_gapped_interactive(ds: xr.Dataset, th_2w_p: float) -> interactive:
    """Set the gap interatively and plot the result.

    Parameters
    ----------
    ds
        Stage 1 `xarray.Dataset`.
    th_2w_p
        The threshold for the ``L_2w_nl_ta`` and ``R_2w_nl_ta`` arrays.

    Returns
    -------
    interactive
        Interactive widget.
    """

    def _plot(th_2w_p):
        set_gapped(ds, th_2w_p)
        plot.one.plot_gapped(ds)

    display(
        interactive(
            _plot,
            th_2w_p=FloatSlider(
                min=0,
                max=1,
                step=0.01,
                value=ds.attrs.get("set_gapped.th_2w_p", 0.5),
                description="th_2w_p: fraction of cutter values, sets the gap",
                **_SLIDER_STYLE,
            ),
        )
    )


def _avoid_widgets_output_scroll():
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


def plot_analysis_interactive(ds: xr.Dataset, n_clusters: int = 11) -> None:
    """Perform entire Stage 1 analysis and plot the result.

    Parameters
    ----------
    ds
        Stage 1 `xarray.Dataset`.
    n_clusters
        Maximum number of clusters to plot.
    """
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
        thresholds = {
            "set_gapped": {"th_2w_p": th_2w_p},
            "set_3w_th": {"th_3w": th_3w},
            "set_3w_tat": {"th_3w_tat": th_3w_tat},
        }
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

    th_old = get_thresholds(ds.attrs)

    # Calculate limits of '3w' slider
    _max = np.maximum(ds.L_3w, ds.R_3w)
    min_3w = -_max.values.min()
    # Not terribly smart, pick mean of negative maxima as initial guess
    th_3w_start = np.nanmean(_max.where(_max > 0, drop=True))

    widget = interactive(
        _plot_and_analyze,
        th_2w_p=FloatSlider(
            min=0,
            max=1,
            step=0.01,
            value=th_old.get("th_2w_p", 0.5),
            description="th_2w_p: fraction of cutter values, sets the gap",
            **_SLIDER_STYLE,
        ),
        th_3w=FloatLogSlider(
            min=0,
            max=math.log10(min_3w),
            value=th_old.get("th_3w", math.log10(th_3w_start)),
            description="th_3w: abs value of 3ω conductance, sets the ZBPs",
            **_SLIDER_STYLE,
        ),
        th_3w_tat=FloatSlider(
            min=0,
            max=1,
            step=0.01,
            value=th_old.get("th_3w_tat", 0.5),
            description="th_3w_tat: fraction of cutter values, sets the ZBPs",
            **_SLIDER_STYLE,
        ),
        _plot_gapped=Checkbox(value=True, description="Show plot_gapped", **_STYLE),
        _plot_clusters=Checkbox(value=True, description="Show plot_clusters", **_STYLE),
        _plot_zbp=Checkbox(value=True, description="Show plot_zbp", **_STYLE),
        _plot_stage_1=Checkbox(value=True, description="Show plot_stage_1", **_STYLE),
        _plot_stage_1_gapped_clusters=Checkbox(
            value=True, description="Show plot_stage_1_gapped_clusters", **_STYLE
        ),
    )
    _avoid_widgets_output_scroll()
    display(widget)
