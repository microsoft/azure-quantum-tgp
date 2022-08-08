# Copyright (c) Microsoft Corporation. All rights reserved.

from __future__ import annotations

from pathlib import Path
from typing import Any

from IPython.display import display
from ipywidgets import (
    Checkbox,
    Output,
    SelectionRangeSlider,
    SelectionSlider,
    interactive,
)
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import tgp
from tgp import plot
from tgp.common import cluster_info, expand_clusters

_STYLE = dict(layout={"width": "500px"}, style={"description_width": "initial"})
_SLIDER_STYLE = dict(_STYLE, continuous_update=False)


def _plot_with_discrete_cbar(
    ds, fig, ax, cmap_name="Greys", cbar_pad=0, label="", location="top", labelpad=7
):
    values = np.sort(np.unique(ds.data))
    ticks = np.linspace(0, 1, len(values))
    cmap = mpl.colors.ListedColormap(plt.cm.get_cmap(cmap_name)(ticks))
    im = ds.plot.imshow(ax=ax, add_colorbar=False, cmap=cmap)
    cbar = fig.colorbar(
        im,
        ax=ax,
        location=location,
        ticks=np.linspace(values.min(), values.max(), 2 * len(values) + 1)[1::2],
        pad=cbar_pad,
    )
    cbar.set_label(label, labelpad=labelpad)
    ticklabels = [f"{x:.2f}" for x in values]
    if location == "top":
        cbar.ax.set_xticklabels(ticklabels)
    else:
        cbar.ax.set_yticklabels(ticklabels)
    return im, cbar


def _get_region_of_interest_1(
    zbp_ds,
    pct_box: int = 20,
    name_cluster: str = "gapped_zbp_cluster",
    name_zbp_cluster_number: str = "zbp_cluster_number",
    zbp_cluster_number: int = 1,
) -> tuple[dict[str, Any], xr.DataArray]:
    clusters = expand_clusters(zbp_ds[name_cluster], dim=name_zbp_cluster_number)
    cluster = clusters.sel(**{name_zbp_cluster_number: zbp_cluster_number})
    info = cluster_info(cluster, pct_box=pct_box)
    return (info, cluster)


def plot_clusters_zbp(
    zbp_ds,
    pct_box: float = 5.0,
    zbp_cluster_number: int = 1,
    fig: mpl.figure.Figure = None,
    ax: mpl.axes.Axes = None,
    fname: Path | str | None = None,
) -> xr.DataArray | None:
    """Plot the zbp clusters."""
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
    clen = int(zbp_ds.gapped_zbp_cluster.max())
    colors = ["white"]
    ticklabels = ["bg"]
    if -1 in zbp_ds["zbp_cluster_number"]:
        colors.append("black")
        ticklabels.append("no")
    colors += list(plt.cm.get_cmap("Set1").colors) + list(
        plt.cm.get_cmap("Set2").colors
    )
    colors = colors[0 : (clen + 1)]

    cmap = mpl.colors.ListedColormap(colors)
    clusters = zbp_ds.gapped_zbp_cluster.T
    im2 = clusters.plot.imshow(
        ax=ax,
        cmap=cmap,
        add_colorbar=False,
    )
    ax.set_title("")
    cluster_ticks = np.array(range(-1, clen, 1))
    cbar = fig.colorbar(
        im2,
        ax=ax,
        location="top",
        ticks=clen * (cluster_ticks + 1.5) / (clen + 1),
        pad=0,
    )
    cbar.set_label("Cluster ID")
    ticklabels.extend(range(1, len(zbp_ds["zbp_cluster_number"]) + 1))
    cbar.ax.set_xticklabels(ticklabels[: len(cluster_ticks)])

    # Create a Rectangle patch on the selected cluster
    try:
        info, selected_cluster = _get_region_of_interest_1(
            zbp_ds, pct_box, zbp_cluster_number=zbp_cluster_number
        )
    except (KeyError, IndexError) as e:
        selected_cluster = None
        print(f"Didn't find any clusters: {e}")
    else:
        x1, y1, x2, y2 = info.bounding_box
        ax.add_patch(
            Rectangle(
                (x1, y1),
                (x2 - x1),
                (y2 - y1),
                linewidth=2,
                edgecolor=colors[zbp_cluster_number],
                facecolor="none",
            )
        )

    if fname is not None:
        plt.savefig(fname, transparent=True)

    return selected_cluster


def _set_zbp_attrs(zbp_ds, plunger_name, field_name):
    zbp_ds.left.attrs["long_name"] = "Probability of ZBP"
    zbp_ds.right.attrs["long_name"] = "Probability of ZBP"

    zbp_ds[field_name].attrs["units"] = "T"
    zbp_ds[field_name].attrs["long_name"] = "$B$"
    zbp_ds[plunger_name].attrs["units"] = "V"
    zbp_ds[plunger_name].attrs["long_name"] = r"$V_\mathrm{gate}$"


def plot_joined_probability(
    zbp_ds,
    plunger_name: str = "V",
    field_name: str = "B",
    zbp_name: str = "zbp",
    figsize: tuple[float, float] = (5.1, 3.15),
    title=None,
    fname=None,
    show=True,
) -> mpl.figure.Figure:
    """Plot the joined probability of ZBP."""
    zbp_threshold = zbp_ds.attrs.get("probability_threshold", np.nan)

    _set_zbp_attrs(zbp_ds, plunger_name, field_name)
    fig, ax = plt.subplots(ncols=1, figsize=figsize)
    cmap = plt.get_cmap("viridis", 2)
    im = zbp_ds[zbp_name].plot.imshow(
        x=field_name, y=plunger_name, ax=ax, add_colorbar=False, cmap=cmap
    )
    ax.set_title(title or "Joint probability")
    im.set_clim(0, 1)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(
        [
            rf"$P(\mathrm{{ZBP}})<{zbp_threshold:.1f}$",
            rf"$P(\mathrm{{ZBP}}) \geq {zbp_threshold:.1f}$",
        ]
    )
    if fname is not None:
        plt.savefig(fname, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_probability_and_clusters(
    zbp_ds,
    zbp_cluster_number: int | None = 1,
    pct_box: float | int = 10,
    fname: str | None = None,
    with_clusters: bool = True,
    with_gap: bool = True,
    figsize: tuple[float, float] = (8, 6),
) -> mpl.figure.Figure:
    """Plot probability of ZBP and clusters."""
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = GridSpec(2, 3, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0, sharex=ax0)
    ax2 = fig.add_subplot(gs[0, 2], sharey=ax0, sharex=ax0)
    axs = [ax0, ax1, ax2]
    if with_gap:
        ax_gap = fig.add_subplot(gs[1, 0], sharey=ax0, sharex=ax0)
        axs.append(ax_gap)
        ax_gap_zbp = fig.add_subplot(gs[1, 1], sharey=ax0, sharex=ax0)
        axs.append(ax_gap_zbp)

    if with_clusters:
        ax_cluster = fig.add_subplot(gs[1, 2])
        axs.append(ax_cluster)

    # ZBP probabilities
    for key, which, ax in zip(
        ("left", "right", "zbp"),
        ("Left", "Right", "Joint"),
        [ax0, ax1, ax2],
    ):
        _plot_with_discrete_cbar(
            zbp_ds[key].squeeze().transpose("V", "B"),
            fig,
            ax,
            "viridis",
            0,
            f"{which} ZBP Probability",
        )

    if with_gap:
        gap_boolean = zbp_ds["gap_boolean"]
        cbar = _plot_binary_with_cbar(gap_boolean, fig, ax_gap, "Gap boolean", "top")
        ticks = []
        if False in gap_boolean.values:
            ticks.append("gpls")
        if True in gap_boolean.values:
            ticks.append("gap")
        cbar.ax.set_xticklabels(ticks)
        cbar = _plot_binary_with_cbar(
            zbp_ds["gapped_zbp"], fig, ax_gap_zbp, "Gapped ZBP", "top"
        )
        cbar.ax.set_xticklabels(["gpls", "gap ZBP"][: len(cbar.ax.get_xticks())])

    # Clusters
    if with_clusters and zbp_cluster_number is not None:
        plot_clusters_zbp(
            zbp_ds,
            pct_box,
            fig=fig,
            ax=ax_cluster,
            zbp_cluster_number=zbp_cluster_number,
        )

    for i, ax in enumerate(axs):
        ax.set_title("")
        label = "abcdefg"[i]
        ax.text(
            0.22,
            0.95,
            rf"$\mathrm{{({label})}}$",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=16,
            color="grey",
        )
    if fname is not None:
        plt.savefig(fname, bbox_inches="tight", transparent=True)

    plt.show()
    return fig


def _color_spline(ax, c, ls, lw=2, sides=["bottom", "top", "right", "left"]):
    for key in sides:
        ax.spines[key].set_color(c)
        ax.spines[key].set_linestyle(ls)
        ax.spines[key].set_linewidth(lw)
    ax.tick_params(axis="x", colors=c)
    ax.tick_params(axis="y", colors=c)
    for tick in ax.get_yticklabels():
        tick.set_color("k")
    for tick in ax.get_xticklabels():
        tick.set_color("k")
    ax.xaxis.set_tick_params(length=4, width=1, color="k")
    ax.yaxis.set_tick_params(length=4, width=1, color="k")


def plot_region_of_interest_2(
    zbp_ds,
    zbp_cluster_number: int,
    cutter_pair_index: float | None = None,
    fname: Path | str | None = None,
    show: bool = True,
    boundary=False,
    padding=1,
    gap_units=" [meV]",
) -> mpl.figure.Figure:
    """Plot the region of interest."""
    fig, axs = plt.subplots(ncols=2, constrained_layout=True, figsize=(8, 6))
    sel = {}
    if "cutter_pair_index" in zbp_ds["gap"].dims:
        if cutter_pair_index is None:
            raise ValueError("Must specify a cutter value.")
        sel["cutter_pair_index"] = cutter_pair_index
    clusters = expand_clusters(zbp_ds.gapped_zbp_cluster, dim="zbp_cluster_number")
    cluster = clusters.sel(**sel, zbp_cluster_number=zbp_cluster_number)
    gap_bool = zbp_ds.gap_boolean.sel(**sel).squeeze()
    zbp_bool = zbp_ds.zbp.squeeze()

    ds = (2 / 4) * gap_bool + (1 / 4) * zbp_bool + (1 / 4) * cluster
    ds = ds.squeeze().transpose("V", "B")
    values = np.sort(np.unique(ds.data))
    cmap = mpl.colors.ListedColormap(
        list(plt.cm.Greys(np.linspace(0, 1, len(values) - 1))) + ["red"]
    )
    im = ds.plot.imshow(
        ax=axs[0],
        add_colorbar=False,
        cmap=cmap,
    )
    axs[0].set_title("")
    cbar = fig.colorbar(
        im,
        ax=axs[0],
        location="top",
        ticks=np.linspace(values.min(), values.max(), 2 * len(values) + 1)[1::2],
    )
    cbar.set_label("Gap and ZBP regions")
    ticks_options = {
        0: "gpls",
        0.25: "gpls ZBP",
        0.5: "gap",
        0.75: "gap ZBP",
        1: "ROI$_2$",
    }
    ticklabels = [ticks_options[x] for x in values]
    cbar.ax.set_xticklabels(ticklabels, fontsize=10)
    if boundary:
        for _zbp_cluster_number in zbp_ds.zbp_cluster_number[1:]:
            (
                zbp_ds["boundary_array"]
                .sel(**sel)
                .sel(zbp_cluster_number=_zbp_cluster_number)
                .squeeze()
                .transpose("V", "B")
                .plot.imshow(
                    ax=axs[0],
                    add_colorbar=False,
                    cmap="ocean",
                )
            )
    axs[0].set_title("")

    idx_B = np.where(cluster.sum("B") > 0)[0]
    first, last = (idx_B[0], idx_B[-1]) if len(idx_B) >= 2 else (idx_B[0], idx_B[0] + 1)
    first = max(0, first - padding)
    last = min(len(cluster.V) - 1, last + padding)
    dashed = (
        max(cluster.V.min(), cluster.V[first]),
        min(cluster.V.max(), cluster.V[last]),
    )
    first_B, *_, last_B = np.where(cluster.sum("V") > 0)[0]

    B_range = (
        max(cluster.B.min(), cluster.B[max(first_B - padding, 0)]),
        min(
            cluster.B.max(),
            cluster.B[min(last_B + padding, len(cluster.B) - padding)],
        ),
    )
    for y in dashed:
        axs[0].axhline(y, ls="--", lw=1, c="k")

    cluster_gap = (zbp_ds.gap * cluster).sel(**sel).squeeze()

    cluster_gap.transpose("V", "B").plot.imshow(
        ax=axs[1],
        add_colorbar=True,
        cmap="gist_heat_r",
        cbar_kwargs={
            "label": r"Gap inside ROI$_2$" + gap_units,
            "location": "top",
        },
    )

    median_gap = float(
        zbp_ds.median_gap.sel(zbp_cluster_number=zbp_cluster_number).squeeze()
    )
    pct_boundary = 100 * float(
        zbp_ds.percentage_boundary.sel(zbp_cluster_number=zbp_cluster_number).squeeze()
    )

    axs[1].text(
        0.1,
        0.15,
        r"$\bar{\Delta}_\mathrm{ex}^{(j)}="
        + rf"{median_gap:.4f}$"
        + gap_units
        + "\n"
        + rf"$\mathrm{{gapless boundary}}={pct_boundary:.0f}$ %",
        transform=axs[1].transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=13,
        color="k",
    )

    axs[1].yaxis.set_major_locator(plt.MaxNLocator(3))

    axs[1].set_ylim(*dashed)
    axs[1].set_xlim(*B_range)
    axs[1].set_title("")
    _color_spline(axs[1], "k", "--", 1, ["bottom", "top"])

    for i, ax in enumerate(axs):
        label = "abcd"[i]
        color = {0: "black", 1: "black", 2: "black", 3: "black"}[i]
        ax.text(
            -0.1,
            1.1,
            rf"$\mathrm{{({label})}}$",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=14,
            color=color,
        )

    if fname is not None:
        plt.savefig(fname, bbox_inches="tight", transparent=True)
    if show:
        plt.show()
    return fig


def _plot_binary_with_cbar(ds, fig, ax, title, location="right"):
    _, cbar = _plot_with_discrete_cbar(
        ds,
        fig,
        ax,
        "Greys",
        cbar_pad=0,
        label="",
        location=location,
    )
    ax.set_title(title)
    return cbar


def plot_gapped_zbp(zbp_ds, cutter_pair_index):
    """Plot gapped ZBPs."""
    figsize = (8, 8)
    fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    sel = {}
    if "cutter_pair_index" in zbp_ds["gap"].dims:
        sel["cutter_pair_index"] = cutter_pair_index
    zbp_ds["gap"].sel(**sel).plot.imshow(ax=axs[0, 0], vmin=0)
    axs[0, 0].set_title(r"Extracted gap $\Delta_{ex}$")

    gap_boolean = zbp_ds["gap_boolean"].sel(**sel)
    cbar = _plot_binary_with_cbar(gap_boolean, fig, axs[0, 1], "Gap boolean")
    ticks = []
    if False in gap_boolean.values:
        ticks.append("Gapless")
    if True in gap_boolean.values:
        ticks.append("Gapped")
    cbar.ax.set_yticklabels(ticks)

    cbar = _plot_binary_with_cbar(
        zbp_ds["zbp"].transpose(), fig, axs[1, 0], "ZPB boolean"
    )
    cbar.ax.set_yticklabels(["no ZBP", "ZBP"][: len(cbar.ax.get_yticks())])
    cbar = _plot_binary_with_cbar(
        zbp_ds["gapped_zbp"].sel(**sel), fig, axs[1, 1], "Gapped ZBP"
    )
    cbar.ax.set_yticklabels(["Gapless", "Gapped ZBP"][: len(cbar.ax.get_yticks())])
    return fig


def plot_left_right_zbp_probability(
    zbp_ds,
    plunger_name: str = "V",
    field_name: str = "B",
    figsize: tuple[float, float] = (5.1, 3.15),
    discrete=False,
    fname=None,
) -> mpl.figure.Figure:
    """Plot left and right ZBP probability."""
    _set_zbp_attrs(zbp_ds, plunger_name, field_name)
    info = [
        ("left", "Left side"),
        ("right", "Right side"),
    ]

    fig, axs = plt.subplots(ncols=2, figsize=figsize, sharey=True)
    cmap = plt.get_cmap("viridis", 2 if discrete else None)
    for i, ax in enumerate(axs):
        key, title = info[i]
        label = "abcd"[i]
        ax.text(
            x=0.02,
            y=0.97,
            s=f"({label})",
            color="w",
            transform=ax.transAxes,
            fontsize=20,
            verticalalignment="top",
            horizontalalignment="left",
        )

        im = zbp_ds[key].plot.imshow(
            x=field_name, y=plunger_name, ax=ax, add_colorbar=False, cmap=cmap
        )
        ax.set_title(title)
        im.set_clim(0, 1)

    if discrete:
        zbp_threshold = zbp_ds.attrs.get("probability_threshold", np.nan)
        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
        cbar.ax.set_yticklabels(
            [
                rf"$P(\mathrm{{ZBP}})<{zbp_threshold:.1f}$",
                rf"$P(\mathrm{{ZBP}}) \geq {zbp_threshold:.1f}$",
            ]
        )
    else:
        cbar = fig.colorbar(im, ax=axs[1])
        cbar.set_label("Probability of ZBP")
    if fname is not None:
        plt.savefig(fname, bbox_inches="tight")
    plt.show()


def plot_gapped_zbp_interactive(zbp_ds) -> None:
    """Plot gapped ZBPs interactively."""
    out = Output()

    def _plot(cutter_pair_index, thresholds) -> mpl.figure.Figure:
        threshold_low, threshold_high = thresholds
        with out:
            tgp.two.set_gap_threshold(zbp_ds, threshold_low, threshold_high)
            out.clear_output(wait=True)
            plot.zbp.plot_gapped_zbp(zbp_ds, cutter_pair_index)
            plt.show()

    gaps = sorted(np.unique(zbp_ds["gap"].values.flat))
    widget = interactive(
        _plot,
        cutter_pair_index=SelectionSlider(
            options=np.atleast_1d(zbp_ds.cutter_pair_index.values.tolist()),
            **_SLIDER_STYLE,
        ),
        thresholds=SelectionRangeSlider(
            description="Lower and upper gap thresholds",
            options=[(f"{g:.4f}", g) for g in gaps] + [("None", None)],
            value=(gaps[1], None),
            **_SLIDER_STYLE,
        ),
    )
    display(widget, out)


def plot_results_interactive(zbp_ds: xr.Dataset) -> None:
    """Plot ZBP analysis results interactively."""

    def _plot(
        cutter_pair_index,
        zbp_cluster_number,
        _plot_probability_and_clusters,
        _plot_region_of_interest_2,
    ):
        zbp_ds_sel = (
            zbp_ds.sel(cutter_pair_index=cutter_pair_index)
            if "cutter_pair_index" in zbp_ds.dims
            else zbp_ds
        )
        if _plot_probability_and_clusters:
            plot_probability_and_clusters(zbp_ds_sel, zbp_cluster_number)
        if _plot_region_of_interest_2:
            if (
                zbp_cluster_number is not None
                and zbp_cluster_number <= zbp_ds_sel.gapped_zbp_cluster.max()
            ):
                plot_region_of_interest_2(
                    zbp_ds_sel, zbp_cluster_number, cutter_pair_index
                )
            else:
                print(
                    f"zbp_cluster_number={zbp_cluster_number} doesn't exist for"
                    f" cutter_pair_index={cutter_pair_index}"
                )

    zbp_cluster_number = (
        zbp_ds.zbp_cluster_number.values
        if len(zbp_ds.zbp_cluster_number.values) > 0
        else [None]
    )

    widget = interactive(
        _plot,
        cutter_pair_index=SelectionSlider(
            options=zbp_ds.cutter_pair_index.values, **_SLIDER_STYLE
        ),
        zbp_cluster_number=SelectionSlider(options=zbp_cluster_number, **_SLIDER_STYLE),
        _plot_probability_and_clusters=Checkbox(
            value=True, description="Show plot_probability_and_clusters", **_STYLE
        ),
        _plot_region_of_interest_2=Checkbox(
            value=True, description="Show plot_region_of_interest_2", **_STYLE
        ),
    )
    plot.one._avoid_widgets_output_scroll()
    display(widget)
