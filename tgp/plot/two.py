# Copyright (c) Microsoft Corporation. All rights reserved.

from __future__ import annotations

from pathlib import Path

from IPython.display import display
from ipywidgets import Checkbox, Output, SelectionSlider, interactive
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr

from tgp import plot
from tgp.two import antisymmetric_conductance_part

_STYLE = dict(layout={"width": "500px"}, style={"description_width": "initial"})
_SLIDER_STYLE = dict(_STYLE, continuous_update=False)


def plot_extracted_gap(
    ds_left,
    ds_right,
    cutter_pair_index,
    plunger_hline: float | None = None,
    fname: Path | str | None = None,
    show: bool = True,
    figsize: tuple = (8, 6),
) -> mpl.figure.Figure:
    """Plot the extracted gap."""
    fig, axs = plt.subplots(ncols=2, constrained_layout=True, figsize=figsize)
    for ds, side, ax in zip([ds_left, ds_right], ["left", "right"], axs):
        im_gap = (
            ds.gap.sel(cutter_pair_index=cutter_pair_index, method="nearest")
            .squeeze()
            .plot.imshow(
                ax=ax,
                add_colorbar=False,
                cmap="gist_heat_r",
                vmin=0,
            )
        )
        g = {"left": "G_{RL}", "right": "G_{LR}"}[side]
        ax.set_title(f"Gap from ${g}$")
        cbar = fig.colorbar(im_gap, ax=ax, location="top", extend="max")
        cbar.set_label(r"extracted gap $\Delta_\mathrm{ex}$")
        if plunger_hline is not None:
            ax.axhline(plunger_hline, ls="--", lw=1, c="cyan")
    if fname is not None:
        plt.savefig(fname, bbox_inches="tight", transparent=True)
    if show:
        plt.show()
    return fig


def plot_gap_extraction(
    ds_left,
    ds_right,
    cutter_pair_index,
    plunger_value,
    plot_filtered_conductance=True,
    norm: mpl.colors.Normalize | None = None,
    fname=None,
    show=True,
) -> mpl.figure.Figure:
    """Plot the gap extraction."""
    fig, axs = plt.subplots(ncols=2, figsize=(8, 6), sharex=True, sharey=True)
    cut_left = ds_left.sel(
        {"cutter_pair_index": cutter_pair_index, "V": plunger_value}, method="nearest"
    )
    if plot_filtered_conductance:
        g_rl = cut_left.filtered_antisym_g
    else:
        g_rl = cut_left.g_rl.squeeze().transpose("B", "left_bias")
        g_rl.data = antisymmetric_conductance_part(
            cond=g_rl.data.T, bias=cut_left["left_bias"].data
        ).T

    _ = (
        g_rl.squeeze()
        .transpose("left_bias", "B")
        .plot.pcolormesh(ax=axs[0], add_colorbar=False, cmap="RdBu_r", norm=norm)
    )
    cut_right = ds_right.sel(
        {"cutter_pair_index": cutter_pair_index, "V": plunger_value}, method="nearest"
    )
    if plot_filtered_conductance:
        g_lr = cut_right.filtered_antisym_g
    else:
        g_lr = cut_right.g_lr.squeeze().transpose("B", "right_bias")
        g_lr.data = antisymmetric_conductance_part(
            cond=g_lr.data.T, bias=cut_right["right_bias"].data
        ).T
    im_right = (
        g_lr.squeeze()
        .transpose("right_bias", "B")
        .plot.pcolormesh(ax=axs[1], add_colorbar=False, cmap="RdBu_r", norm=norm)
    )
    fig.colorbar(
        im_right,
        ax=axs.ravel().tolist(),
        label=r"$A(G_\mathrm{LR})$, $A(G_\mathrm{RL})$ [$e^2/h$]",
    )
    cut_left.gap.plot(ax=axs[0])
    cut_right.gap.plot(ax=axs[1])

    axs[0].set_title(r"$G_\mathrm{LR}$ and $\Delta_\mathrm{ex}$ left")
    axs[1].set_title(r"$G_\mathrm{RL}$ and $\Delta_\mathrm{ex}$ right")
    axs[0].set_ylabel(r"Left bias [$\mu$V]")
    axs[1].set_ylabel(r"Right bias [$\mu$V]")
    which = "Filtered" if plot_filtered_conductance else "Unfiltered"
    fig.suptitle(f"{which} antisymmetric part of $G$", fontsize=16)
    if show:
        plt.show()

    if fname is not None:
        plt.savefig(fname, bbox_inches="tight")
    return fig


def plot_gap_extraction_interactive(ds_left, ds_right) -> None:
    """Plot the gap extraction interactively."""
    out = Output()

    def _plot(
        cutter_pair_index, V, plot_filtered_conductance, with_2d_plot
    ) -> mpl.figure.Figure:
        with out:
            out.clear_output(wait=True)
            plot.two.plot_gap_extraction(
                ds_left,
                ds_right,
                cutter_pair_index,
                V,
                plot_filtered_conductance,
            )
            if with_2d_plot:
                plot.two.plot_extracted_gap(
                    ds_left, ds_right, cutter_pair_index, plunger_hline=V
                )
            plt.show()

    widget = interactive(
        _plot,
        cutter_pair_index=SelectionSlider(
            options=ds_left.cutter_pair_index.values, **_SLIDER_STYLE
        ),
        V=SelectionSlider(options=ds_left.V.values, **_SLIDER_STYLE),
        plot_filtered_conductance=Checkbox(
            value=True,
            description="Plot the filtered conductance (used in gap extraction)",
            **_STYLE,
        ),
        with_2d_plot=Checkbox(
            value=True,
            description="Include 2D plot of extracted gap (plot_extracted_gap)",
            **_STYLE,
        ),
    )
    return display(widget, out)


def plot_data(
    ds_left: xr.Dataset,
    ds_right: xr.Dataset,
    V: float,
    cutter_pair_index: int,
    with_hline: bool = True,
    unit: str = "",
    norm: mpl.colors.Normalize | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> mpl.figure.Figure:
    """Plot the raw or processed conductance data at a fixed ``V`` and ``cutter_pair_index``."""
    left_sel = ds_left.sel({"V": V, "cutter_pair_index": cutter_pair_index}).squeeze()
    right_sel = ds_right.sel({"V": V, "cutter_pair_index": cutter_pair_index}).squeeze()
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=figsize)
    left_sel["g_ll"].squeeze().transpose("left_bias", "B").real.plot.pcolormesh(
        ax=axs[0, 0],
        add_colorbar=True,
        cbar_kwargs=dict(label=unit),
        cmap="viridis",
        vmin=0,
    )
    axs[0, 0].set_title("$G_{ll}$")
    right_sel["g_rr"].squeeze().transpose("right_bias", "B").real.plot.pcolormesh(
        ax=axs[0, 1],
        add_colorbar=True,
        cbar_kwargs=dict(label=unit),
        cmap="viridis",
        vmin=0,
    )
    axs[0, 1].set_title("$G_{rr}$")

    left_sel["g_rl"].squeeze().transpose("left_bias", "B").real.plot.pcolormesh(
        ax=axs[1, 0], norm=norm, add_colorbar=True, cbar_kwargs=dict(label=unit)
    )
    axs[1, 0].set_title("$G_{rl}$")

    right_sel["g_lr"].squeeze().transpose("right_bias", "B").real.plot.pcolormesh(
        ax=axs[1, 1], norm=norm, add_colorbar=True, cbar_kwargs=dict(label=unit)
    )
    axs[1, 1].set_title("$G_{lr}$")

    if with_hline:
        for ax in axs.flatten():
            ax.axhline(0, color="black", linestyle="--")

    plt.tight_layout()
    plt.show()
    return fig


def plot_data_interactive(
    ds_left: xr.Dataset,
    ds_right: xr.Dataset,
    norm: mpl.colors.Normalize | None = None,
    unit: str | None = None,
    figsize=(10, 6),
) -> interactive:
    """Plot the raw or processed conductance data interactively."""
    ps = ds_left.coords["V"].values
    cutters = ds_left.coords["cutter_pair_index"].values
    unit = unit or ""

    def _plot(V, cutter_pair_index, with_hline):
        return plot_data(
            ds_left,
            ds_right,
            V,
            cutter_pair_index,
            with_hline=with_hline,
            unit=unit,
            norm=norm,
            figsize=figsize,
        )

    return interactive(
        _plot,
        V=SelectionSlider(options=list(ps), **_SLIDER_STYLE),
        cutter_pair_index=SelectionSlider(options=list(cutters), **_SLIDER_STYLE),
        with_hline=Checkbox(
            value=True, description="Include horizontal line at 0 bias", **_STYLE
        ),
    )
