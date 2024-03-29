# Copyright (c) Microsoft Corporation. All rights reserved.

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import warnings

from PIL import Image
import matplotlib as mpl
import matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import xarray as xr

import tgp
from tgp.frequency_correction import correct_frequencies
import tgp.plot.paper

SAVE_FIGURES = True
SAVE_FOLDER = Path(".")


def set_saving_figures(save: bool = True, folder: str = "."):
    global SAVE_FIGURES
    SAVE_FIGURES = save
    global SAVE_FOLDER
    SAVE_FOLDER = Path(folder)
    SAVE_FOLDER.mkdir(exist_ok=True, parents=True)


def maybe_save_fig(fname: str | Path | None):
    """Save the figure if `set_saving_figures` is called with True."""
    if SAVE_FIGURES and fname is not None:
        plt.savefig(SAVE_FOLDER / fname, **tgp.plot.paper.SAVE_KWARGS)


def load(fname: str) -> xr.Dataset:
    ds = xr.load_dataset(
        fname,
        format="NETCDF4",
        engine="h5netcdf",
        invalid_netcdf=True,
    )
    return tgp.prepare.prepare(ds).fillna(0)


def correct_two(
    ds_left: xr.Dataset,
    ds_right: xr.Dataset,
    lockin_left: str,
    lockin_right: str,
    fridge_parameters: dict | str,
    drop_indices: list[int] | tuple[list[int], list[int]] | None = None,
    max_bias_index: int | tuple[int, int] | None = None,
    norm: float = 1.0,
    phase_shift_left: float = 0.0,
    phase_shift_right: float = 0.0,
) -> tuple[tuple[xr.Dataset, xr.Dataset], tuple[xr.Dataset, xr.Dataset]]:
    ds_left, ds_right = correct_frequencies(
        ds_left,
        ds_right,
        lockin_left,
        lockin_right,
        fridge_parameters,
        phase_shift_left,
        phase_shift_right,
    )

    ds_c_left, ds_c_right = tgp.prepare.correct_bias(
        ds_left,
        ds_right,
        drop_indices,
        max_bias_index,
        norm,
        method="manual",
    )
    return (ds_left, ds_right), (ds_c_left, ds_c_right)


def analyze_two(
    ds_left: xr.Dataset,
    ds_right: xr.Dataset,
    min_cluster_size: int = 7,
    zbp_average_over_cutter: bool = True,
    zbp_probability_threshold: float = 0.6,
    gap_threshold_high: float = 70e-3,
    gap_threshold_factor: float = 0.05,
    cluster_gap_threshold: float | None = None,
    cluster_volume_threshold: float | None = None,
    cluster_percentage_boundary_threshold: float | None = None,
):
    ds_left, ds_right = tgp.two.extract_gap(
        ds_left, ds_right, gap_threshold_factor=gap_threshold_factor
    )
    zbp_ds = tgp.two.zbp_dataset_derivative(
        ds_left,
        ds_right,
        average_over_cutter=zbp_average_over_cutter,
        zbp_probability_threshold=zbp_probability_threshold,
    )

    tgp.two.set_zbp_gap(zbp_ds, ds_left, ds_right)
    tgp.two.set_gap_threshold(zbp_ds, threshold_high=gap_threshold_high)

    zbp_ds = tgp.two.cluster_and_score(
        zbp_ds,
        min_cluster_size=min_cluster_size,
        cluster_gap_threshold=cluster_gap_threshold,
        cluster_volume_threshold=cluster_volume_threshold,
        cluster_percentage_boundary_threshold=cluster_percentage_boundary_threshold,
    )
    return SimpleNamespace(
        zbp_ds=zbp_ds,
        ds_left=ds_left,
        ds_right=ds_right,
    )


def plot_grad(ax, x, y, **kwargs):
    t = np.linspace(0, 1, x.shape[0])
    points = np.array([x, y]).transpose().reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = mpl.collections.LineCollection(segs, **kwargs)
    lc.set_array(t)
    ax.add_collection(lc)


def h(ks, m=1, mu=0, alpha=2, vx=1):
    s0 = np.array([[1, 0], [0, 1]])
    s1 = np.array([[0, 1], [1, 0]])
    s2 = np.array([[0, -1j], [1j, 0]])
    e = np.array(
        [
            np.linalg.eigh((k**2 / (2 * m) - mu) * s0 + alpha * k * s2 + vx * s1)[0]
            for k in ks
        ]
    )
    return e.T


def plot_1d_phase_diagram(fname) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))

    k = np.arange(-5.2, 5.201, 0.05)
    e1, e2 = h(k)

    colors = [[0.8359375, 0.15234375, 0.15625], [0.12109375, 0.46484375, 0.703125]]

    cmap = mpl.colors.LinearSegmentedColormap.from_list("Custom", colors, N=500)

    kwargs = dict(lw=2)
    axs[0].plot([-10, -0.25], [0, 0], ls="--", lw=1, c="#777777")
    axs[0].plot([1.5, 10], [0, 0], ls="--", lw=1, c="#777777")

    mc = np.abs(k) <= 1.0001
    ml = k <= -0.99
    mr = k >= 0.99
    axs[0].plot(k[ml], e1[ml], **kwargs, c=colors[0])
    axs[0].plot(k[mr], e1[mr], **kwargs, c=colors[-1])
    plot_grad(axs[0], k[mc], e1[mc], **kwargs, cmap=cmap)

    axs[0].plot(k[ml], e2[ml], **kwargs, c=colors[-1])
    axs[0].plot(k[mr], e2[mr], **kwargs, c=colors[0])
    plot_grad(axs[0], k[mc], e2[mc], **kwargs, cmap=cmap.reversed())

    axs[0].set_xlim(-5.2, 5.2)
    axs[0].set_ylim(-3, 5)
    axs[0].set_xlabel(r"$k$")
    axs[0].set_ylabel(r"$E$")
    axs[0].set_xticks([0])
    axs[0].set_yticks([0])

    axs[0].annotate(
        text="",
        xy=(0, -1.0),
        xytext=(0, 1),
        arrowprops=dict(arrowstyle="<|-|>", fc="k", ec="k"),
    )
    axs[0].text(0.25, 0, "$V_x$", c="k", ha="left", va="center")

    mu = np.arange(-5, 5.001, 0.01)
    mug, Vxg = np.meshgrid(mu, np.arange(-2, 6.001, 0.01))

    cmap = mpl.colors.ListedColormap(["tab:blue", "tab:red"])

    c = 1.0 * (-1 + 2 * (Vxg > np.sqrt(1.0 + mug**2)))

    pcm = axs[1].pcolormesh(
        Vxg.T,
        mug.T,
        c.T,
        shading="auto",
        vmin=-1,
        vmax=1,
        cmap=cmap,
        linewidth=0,
        rasterized=True,
    )
    pcm.set_edgecolor("face")
    axs[1].plot(np.sqrt(1 + mu**2), mu, lw=0.5, c="k")
    axs[1].set_xlim(0, 4)
    axs[1].set_ylim(-3, 3)
    axs[1].set_xlabel(r"$V_x / \Delta_\mathrm{ind}$")
    axs[1].set_ylabel(r"$\mu / \Delta_\mathrm{ind}$")

    for i, ax in enumerate(axs):
        tgp.plot.paper.add_subfig_label(
            ax, "(" + "ab"[i] + ")", width=0.14, height=0.14
        )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.38)
    maybe_save_fig(fname)
    plt.show()


def plot_Hall_bar(ds_fname: str, plot_fname: str = "Hall_bar_mobility.pdf") -> None:
    ds = xr.load_dataset(ds_fname)
    fig, ax = plt.subplots(ncols=1, figsize=(8.5, 6))

    ax0 = inset_axes(
        ax,
        width=2.1,
        height=1.6,
        bbox_to_anchor=(0.13, 0.96),
        bbox_transform=ax.transAxes,
        loc="upper left",
        borderpad=0,
    )
    ax1 = inset_axes(
        ax,
        width=2.1,
        height=1.6,
        bbox_to_anchor=(0.66, 0.96),
        bbox_transform=ax.transAxes,
        loc="upper left",
        borderpad=0,
    )

    ax0.plot(ds["gate_voltage"], ds["density"], ".-", c="tab:blue", label="experiment")
    ax0.plot(
        ds["gate_voltage"], ds["density_fit"], ".-", color="C1", label="used in fit"
    )
    ax0.plot(
        ds["gate_voltage"][1:25],
        ds["density_pred"][1:25],
        color="black",
        lw=1,
        label="prediction",
    )

    ax1.plot(
        ds["gate_voltage"], 10 * ds["mobility"], ".-", color="C0", label="experiment"
    )
    ax1.plot(
        ds["gate_voltage"],
        10 * ds["mobility_fit"],
        ".-",
        color="C1",
        label="used in fit",
    )
    ax1.plot(
        ds["gate_voltage"][1:25],
        10 * ds["mobility_pred"][1:25],
        color="black",
        lw=1,
        label="prediction",
    )

    ax.plot(ds["density"], 10 * ds["mobility"], "o-", color="C0", label="experiment")
    ax.plot(
        ds["density_fit"],
        10 * ds["mobility_fit"],
        "o-",
        color="C1",
        label="used in fit",
    )
    ax.plot(
        ds["density_pred"][1:25],
        10 * ds["mobility_pred"][1:25],
        color="black",
        label="prediction",
    )

    ax0.set_ylabel(r"$n_\mathrm{e}$ [10$^{12}$ cm$^{-2}$]")
    ax1.set_ylabel(r"$\mu$ [10$^{3}$ cm$^{2}/$Vs]")
    ax.set_xlabel(r"$n_\mathrm{e}$ [10$^{12}$ cm$^{-2}$]")
    ax.set_ylabel(r"$\mu$ [10$^{3}$ cm$^{2}/$Vs]")
    for a in [ax0, ax1]:
        a.set_xlim(-0.6, None)
        a.set_xlabel(r"$V_\mathrm{g}$ [V]")

    ax0.set_ylim(0.0, 1.5)
    ax1.set_ylim(0, 70)
    ax.set_xlim(0.0, 1.5)
    ax.set_ylim(0, 120)

    maybe_save_fig(plot_fname)
    plt.show()


def plot_clean_gap(
    ds,
    gap_max=60,
    field_min=None,
    field_max=3.85,
    plunger_min=-1.4,
    plunger_max=-0.65,
    label: str | None = None,
    fname: str | Path | None = None,
):
    ds.gap["f"].attrs["long_name"] = "$B$"
    ds.gap["f"].attrs["units"] = "T"

    ds.gap["p"].attrs["long_name"] = r"$V_\mathrm{p}$"
    ds.gap["p"].attrs["units"] = "V"

    ds.gap.attrs["long_name"] = r"$\mathcal{Q}\Delta$"
    ds.gap.attrs["units"] = r"$\mu$eV"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    gap = 1e3 * ds.gap
    im = gap.plot.pcolormesh(
        ax=ax,
        x="f",
        y="p",
        vmin=-gap_max,
        vmax=gap_max,
        cmap="RdBu",
        rasterized=True,
        shading="auto",
        cbar_kwargs=dict(label=f"{ds.gap.long_name} [{ds.gap.units}]", pad=0.025),
    )
    im.set_edgecolor("face")

    c = gap.plot.contour(x="f", y="p", ax=ax, levels=[-0.1])
    blur_contours(ax, c)

    ax.set_xlim(xmin=field_min, xmax=field_max)
    ax.set_ylim(ymin=plunger_min, ymax=plunger_max)
    plt.locator_params(axis="x", nbins=4)
    fig.tight_layout()

    if label is not None:
        tgp.plot.paper.add_subfig_label(ax, f"({label})", width=0.085, height=0.115)

    maybe_save_fig(fname)


def blur_contours(
    ax,
    contours,
    sigma=(20, 20),
    color="k",
    linewidth=0.9,
    ignore_small_contours_below=20,
):
    # Filter contours to make them less jagged
    for collection in contours.collections:
        for p in collection.get_paths():
            if len(p) < ignore_small_contours_below:
                continue
            cc = p.vertices
            cc[:, 0] = gaussian_filter(
                cc[:, 0], sigma=sigma[0], mode="nearest", order=0
            )
            cc[:, 1] = gaussian_filter(
                cc[:, 1], sigma=sigma[1], mode="nearest", order=0
            )
            cc = np.vstack([cc, cc[0]])
            ax.plot(*cc.T, color, linewidth=linewidth)

    for path in contours.collections:  # remove initial contour
        path.remove()


def _reshape(ds, a, b):
    return ds[a].values[:, None].repeat(ds[b].shape[1], axis=1)


def plot_local_and_non_local_conductance(
    ds_left, ds_right, fname, antisymmetrize: bool = True, with_gap_trace: bool = False
):
    ds_left = tgp.prepare.rename(ds_left)
    ds_right = tgp.prepare.rename(ds_right)

    fig, axs = plt.subplots(1, 4, figsize=(4.2 * 4, 4.3))

    vmax_l = 1.5
    vmax_nl = 0.03

    kw = dict(
        cmap="viridis",
        vmin=0.0,
        vmax=vmax_l,
        linewidth=0,
        rasterized=True,
        shading="nearest",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pcm = axs[0].pcolormesh(
            _reshape(ds_left, "V", "g_ll"),
            ds_left["lb"],
            ds_left["g_ll"],
            **kw,
        )
    pcm.set_edgecolor("face")
    axs[0].set_ylabel(r"Left bias [$\mu$V]")
    cax = axs[0].inset_axes([0, 1.05, 1, 0.05], transform=axs[0].transAxes)
    cbar = plt.colorbar(
        pcm,
        extend="max",
        ax=axs[0],
        cax=cax,
        orientation="horizontal",
        ticklocation="top",
    )
    cbar.set_label(r"$G_\mathrm{LL}$ [e$^2$/h]")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pcm = axs[1].pcolormesh(
            _reshape(ds_right, "V", "g_rr"),
            ds_right["rb"],
            ds_right["g_rr"],
            **kw,
        )
    pcm.set_edgecolor("face")
    axs[1].set_ylabel(r"Right bias [$\mu$V]")
    cax = axs[1].inset_axes([0, 1.05, 1, 0.05], transform=axs[1].transAxes)
    cbar = plt.colorbar(
        pcm,
        extend="max",
        ax=axs[1],
        cax=cax,
        orientation="horizontal",
        ticklocation="top",
    )
    cbar.set_label(r"$G_\mathrm{RR}$ [e$^2$/h]")

    g_rl = np.squeeze(ds_left["g_rl"].data)
    if antisymmetrize:
        g_rl = 0.5 * (g_rl - g_rl[:, ::-1])
    kw2 = dict(
        cmap="PuOr_r",
        vmin=-vmax_nl,
        vmax=vmax_nl,
        linewidth=0,
        rasterized=True,
        shading="nearest",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pcm = axs[2].pcolormesh(
            _reshape(ds_left, "V", "g_rl"), ds_left["lb"], g_rl, **kw2
        )
    pcm.set_edgecolor("face")
    axs[2].set_ylabel(r"Left bias [$\mu$V]")
    cax = axs[2].inset_axes([0, 1.05, 1, 0.05], transform=axs[2].transAxes)
    cbar = plt.colorbar(
        pcm,
        extend="both",
        ax=axs[2],
        cax=cax,
        orientation="horizontal",
        ticklocation="top",
    )
    cbar.set_label(
        r"$A(G_\mathrm{RL})$ [e$^2$/h]"
        if antisymmetrize
        else r"$G_\mathrm{RL}$ [e$^2$/h]"
    )

    g_lr = np.squeeze(ds_right["g_lr"].data)
    if antisymmetrize:
        g_lr = 0.5 * (g_lr - g_lr[:, ::-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pcm = axs[3].pcolormesh(
            _reshape(ds_right, "V", "g_lr"), ds_right["rb"], g_lr, **kw2
        )
    pcm.set_edgecolor("face")
    axs[3].set_ylabel(r"Right bias [$\mu$V]")
    cax = axs[3].inset_axes([0, 1.05, 1, 0.05], transform=axs[3].transAxes)
    cbar = plt.colorbar(
        pcm,
        extend="both",
        ax=axs[3],
        cax=cax,
        orientation="horizontal",
        ticklocation="top",
    )
    cbar.set_label(
        r"$A(G_\mathrm{LR})$ [e$^2$/h]"
        if antisymmetrize
        else r"$G_\mathrm{LR}$ [e$^2$/h]"
    )

    gap_parent = 300.0
    gap_induced = 130.0
    depletion = -1.27

    def pgp(gp):
        p = np.ones_like(gp) * depletion
        p[np.abs(gp) < gap_induced] *= np.nan
        return p

    # Dotted line at the parent gap
    for i, ax in enumerate(axs[:2]):
        ax.plot(
            [ds_left["V"].max(), ds_left["V"].min()],
            [gap_parent, gap_parent],
            c="w",
            lw=1.5,
            ls=":",
        )
        ax.plot(
            [ds_left["V"].max(), ds_left["V"].min()],
            [-gap_parent, -gap_parent],
            c="w",
            lw=1.5,
            ls=":",
        )

    # Dashed line at the induced gap
    for i, ax in enumerate(axs[2:]):
        ax.plot(
            [ds_left["V"].max(), depletion],
            [gap_parent, gap_parent],
            c="k",
            lw=1.5,
            ls=":",
        )
        ax.plot(
            [ds_left["V"].max(), depletion],
            [-gap_parent, -gap_parent],
            c="k",
            lw=1.5,
            ls=":",
        )
        gp = np.linspace(-gap_parent, gap_parent, 201)
        ax.plot(pgp(gp), gp, c="k", lw=1.5, ls="--")
        for b in [-gap_induced, gap_induced]:
            ax.plot([depletion, depletion + 0.025], [b] * 2, c="k", lw=1.5, ls="--")
    if with_gap_trace:
        # Plot gap
        ds_left, ds_right = tgp.two.extract_gap(ds_left, ds_right)
        for ds, ax in zip([ds_left, ds_right], axs[-2:]):
            gap = 1e6 * ds.gap
            gap = gap.where((gap.V > depletion) & (gap < gap_parent))
            for sign in (-1, +1):
                (sign * gap).plot(x="V", ax=ax, color="k", lw=1.5)

    for i, ax in enumerate(axs):
        ax.set_xticks(np.arange(-1.5, -0.25, 0.25))
        ax.set_xlabel(r"$V_\mathrm{p}$ [V]")
        ax.set_ylim([-450, 450])
        ax.set_xlim(ds_left["V"].min(), -0.75)
        tgp.plot.paper.add_subfig_label(
            ax, "(" + "abcd"[i] + ")", width=0.14, height=0.16
        )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35)

    maybe_save_fig(fname)
    plt.show()


def load_cached_broadened(
    fname: Path,
    T_mK: float,
    force: bool = False,
    returns: bool = True,
    folder: Path = None,
) -> xr.Dataset:
    """Load cached broadened data."""
    if folder is None:
        folder = fname.parent
    else:
        folder = fname.parent / folder
        folder.mkdir(exist_ok=True, parents=True)
    fname_pre = folder / f"{fname.stem}_{T_mK:.1f}mK.nc"
    if (
        not force and fname_pre.exists()
    ):  # We precalculated the data for this temperature
        ds = xr.load_dataset(fname_pre)
    else:
        ds = xr.load_dataset(fname, engine="h5netcdf")
        ds = tgp.prepare.prepare(ds, fillna=False)
        # Broadening takes 6GB of memory and takes a few minutes
        ds = tgp.prepare.broaden_with_temperature(ds, T_mK)
        ds.to_netcdf(fname_pre, engine="h5netcdf")
    if returns:
        return ds


def plot_pf(
    ds_a,
    ds_b,
    threshold=10,
    ylim: tuple[float, float] | None = None,
    fname: str | None = None,
):
    fig, axs = plt.subplots(
        ncols=2, constrained_layout=True, figsize=(13, 4.5), sharey=True
    )

    _plot_pf_panel(
        fig,
        axs[0],
        ds_a,
        threshold=threshold,
        label="a",
        ylim=ylim,
        show_colorbars=False,
    )

    _plot_pf_panel(
        fig,
        axs[1],
        ds_b,
        threshold=threshold,
        label="b",
        ylim=ylim,
    )

    maybe_save_fig(fname)
    plt.show()


def _plot_pf_panel(
    fig,
    ax,
    ds_sel,
    threshold=10,
    label="a",
    ylim: tuple[float, float] | None = None,
    show_colorbars: bool = True,
):
    ds_sel["b_field"].attrs["units"] = "T"
    ds_sel["b_field"].attrs["long_name"] = "$B$"

    ds_sel["mu"].attrs["units"] = "meV"
    ds_sel["mu"].attrs["long_name"] = r"$\mu$"

    gapped = (ds_sel["pfaffian"] == -1) * (ds_sel["e1"] > threshold * 1e-3)

    colors = (
        (0, [0.8359375, 0.15234375, 0.15625, 0.75]),
        (1, [0.12109375, 0.46484375, 0.703125, 0.2]),
    )

    cmap = LinearSegmentedColormap.from_list("pfaffian", colors=colors, N=2)
    pcm = ds_sel["pfaffian"].plot.pcolormesh(
        x="b_field",
        y="mu",
        ax=ax,
        cmap=cmap,
        vmin=-2,
        vmax=2,
        rasterized=True,
        add_colorbar=False,
    )

    cmap2 = LinearSegmentedColormap.from_list(
        "EL", colors=((0, [0, 0, 0, 0]), (1, [0, 0, 0, 1])), N=2
    )
    pcm2 = gapped.plot.pcolormesh(
        x="b_field",
        y="mu",
        ax=ax,
        cmap=cmap2,
        vmin=-0.5,
        vmax=1.5,
        rasterized=True,
        add_colorbar=False,
    )
    ax.set_title("")

    if ylim is not None:
        ax.set_ylim(ylim)

    if show_colorbars:
        cax = fig.add_axes([1.015, 0.4, 0.025, 0.25])
        cbar = fig.colorbar(
            pcm,
            cax=cax,
            orientation="vertical",
            ticks=[-1, 1],
        )
        cbar.ax.set_yticklabels([r"$\mathcal{Q} = -1$", r"$\mathcal{Q} = +1$"])

        cax = fig.add_axes([1.015, 0.130, 0.025, 0.17])
        cbar = fig.colorbar(
            pcm2,
            cax=cax,
            orientation="vertical",
            ticks=[1],
        )
        cbar.ax.set_ylim(0.5, 1.5)
        cbar.ax.set_yticklabels(
            [f"$\\mathcal{{Q}} = -1$ &\n${{E_1}} > {threshold} \\,\\mu$eV"]
        )

        ax.set_ylabel("")

    title = rf"$\delta V = {ds_sel.rms.item():.1f}\,$meV"
    ax.text(
        ds_sel["b_field"].min() + 0.07,
        ds_sel["mu"].min() + 0.15,
        title,
        c="k",
        ha="left",
        va="baseline",
        fontsize=12,
    )

    tgp.plot.paper.add_subfig_label(ax, "(" + label + ")", width=0.075, height=0.10)


def plot_conductance_cuts(ds_c_left, selected_plunger, selected_field, fname):
    n = 400
    cmap1 = mpl.cm.get_cmap("viridis", n + 1)

    fig, ax = plt.subplots(1, 4, figsize=(0.85 * 4 * 4, 4), sharex=True, sharey=True)
    ims = []
    for c in range(ds_c_left["cutter_pair_index"].size):
        g_ll = ds_c_left.sel(
            {"V": selected_plunger, "cutter_pair_index": c}, method="nearest"
        ).g_ll
        B = np.array(g_ll["B"])
        lb = np.array(g_ll["left_bias"])
        g_ll = np.array(g_ll)

        im = ax[c].pcolormesh(
            B,
            1e3 * lb,
            g_ll.T,
            shading="auto",
            vmin=0,
            vmax=1,
            cmap=cmap1,
            linewidth=0,
            rasterized=True,
        )
        ims.append(im)
        im.set_edgecolor("face")
        if c == 0:
            ax[c].set_ylabel(r"Left bias [$\mu$V]")
        ax[c].set_xlabel("$B$ [T]")

        ax[c].plot([selected_field] * 2, [-50, 50], c="w", ls=":", lw=2)

        tgp.plot.paper.add_subfig_label(
            ax[c], "(" + "abcd"[c] + ")", width=0.16, height=0.12
        )
        ax[c].text(
            0.03,
            0.55,
            f"#{c+1}",
            c="w",
            ha="left",
            va="center",
            transform=ax[c].transAxes,
            fontsize=12,
        )
        ax[c].set_xticks([0.75, 1, 1.25, 1.5])

    ax[0].set_yticks(np.arange(-80, 81, 20))

    pos = ax[3].get_position()
    cax = fig.add_axes([pos.x1 + 0.016, pos.y0 - 0.002, 0.01, pos.y1 - pos.y0])
    cb = fig.colorbar(ims[-1], cax=cax, orientation="vertical", extend="max")
    cb.set_label(r"$G_\mathrm{LL}$ [$e^2/h$]")

    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    maybe_save_fig(fname)
    plt.show()


def plot_conductance_traces(ds_c_left, selected_field, selected_plunger, fname):
    fig, ax1 = plt.subplots(1, 1, figsize=(0.85 * 5, 4))

    for c in range(ds_c_left["cutter_pair_index"].size):
        g = ds_c_left.sel(
            {
                "B": selected_field,
                "V": selected_plunger,
                "cutter_pair_index": c,
            },
            method="nearest",
        ).g_ll
        g = g.where(1e3 * np.abs(g["left_bias"]) < 50, np.nan)
        ax1.plot(1e3 * g["left_bias"], g, label=f"#{c+1}", lw=2, marker="o", ms=5)

    ax1.set_xlabel(r"Left bias [$\mu$V]")
    ax1.set_ylabel(r"$G_\mathrm{LL}$ [$e^2/h$]")

    ax1.set_ylim(-0.02, 1.19)

    tgp.plot.paper.add_subfig_label(ax1, "(e)", width=0.12, height=0.12)

    ax1.legend()
    plt.tight_layout()
    maybe_save_fig(fname)
    plt.show()


def plot_disorder_strength_vs_n2D(df_β, df_δ, df_ε, fname: str | None = None):
    def curve(x, a, b):
        return (a * x) ** 0.5 + b

    dfs = [df_β, df_δ, df_ε]
    labels = [
        r"SLG, $\beta$-stack",
        r"DLG, $\delta$-stack",
        r"DLG, $\varepsilon$-stack",
    ]
    colors = ["tab:red", "tab:green", "tab:blue"]
    fmts = ["o--", "o-.", "o:"]
    fig, ax = plt.subplots(figsize=(7.5, 5))

    x1 = np.linspace(0, 5, 101)
    for fmt, color, label, df1 in zip(fmts, colors, labels, dfs):
        df1 = df1.sort_values("n_2D")
        x = df1["n_2D"].values / 1e12
        y = df1["V_d"].values
        a1, b1 = curve_fit(curve, x[x < 5], y[x < 5])[0]
        ax.scatter(x, y, marker=fmt[0], color=color, zorder=1000)
        ax.plot(
            x1,
            [curve(x2, a1, b1) for x2 in x1],
            fmt[1:],
            color=color,
            lw=2,
            zorder=1000,
        )
        ax.plot([], [], fmt, label=label, color=color)

    ax.set(
        ylim=[0, None],
        yticks=np.arange(0, 2.5, 0.5),
        xlabel=r"$n_\mathrm{2D,int}$ [$10^{12} / \mathrm{cm}^2$]",
        ylabel=r"Disorder strength, $\delta V$ [meV]",
    )
    ax.set_ylim([0, None])
    ax.tick_params(axis="x", which="minor", bottom=True)
    plt.minorticks_on()
    plt.legend()
    plt.grid(lw=0.25)
    maybe_save_fig(fname)
    plt.show()


def _plot_xi_panel(
    ds_sel,
    ax=None,
    xlims=(0.5, 3.6),
    ylims=(-1, 2),
    sigma: float | tuple[float, float] = None,
    fname: str | None = None,
    with_cbar: bool = True,
    show: bool = True,
):
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True, figsize=(7.5, 4.5))

    ds_sel["b_field"].attrs["units"] = "T"
    ds_sel["b_field"].attrs["long_name"] = "$B$"

    ds_sel["mu"].attrs["units"] = "meV"
    ds_sel["mu"].attrs["long_name"] = r"$\mu$"

    Z = ds_sel["bdg_correlation_length"] / 1e3
    if sigma:
        Z = xr.apply_ufunc(
            gaussian_filter,
            Z,
            input_core_dims=[["mu", "b_field"]],
            output_core_dims=[["mu", "b_field"]],
            kwargs=dict(sigma=sigma),
        )
    pcm = Z.plot.pcolormesh(
        x="b_field",
        y="mu",
        rasterized=True,
        cmap="magma_r",
        norm=LogNorm(vmin=0.3, vmax=10),
        add_colorbar=False,
        ax=ax,
    )
    if with_cbar:
        cb = plt.colorbar(
            pcm,
            label="$\\xi(0)$ [$\\mu$m]",
            ticks=[0.3, 1, 3, 10],
            extend="both",
            pad=0.025,
        )
        cb.ax.set_yticklabels([0.3, 1, 3, 10])

        for length, ls in zip([1.0, 3], ["-", "--"]):
            cb.ax.plot([0, 1], [length, length], "w", ls=ls)

    Z.plot.contour(
        ax=ax, x="b_field", y="mu", levels=[1.0, 3], colors="w", linestyles=["-", "--"]
    )

    ax.text(
        ds_sel["b_field"].min() + 0.07,
        ds_sel["mu"].min() + 0.15,
        f"$\\delta V = {ds_sel.rms.item():.1f}\\,$meV",
        c="k",
        ha="left",
        va="baseline",
        fontsize=12,
    )

    ax.set_title("")
    ax.set_xlabel("$B$ [T]")
    ax.set_ylabel("$\\mu$ [meV]")
    ax.set_yticks(np.arange(-1, 2.1, 0.5))
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)

    maybe_save_fig(fname)
    if show:
        plt.show()


def plot_xi(ds_a, ds_b, xlims=(0.5, 3.6), ylims=(-1, 2), fname: str | None = None):
    fig, axs = plt.subplots(
        ncols=2, constrained_layout=True, figsize=(13, 4.5), sharey=True
    )
    _plot_xi_panel(
        ds_a, ax=axs[0], xlims=xlims, ylims=ylims, show=False, with_cbar=False
    )
    tgp.plot.paper.add_subfig_label(axs[0], "(a)", width=0.075, height=0.10)
    _plot_xi_panel(
        ds_b,
        ax=axs[1],
        xlims=xlims,
        ylims=ylims,
        show=False,
        with_cbar=True,
        sigma=1,
    )
    axs[1].set_ylabel("")
    tgp.plot.paper.add_subfig_label(axs[1], "(b)", width=0.075, height=0.10)
    maybe_save_fig(fname)
    plt.show()


def _select_closest_bias(
    data: xr.Dataset | xr.DataArray,
    bias: float,
    bias_tolerance: float,
    uncorrected_bias_label: str = "left_bias",
    corrected_bias_label: str = "left_bias_sample",
):
    """Select the closest bias within the specified tolerance.

    Parameters
    ----------
    data
        Input data containing the bias values.
    bias
        Target bias value.
    bias_tolerance
        Tolerance for the closest bias value.
    uncorrected_bias_label
        Label for the uncorrected bias dimension in the input data.
    corrected_bias_label
        Label for the corrected bias dimension in the input data.

    Returns
    -------
    xr.DataArray:
        DataArray with the closest bias values.
    """
    not_all_nan_slices = ~(
        np.isnan(data[corrected_bias_label]).all(dim=uncorrected_bias_label)
    )
    corrected_bias = data[corrected_bias_label].where(not_all_nan_slices, 0)
    closest_bias_index = np.abs(corrected_bias - bias).argmin(
        dim=uncorrected_bias_label
    )
    bias_corresponding_to_index = corrected_bias[
        {uncorrected_bias_label: closest_bias_index}
    ]
    return data[{uncorrected_bias_label: closest_bias_index}].where(
        (np.abs(bias_corresponding_to_index - bias) < bias_tolerance)
        & (not_all_nan_slices)
    )


def _compute_Y(
    nonlocal_data: xr.DataArray,
    parent_gap: float,
    uncorrected_bias_label: str = "left_bias",
    corrected_bias_label: str = "left_bias_sample",
):
    """Compute the average value of nonlocal_data over the range defined by parent_gap.

    Parameters
    ----------
    nonlocal_data
        Input data containing nonlocal conductance values.
    parent_gap
        Range for averaging the nonlocal_data.
    corrected_bias_label
        Label for the corrected bias dimension in the input data.
    uncorrected_bias_label
        Label for the uncorrected bias dimension in the input data.

    Returns
    -------
    xr.DataArray
        DataArray with the computed average values.
    """

    mask_below_gap = (nonlocal_data[corrected_bias_label] > -parent_gap) & (
        nonlocal_data[corrected_bias_label] < parent_gap
    )
    return (
        xr.apply_ufunc(
            np.trapz,
            nonlocal_data.where(mask_below_gap, 0),
            nonlocal_data.where(mask_below_gap, 0)[corrected_bias_label],
            input_core_dims=[[uncorrected_bias_label], [uncorrected_bias_label]],
        )
        / parent_gap
    )


def _correct_hymp_dataset(
    ds,
    r_left: float = 1850.0,
    r_right: float = 1850.0,
    r_gnd: float = 1850.0,
    left_dc_current_name: str = "i_l",
    right_dc_current_name: str = "i_r",
) -> xr.Dataset:
    from tgp.bias_correction import correct_three_terminal

    return correct_three_terminal(
        ds,
        r_left=r_left,
        r_right=r_right,
        r_gnd=r_gnd,
        left_dc_current_name=left_dc_current_name,
        right_dc_current_name=right_dc_current_name,
    )


def load_hmp_datasets(path: Path, parent_gap: float = 0.2) -> list[xr.Dataset]:
    """Load datasets from a directory and process them.

    Parameters
    ----------
    path
        Path to the directory containing the datasets.
    parent_gap
        Range for averaging the nonlocal_data.

    Returns
    -------
    list[xr.Dataset]
        Tuple containing the combined dataset and a list of individual datasets.
    """
    path = Path(path)
    paths = path.glob("*.nc")
    datasets = []
    for p in sorted(paths):
        ds = xr.load_dataset(p)
        L = ds.L
        ds = _correct_hymp_dataset(ds[["g_rr", "g_rl", "g_lr", "g_ll", "i_l", "i_r"]])
        for k in ["left_bias", "right_bias", "left_bias_sample", "right_bias_sample"]:
            ds[k] = ds[k] * 1000
            ds[k].attrs["units"] = "mV"
        ds = ds.assign_coords(L=L)
        ds["L"].attrs["units"] = "nm"
        ds["iG"] = _compute_Y(ds.g_rl, parent_gap)
        ds["Gzb"] = _select_closest_bias(ds.g_rl, 0, 1e-2)
        ds = ds.squeeze("right_bias")
        datasets.append(ds)

    return datasets


def _localized_model(x: np.ndarray, A: float, localization_length: float) -> np.ndarray:
    """Localized conductance model.

    Parameters
    ----------
    x:
        x-data for the model.
    A
        Amplitude parameter.
    localization_length
        Localization length parameter.

    Returns
    -------
    np.ndarray
        Resulting conductance values.
    """
    return A * np.exp(-x / (localization_length / 2))


def _fit_conductance_scaling_model(
    x: np.ndarray,
    y: np.ndarray,
    noise_floor: float,
) -> tuple[float, float, float, float, float]:
    """Fit the conductance scaling model to the data.

    Parameters
    ----------
    x
        x-data to be fitted.
    y
        y-data to be fitted.
    noise_floor
        Noise floor for weighing the fit.

    Returns
    -------
    Tuple[float, float, float, float, float]:
        Tuple containing the fitted parameters (A, localization_length),
        R-squared, and uncertainties (A_err, l_err).
    """
    from scipy.stats import linregress, t
    from sklearn.metrics import r2_score

    inds = y > noise_floor
    x, y = x[inds], y[inds]

    res = linregress(x, np.log(y), alternative="less")
    localization_length = -2 / res.slope

    A = np.exp(res.intercept)
    y_pred = _localized_model(x, A, localization_length)

    r2 = r2_score(y, y_pred) if np.sum(np.isnan(y_pred)) == 0 else 0

    ts = abs(t.ppf(0.05 / 2, 4))
    l_err = ts * res.stderr
    A_err = ts * res.intercept_stderr * A / np.abs(res.intercept)

    return A, localization_length, r2, A_err, l_err


def _localization_length_estimate(
    gnl: xr.DataArray,
    noise_floor: float,
    *,
    length_dim: str = "L",
    ax: mpl.axes.Axes | None = None,
    draw_line: bool = False,
    label: str | None = None,
    xlabel: str = r"$V_\mathrm{p}$",
) -> xr.Dataset:
    """Estimate the localization length from conductance data.

    Parameters
    ----------
    gnl
        Input data containing nonlocal conductance values.
    length_dim
        Length dimension label.
    ax
        Axes to plot on, if none is provided, no plot is drawn.
    draw_line
        Whether to draw a line on the plot.
    label
        Plot label.
    xlabel
        x-axis label.
    noise_floor
        Noise floor for weighing the fit.

    Returns
    -------
    xr.Dataset
    """
    A, localization_length, r2, A_err, l_err = xr.apply_ufunc(
        _fit_conductance_scaling_model,
        gnl.L,
        gnl,
        kwargs=dict(noise_floor=noise_floor),
        input_core_dims=[[length_dim], [length_dim]],
        output_core_dims=[(), (), (), (), ()],
        vectorize=True,
        output_dtypes=[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    )
    ds = xr.Dataset(
        {
            "A": A.astype(float),
            "localization_length": localization_length.astype(float),
            "r2": r2.astype(float),
            "A_err": A_err.astype(float),
            "l_err": l_err.astype(float),
        }
    )
    if ax is not None:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\mathrm{localization_length}$")

        x, y = xr.broadcast(gnl.active_gates, ds.localization_length)

        if draw_line:
            ax.plot(x, 1e-3 * y)

        s = np.array(np.maximum(ds.r2, 0), dtype=float)
        im = ax.scatter(
            x,
            1e-3 * y,
            label=label,
            c=ds.r2,
            vmin=0,
            vmax=1,
            s=s * matplotlib.rcParams["lines.markersize"] ** 2,
        )
        plt.colorbar(im, label="$R^2$", ax=ax, pad=-0.04, aspect=40)

    return ds


def _noisy_threshold(
    data: xr.DataArray, threshold: float, dim: str = "active_gates"
) -> xr.DataArray:
    """Find the first index along a dimension where data exceeds a threshold.

    Parameters
    ----------
    data
        Input data.
    threshold
        Threshold value.
    dim
        Dimension to search along.

    Returns
    -------
    xr.DataArray
        Array with the first index where data exceeds the threshold.
    """
    return (data > threshold).idxmax(dim)


def _integrated_length_scaling(
    datasets: xr.Dataset, integration_window: float, variable: str = "g_rl"
) -> xr.DataArray:
    """Compute the integrated length scaling for a list of datasets.

    Parameters
    ----------
    datasets
        Input datasets.
    integration_window
        Range for averaging the nonlocal_data.
    variable
        Variable to compute the integrated length scaling on.

    Returns
    -------
    xr.DataArray
        Integrated length scaling values.
    """
    arrs = [_compute_Y(ds[variable], integration_window) for ds in datasets]
    return xr.concat(arrs, "L")


def _plot_localization_length(
    datasets: xr.Dataset,
    wire_pinchoff: xr.DataArray,
    ax: mpl.axes.Axes,
    *,
    noise_floor: float,
    variable: str = "g_rl",
    local_normalization: bool = True,
    variable_l: str = "g_ll",
    variable_r: str = "g_rr",
) -> tuple[xr.Dataset, xr.Dataset]:
    """Plot the localization length for a list of datasets.

    Parameters
    ----------
    datasets
        Input datasets.
    wire_pinchoff
        Wire pinchoff value.
    ax
        Axes to plot on.
    noise_floor
        Noise floor for weighing the fit.
    variable
        Variable to compute the localization length on.
    local_normalization
        Whether to perform local normalization.
    variable_l
        Left variable for local normalization.
    variable_r
        Right variable for local normalization.

    Returns
    -------
    Tuple[xr.Dataset, xr.Dataset]
    """
    ds = _integrated_length_scaling(datasets, 0.01, variable=variable)
    if local_normalization:
        ds_l = _integrated_length_scaling(datasets, 0.01, variable=variable_l)
        ds_r = _integrated_length_scaling(datasets, 0.01, variable=variable_r)
        ds = ds / np.sqrt(ds_l * ds_r)

    ds = ds.sel(active_gates=ds.active_gates[ds.active_gates > wire_pinchoff.values])

    ds_localization_length = _localization_length_estimate(
        -ds, ax=ax, noise_floor=noise_floor
    )
    return ds, ds_localization_length


def _plot_length_scaling(
    datasets: xr.Dataset,
    axs: mpl.axes.Axes,
    cax: mpl.axes.Axes,
    ylabel: str = r"$V_\mathrm{p}$ [V]",
    vmin: float = 0,
    vmax: float = 2,
    varname: str = "g_rl",
) -> None:
    """Plot the length scaling for a list of datasets.

    Parameters
    ----------
    datasets
        Input datasets.
    axs
        Axes to plot on.
    cax
        Axes for the colorbar.
    ylabel
        y-axis label.
    vmin
        Minimum value for the colormap.
    vmax
        Maximum value for the colormap.
    varname
        Variable to plot.
    """

    for ax in axs:
        ax.set_xlabel(r"Bias [$\mu$V]")
        ax.set_xticks(ticks=[-0.2, 0, 0.2])
        ax.set_xticklabels(["$-200$", "$0$", "$200$"])

    axs[0].set_ylabel(ylabel)
    cmap = plt.get_cmap("OrRd")

    norm = matplotlib.colors.Normalize(vmin, vmax)

    for ds, ax in zip(datasets, axs):
        x, y, z = xr.broadcast(ds.left_bias_sample, ds.active_gates, -ds[varname])

        excluded_region = x.isnull().any("left_bias")

        x = x.where(~excluded_region, drop=True)
        y = y.where(~excluded_region, drop=True)
        z = z.where(~excluded_region, drop=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = ax.pcolormesh(
                x, y, z, norm=norm, cmap=cmap, linewidth=0, rasterized=True
            )
        prefix = r"$L =" if np.isclose(ds.L, 1000) else "$"
        ax.set_title(prefix + r"%g\,\mu$m" % (float(ds.L.values / 1000)))

    plt.colorbar(
        im,
        cax=cax,
        label=r"$-G_{\mathrm{RL}}$ [$e^2/h$]",
        extend="both",
        extendfrac=0.03,
    )


def plot_hybrid_mobility_protocol(
    datasets: xr.Dataset,
    *,
    wire_pinchoff_threshold: float = 0.05,
    mesa_pinchoff_threshold: float = 4,
    local_normalization: bool = True,
    fig_scale: float = 2.5,
    fig_relative_height: float = 1 / 6,
    pb_relative_width: float = 0.7,
    pb_cb_width: float = 0.05,
    spacer: float = 0.001,
    lower_gate_voltage_padding: float = 0.05,
    yscale: str = "log",
    ylims: tuple[float | None, float | None] = (None, None),
    dpi: int | None = 300,
    noise_floor: float = 3e-3,
    variable: str = "g_rl",
    fname: str | None = None,
) -> tuple[mpl.figure.Figure, xr.Dataset, xr.Dataset]:
    """Create a plot for the paper.

    Parameters
    ----------
    datasets
        Input datasets.
    wire_pinchoff_threshold
        Wire pinchoff threshold.
    mesa_pinchoff_threshold
        Mesa pinchoff threshold.
    local_normalization
        Whether to perform local normalization.
    fig_scale
        Figure scale.
    fig_relative_height
        Figure relative height.
    pb_relative_width
        Plot box relative width.
    pb_cb_width
        Plot box colorbar width.
    spacer
        Spacer width.
    lower_gate_voltage_padding
        Lower gate voltage padding.
    yscale
        y-axis scale.
    ylims
        y-axis limits.
    dpi
        DPI for the figure.
    noise_floor
        Noise floor for weighing the fit.
    variable
        Variable to compute the localization length on.

    Returns
    -------
    Tuple[mpl.figure.Figure, xr.Dataset, xr.Dataset]
    """
    fig = plt.figure(
        constrained_layout=True,
        figsize=(fig_scale * 6.5, fig_scale * 13 * fig_relative_height),
        dpi=dpi,
    )

    pb_total_width = pb_relative_width + pb_cb_width
    pbcol_width = pb_total_width / (1 - pb_total_width) / len(datasets)
    widths = len(datasets) * [pbcol_width] + [pb_cb_width] + [spacer] + [1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=2, width_ratios=widths)

    axs_pb = [fig.add_subplot(spec[0:2, 0])]
    axs_pb += [
        fig.add_subplot(spec[0:2, i], sharey=axs_pb[0]) for i in range(1, len(datasets))
    ]
    for ax in axs_pb[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

    ax_cb = fig.add_subplot(spec[0:2, len(axs_pb)])
    ax_ll = fig.add_subplot(spec[1, -1])
    ax_2 = fig.add_subplot(spec[0, -1])

    # pinchoffs
    wire_pinchoff = _noisy_threshold(-datasets[0].Gzb, wire_pinchoff_threshold)
    mesa_pinchoff = _noisy_threshold(-datasets[-1].Gzb, mesa_pinchoff_threshold)

    gate_lims = (wire_pinchoff - lower_gate_voltage_padding, mesa_pinchoff)

    # Length scaling
    _plot_length_scaling(datasets, axs_pb, cax=ax_cb)
    for ax in axs_pb:
        ax.set_ylim(*gate_lims)
        ax.axhline(wire_pinchoff, c="k", linestyle=":", lw=2)

    # Localization length
    ds, ds_localization_length = _plot_localization_length(
        datasets,
        wire_pinchoff,
        ax_ll,
        local_normalization=local_normalization,
        noise_floor=noise_floor,
        variable=variable,
    )
    ax_ll.set_xlim(gate_lims[0] - 0.07, gate_lims[1])

    ax_ll.set_yscale(yscale)
    ax_ll.set_ylim(*ylims)
    ax_ll.axvline(wire_pinchoff, c="k", linestyle=":", lw=2)
    ax_ll.set_ylabel(r"$\ell_{\mathrm{loc}}$ [$\mu$m]")

    ax_2.set_xlabel(r"$L$ [$\mu$m]")
    ax_2.set_ylabel(r"$-G_\mathrm{RL}/\sqrt{G_\mathrm{RR}G_\mathrm{LL}}$")

    ax_2.set_ylim(5e-3, 1)
    ax_2.set_xlim(-1, 8.5)
    ax_2.set_yscale("log")

    Vp_arr = [-1.0]
    for Vp in Vp_arr:
        y = -ds.sel(active_gates=Vp, method="nearest")
        x = (y.L / 1e3).data
        ds_loc1 = _localization_length_estimate(y, noise_floor=noise_floor)
        _mfp1 = float(ds_loc1.localization_length.values.flatten()[0] / 1e3)

        sp = ax_2.plot(x, y, "o", label=r"$V_\mathrm{{p}} = %g\,$V" % Vp)
        ax_2.plot(
            x,
            ds_loc1.A.values.flatten()[0] * np.exp(-(np.array(x)) / (_mfp1 / 2)),
            label=None,
            color=sp[0].get_color(),
        )

    ax_2.legend(loc="upper right")

    tgp.plot.paper.add_subfig_label(
        axs_pb[0], "(a)", width=0.20 / 0.7, height=0.05 / 0.7
    )
    tgp.plot.paper.add_subfig_label(
        ax_2, "(b)", width=0.09 / 0.7, height=0.43 / 0.7 * 0.9, text_yshift=-0.06
    )
    tgp.plot.paper.add_subfig_label(
        ax_ll, "(c)", width=0.09 / 0.7, height=0.28 / 0.7 * 0.9, text_yshift=-0.01
    )

    plt.minorticks_on()
    maybe_save_fig(fname)
    return fig, ds, ds_localization_length


def plot_hall_bar_trans_long_resistance(
    ds_hb: xr.Dataset, device_image_fname: Path, fname: str | None = None
) -> None:
    fig, ax = plt.subplots(
        figsize=(13, 3.6), ncols=3, nrows=1, gridspec_kw={"width_ratios": [1.2, 2, 2]}
    )

    # Plot image
    img = np.asarray(Image.open(device_image_fname))
    ax[0].imshow(img)
    ax[0].axis("off")
    tgp.plot.paper.add_subfig_label(ax[0], "(a)", width=0.11 * 2 / 1.2, height=0.13)

    # Plot R_xy
    Vg = np.arange(-0.55, 0.01, 0.11)
    for v in Vg:
        ds_hb.R_xy.real.sel(V_gate=v, method="nearest").plot(
            ax=ax[1], marker="o", ms=4, label=rf"$V_\mathrm{{g}} = {v:.2f}\,$V"
        )
    ax[1].set_xlabel("$B$ [T]")
    ax[1].set_ylabel(r"$R_{xy}$ [$\Omega$]")
    ax[1].legend(bbox_to_anchor=(0.12, 1), loc="upper left", fontsize=10)
    ax[1].set_title("")
    ax[1].set_xlim(-0.1, None)
    ax[1].set_xticks(np.arange(0, 1.01, 0.2))
    tgp.plot.paper.add_subfig_label(ax[1], "(b)", width=0.10, height=0.13)

    # Plot R_xx
    width = 40e-6
    length = 200e-6
    B_sel = 0.2
    (ds_hb.R_xx.real * width / length).sel(B=B_sel, method="nearest").plot(
        ax=ax[2], marker="o", ms=4, label=rf"$B={B_sel:g}\,$T"
    )
    ax[2].set_xlabel(r"$V_\mathrm{g}$ [V]")
    ax[2].set_ylabel(r"$R_{xx}$ [$\Omega/$sq]")
    ax[2].legend()
    ax[2].set_title("")
    ax[2].set_xlim(-0.67, None)
    ax[2].set_xticks(np.arange(-0.6, 0.01, 0.2))
    tgp.plot.paper.add_subfig_label(ax[2], "(c)", width=0.10, height=0.13)

    # Save and show plot
    plt.tight_layout()
    maybe_save_fig(fname)
    plt.show()


def plot_device_characterization(
    ds: xr.Dataset,
    V_sidecutter_left_gate: float = -1.95,
    gap: float = 129e-6,
    fname: str | None = None,
) -> None:
    ds = ds.sel(
        V_sidecutter_left_gate=V_sidecutter_left_gate,
        method="nearest",
    )

    fig, ax = plt.subplots(ncols=2, figsize=(12, 3.3))
    ax[0].plot(ds.left_bias_sample, ds.g_ll)
    ax[0].set_ylabel(r"$G_\mathrm{LL}$ [$e^2/h$]")
    ax[0].set_ylim(0, None)
    ax[1].plot(ds.left_bias_sample, ds.g_rl)
    ax[1].set_ylabel(r"$G_\mathrm{RL}$ [$e^2/h$]")
    ax[1].set_ylim(-0.079, 0.079)

    for i, a in enumerate(ax):
        a.set_title("")
        ticks = np.arange(-0.0004, 0.00041, 0.0002)
        a.set_xticks(ticks=ticks)
        a.set_xticklabels(labels=["$%g$" % (1e6 * t) for t in ticks])
        a.set_xlabel(r"Left bias [$\mu$V]")
        tgp.plot.paper.add_subfig_label(
            a, "(" + "ab"[i] + ")", width=0.083, height=0.15
        )
        for gp in [-gap, gap]:
            a.axvline(gp, ls="--", lw=1.0, c="#a0a0a0", zorder=1)
        a.text(
            0.5,
            0.85,
            r"$2\Delta_\mathrm{ind}$",
            fontsize=14,
            ha="center",
            va="center",
            transform=a.transAxes,
        )
    plt.tight_layout()
    maybe_save_fig(fname)
    plt.show()
