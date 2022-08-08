# Copyright (c) Microsoft Corporation. All rights reserved.

from __future__ import annotations

from copy import deepcopy

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from scipy.ndimage import center_of_mass
import xarray as xr

import tgp

SAVE_KWARGS = dict(
    dpi=300,
    transparent=True,
    bbox_inches="tight",
    pad_inches=0.1,
)


def add_subfig_label(ax, label: str, width: float, height: float) -> None:
    """Add a label to a subfigure."""
    x0, x1 = ax.get_xlim()
    w = width * (x1 - x0)
    y0, y1 = ax.get_ylim()
    h = height * (y1 - y0)
    ax.add_patch(mpl.patches.Rectangle((x0, y1), w, -h, color="k", zorder=1000))
    ax.text(
        x0 + w / 2,
        y1 - h / 2,
        label,
        c="w",
        ha="center",
        va="center",
        size=16,
        zorder=1001,
    )


def _list_diff(a, b):
    sb = set(b)
    return [x for x in a if x not in sb]


def plot_stage2_diagram(
    ds: xr.Dataset,
    cutter_value: float,
    zbp_cluster_numbers: all | list[int] | None = None,
    plunger_cut: int | list[int] | None = None,
    field_cut: float | None = None,
    device_info: dict = dict(),
    pct_boundary_shifts: dict[int, tuple[float, float]] | None = None,
    plunger_lim: tuple[int | None, int | None] = (None, None),
    gap_lim: float = 60.0,
    gap_ticks: list[float] | None = None,
    invariant: str | None = None,
    line_cuts_kw=dict(lw=3, color="k", linestyle=":"),
) -> tuple[mpl.figure.Figure, list[mpl.axes.Axes]]:
    """Plot the stage 2 diagram."""

    fig, axs = plt.subplots(
        ncols=2, constrained_layout=True, figsize=(14.5, 5.5), sharey=True
    )

    if "cutter_pair_index" in ds.dims:
        ds_sel = ds.sel(cutter_pair_index=cutter_value)
    else:
        ds_sel = ds

    gap_bool = 1.0 * ds_sel.gap_boolean.squeeze()
    cmap = mpl.colors.ListedColormap(["w", "tab:blue"])
    pl_kw = dict(
        add_colorbar=False,
        shading="nearest",
        infer_intervals=False,
        linewidth=0,
        rasterized=True,
        vmin=-0.5,
        vmax=1.5,
    )
    im = (
        gap_bool.squeeze()
        .transpose("V", "B")
        .plot.pcolormesh(ax=axs[0], cmap=cmap, zorder=1, **pl_kw)
    )
    cax = fig.add_axes([-0.12, 0.41, 0.03, 0.25])
    cbar = fig.colorbar(im, cax=cax, orientation="vertical", ticks=[0, 1])
    cbar.ax.set_yticklabels(["Gapless", "Gapped"])

    zbp_bool = ds_sel.zbp.squeeze()
    cmap = mpl.colors.ListedColormap([np.array([255, 229, 82]) / 256, "tab:orange"])
    im = (
        gap_bool.where(zbp_bool, np.nan)
        .squeeze()
        .transpose("V", "B")
        .plot.pcolormesh(ax=axs[0], cmap=cmap, zorder=2, **pl_kw)
    )
    cax = fig.add_axes([-0.12, 0.1 - 0.004, 0.03, 0.25])
    cbar = fig.colorbar(
        im,
        cax=cax,
        orientation="vertical",
        ticks=[0, 1],
    )
    cbar.ax.set_yticklabels(["Gapless & ZBP", "Gapped & ZBP"])

    axs[0].set_title("")

    reps = 20
    B = np.array(ds["B"])
    V = np.array(ds["V"])
    B1 = np.linspace(B.min(), B.max(), B.size * reps)
    V1 = np.linspace(V.min(), V.max(), V.size * reps)

    if invariant is not None:
        cs = (
            ds[invariant]
            .astype(float)
            .interp(B=B1, V=V1, method="nearest")
            .plot.contourf(
                x="B",
                y="V",
                ax=axs[0],
                levels=[-1, 0.5, 2],
                colors=[(1, 1, 1, 0), (1, 1, 1, 0)],
                hatches=[None, r"\\\\"],
                linestyles="-",
                zorder=1001,
                add_colorbar=False,
            )
        )

        artists, _ = cs.legend_elements(str_format="{:2.1f}".format)
        plt.legend(
            [artists[1]],
            [r"Topological"],
            handlelength=2.3,
            handleheight=5.8,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(-0.35, 3.1),
            handletextpad=0.5,
        )

    gp = np.abs(ds_sel.gap)
    clusters = tgp.common.expand_clusters(
        ds_sel.gapped_zbp_cluster, dim="zbp_cluster_number"
    )
    existing_zbp_cluster_numbers = np.array(
        clusters.zbp_cluster_number, dtype=int
    ).tolist()
    if zbp_cluster_numbers is None:
        zbp_cluster_numbers = []
    elif zbp_cluster_numbers == "all":
        zbp_cluster_numbers = existing_zbp_cluster_numbers
    else:
        invalid_indices = _list_diff(zbp_cluster_numbers, existing_zbp_cluster_numbers)
        if invalid_indices:
            zbp_cluster_numbers = _list_diff(zbp_cluster_numbers, invalid_indices)
            print(
                "Warning: Clusters with indices",
                invalid_indices,
                "do not exist and are removed from zbp_cluster_numbers",
            )
    for i in zbp_cluster_numbers:
        gp = gp * (1.0 - 2.0 * clusters.sel(zbp_cluster_number=i))

    pcm = (1e3 * gp).plot.pcolormesh(
        ax=axs[1],
        vmin=-gap_lim,
        vmax=gap_lim,
        cmap="RdBu",
        linewidth=0,
        rasterized=True,
        add_colorbar=False,
        shading="nearest",
        infer_intervals=False,
    )
    pcm.set_edgecolor("face")
    cax = axs[1].inset_axes([1.03, 0, 0.03, 1], transform=axs[1].transAxes)
    exceed_min = int(np.min(1e3 * gp) < -gap_lim)
    exceed_max = int(np.max(1e3 * gp) > gap_lim)
    extend = {0: "neither", 1: "min", 2: "max", 3: "both"}[exceed_min + 2 * exceed_max]
    if gap_ticks is None:
        gap_ticks = np.arange(-gap_lim, gap_lim + 1e-5, 10)
    cb = fig.colorbar(
        pcm,
        ax=axs[1],
        cax=cax,
        ticks=gap_ticks,
        extend=extend,
        extendfrac=0.03,
    )
    cb.set_label(r"$q\Delta$ [$\mu$eV]")

    for i in zbp_cluster_numbers:
        print(f"Cluster #{i}")

        cluster = clusters.sel(zbp_cluster_number=i)

        for ax in axs[:2]:
            cluster.astype(float).interp(B=B1, V=V1, method="nearest").plot.contour(
                x="B",
                y="V",
                ax=ax,
                levels=[0.5],
                colors="k",
                linewidths=1.5,
                linestyles="-",
                zorder=1000,
            )

        pct_boundary = 100 * float(
            ds_sel.percentage_boundary.sel(zbp_cluster_number=i).squeeze()
        )
        print(rf"    {pct_boundary:.0f}% gapless boundary")

        gp = np.asarray(np.abs(ds_sel.gap) * clusters.sel(zbp_cluster_number=i))
        top_gap = np.percentile(gp[np.nonzero(gp)], 80.0)
        print(rf"    Top 20% percentile gap = {1e3*top_gap:.2g} ueV")
        median_gap = np.median(gp[np.nonzero(gp)])
        print(rf"    Median gap = {1e3*median_gap:.2g} ueV")

        dB, dV = np.diff(B)[0], np.diff(V)[0]
        Vc, Bc = center_of_mass(np.array(cluster.transpose("V", "B")))
        Bc = B[0] + dB * Bc
        Vc = V[0] + dV * Vc
        pct_boundary_shifts = pct_boundary_shifts or {}
        shift_x, shift_y = pct_boundary_shifts.get(i, (0, 0))
        axs[0].text(
            Bc + shift_x,
            Vc + shift_y,
            f"{pct_boundary:.0f}% gapless boundary",
            c="k",
            ha="center",
            va="center",
            size=14,
            zorder=1100,
        )

        mu_B = 9.274010078328e-24  # [J/T]
        e = 1.60217663e-19  # [C]

        volume_px = np.sum(cluster)
        volume_mVT = dB * 1e3 * dV * volume_px
        print("    Volume = %d pixels = %.2g mV*T" % (volume_px, volume_mVT))
        if "lever_arm" in device_info and "g_factor" in device_info:
            volume_ueV = (
                device_info["lever_arm"]
                * 0.5
                * mu_B
                * device_info["g_factor"]
                * volume_mVT
                * 1e6
                / e
            )
            print(
                "           = (%.2g ueV)^2 = %.2g (top 20%% percentile gap)^2"
                % (np.sqrt(volume_ueV), volume_ueV / (1e3 * top_gap) ** 2)
            )
        print("    Center of mass B = %.1f T" % Bc)

    if plunger_cut is not None:
        plunger_cut = np.atleast_1d(plunger_cut)
        for pc in plunger_cut:
            axs[1].axhline(y=pc, **line_cuts_kw, zorder=1050)
            print(rf"Plunger cut at {pc:g} V")

    if field_cut is not None:
        axs[1].axvline(x=field_cut, **line_cuts_kw, zorder=1050)
        print(rf"Field cut at {field_cut:g} T")

    axs[0].yaxis.set_label_text(r"$V_\mathrm{p}$ [V]")
    axs[0].xaxis.set_label_text("$B$ [T]")
    axs[0].set_title("")

    axs[1].yaxis.set_label_text(r"")
    axs[1].xaxis.set_label_text("$B$ [T]")
    axs[1].set_title("")

    for ax in axs:
        ax.set_ylim(plunger_lim)

    for i, ax in enumerate(axs):
        add_subfig_label(ax, "(" + "ab"[i] + ")", width=0.085, height=0.09)

    return fig, axs


def plot_conductance(
    zbp_ds: xr.Dataset,
    ds_left: xr.Dataset,
    ds_right: xr.Dataset,
    selected_cutter: int,
    selected_plunger: float,
    zbp_cluster_numbers: list[int],
    field_ticks: list[float] | None = None,
    bias_max: float | None = None,
    bias_ticks: tuple[float, float] | None = None,
    g_local_ticks: list[float] | None = None,
    g_local_max: float | None = None,
    g_nonlocal_ticks: list[float] | None = None,
    g_nonlocal_max: float | None = None,
    labels: str = "cdef",
):
    """Plot local and non-local conductance as a function of bias and field."""
    sel = {"V": selected_plunger, "cutter_pair_index": selected_cutter}
    g_L = ds_left.sel(sel, method="nearest")
    g_R = ds_right.sel(sel, method="nearest")
    if "cutter_pair_index" in zbp_ds.dims:
        zbp_ds = zbp_ds.sel(cutter_pair_index=selected_cutter)

    B = g_L.g_ll["B"].values
    lb = g_L.g_ll["left_bias"].values
    rb = g_R.g_rr["right_bias"].values

    n = 400
    cmap1 = mpl.cm.get_cmap("viridis", n + 1)
    cmap2 = mpl.cm.get_cmap("PuOr_r", n + 1)

    biases = [lb, rb, lb, rb]
    G = [g_L, g_R, g_L, g_R]
    xlabels = [
        r"Left bias [$\mu$V]",
        r"Right bias [$\mu$V]",
        r"Left bias [$\mu$V]",
        r"Right bias [$\mu$V]",
    ]
    clabels = [
        r"$G_\mathrm{LL}$ [$e^2/h$]",
        r"$G_\mathrm{RR}$ [$e^2/h$]",
        r"$A(G_\mathrm{RL})$ [$e^2/h$]",
        r"$A(G_\mathrm{LR})$ [$e^2/h$]",
    ]

    fig, axs = plt.subplots(1, 4, figsize=(4.2 * 4, 4.3), sharey=False)

    pcm = [None] * 4
    for i, (ds, ax, xlabel, clabel) in enumerate(zip(G, axs, xlabels, clabels)):
        side = "left" if "left_bias" in ds.dims else "right"
        other_side = "right" if side == "left" else "left"
        bias_name = f"{side}_bias"
        bias = ds[bias_name].values
        if i < 2:
            g = ds[f"g_{side[0]}{side[0]}"]
            cmap = cmap1
            vmin = 0
            if g_local_max is not None:
                vmax = g_local_max
            else:
                vmax = np.nanmax(
                    [np.nanmax(np.abs(ds.g_ll)), np.nanmax(np.abs(ds.g_rr))]
                )
            pcm[i] = ax.pcolormesh(
                B,
                1e3 * bias,
                g.T,
                shading="auto",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                linewidth=0,
                rasterized=True,
            )
        else:
            g = ds[f"g_{other_side[0]}{side[0]}"]
            gr = g.copy()
            gr[bias_name] = g[bias_name][::-1]
            g = 0.5 * (g - gr)
            cmap = cmap2
            if g_nonlocal_max is not None:
                vmin = -g_nonlocal_max
                vmax = g_nonlocal_max
            else:
                vmax = np.nanmax(
                    [np.nanmax(np.abs(g_L.g_rl)), np.nanmax(np.abs(g_R.g_lr))]
                )
                vmin = -vmax
            pcm[i] = ax.pcolormesh(
                B,
                1e3 * bias,
                g.T.squeeze(),
                vmin=vmin,
                vmax=vmax,
                shading="auto",
                cmap=cmap,
                linewidth=0,
                rasterized=True,
            )
            gap = ds.gap.where(ds.gap <= ds[bias_name][-2])
            ax.plot(gap["B"], 1e3 * np.array([-gap, gap]).T, c="black")

        pcm[i].set_edgecolor("face")

        if bias_max is not None:
            ax.set_ylim(-bias_max, bias_max)
        if bias_ticks is not None:
            ax.set_yticks(bias_ticks)

        cax = ax.inset_axes([0, 1.05, 1, 0.05], transform=ax.transAxes)
        exceed_min = int(np.nanmin(g) < vmin - 1e-8)
        exceed_max = int(np.nanmax(g) > vmax + 1e-8)
        if i < 2:
            exceed_min = 0
        cb = fig.colorbar(
            pcm[i],
            ax=ax,
            cax=cax,
            orientation="horizontal",
            ticklocation="top",
            extend={0: "neither", 1: "min", 2: "max", 3: "both"}[
                exceed_min + 2 * exceed_max
            ],
        )
        ax.set_xlabel("$B$ [T]")
        if field_ticks is not None:
            ax.set_xticks(field_ticks)
        ax.set_ylabel(xlabel)
        cb.set_label(clabel)
        if i < 2:
            if g_local_ticks is not None:
                cb.set_ticks(g_local_ticks)
        else:
            if g_nonlocal_ticks is not None:
                cb.set_ticks(g_nonlocal_ticks)

    if zbp_cluster_numbers:
        dim = "zbp_cluster_number"
        clusters = tgp.common.expand_clusters(zbp_ds.gapped_zbp_cluster, dim=dim)
        is_in_roi2 = clusters.sel(zbp_cluster_number=zbp_cluster_numbers).sum(dim=dim)
        is_in_roi2 = is_in_roi2.sel({"V": selected_plunger}, method="nearest")
        roi2i = np.argwhere(np.array(is_in_roi2) > 0.5)[:, 0]
        dB = np.diff(B)[0]
        if len(roi2i) > 0:
            roi2min = B[0] + dB * (np.min(roi2i) - 0.5)
            roi2max = B[0] + dB * (np.max(roi2i) + 0.5)
            for i, ax in enumerate(axs):
                bmax = 1e3 * float(np.max(biases[i]))
                for mm in [roi2min, roi2max]:
                    ax.plot(
                        [mm] * 2,
                        np.array([-1, 1]) * bmax,
                        c="w" if i < 2 else "k",
                        ls="--",
                        lw=1,
                    )

    for i, ax in enumerate(axs):
        add_subfig_label(ax, f"({labels[i]})", width=0.14, height=0.16)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35)

    return fig, axs


def plot_conductance_waterfall(
    zbp_ds: xr.Dataset,
    ds_left: xr.Dataset,
    ds_right: xr.Dataset,
    selected_cutter: int,
    selected_plunger: float,
    zbp_cluster_numbers: list[int],
    field_lim: tuple[float | None, float | None] = (None, None),
    bias_max: float = 0.1,
    bias_ticks: tuple[float, float] | None = None,
    scale_g_local: float = 0.15,
    gbar_local: float = 0.5,
    scale_g_nonlocal: float = 0.6,
    gbar_nonlocal: float = 0.1,
    g_bar_pos: float = -0.07,
    labels: str = "ghij",
) -> tuple[mpl.figure.Figure, list[mpl.axes.Axes]]:
    """Plot conductance waterfall.

    Raises
    ------
    ValueError
        If no clusters numbers are given.
    """
    if not zbp_cluster_numbers:
        raise ValueError("No cluster numbers given")
    if "cutter_pair_index" in zbp_ds.dims:
        zbp_ds = zbp_ds.sel(cutter_pair_index=selected_cutter)
    sel = {"V": selected_plunger, "cutter_pair_index": selected_cutter}
    g_L = ds_left.sel(sel, method="nearest")
    g_R = ds_right.sel(sel, method="nearest")
    G = [g_L, g_R, g_L, g_R]
    B = g_L.g_ll["B"].values
    xlabels = [
        r"Left bias [$\mu$V]",
        r"Right bias [$\mu$V]",
        r"Left bias [$\mu$V]",
        r"Right bias [$\mu$V]",
    ]
    clabels = [
        r"$G_\mathrm{LL}$ [$e^2/h$]",
        r"$G_\mathrm{RR}$ [$e^2/h$]",
        r"$A(G_\mathrm{RL})$ [$e^2/h$]",
        r"$A(G_\mathrm{LR})$ [$e^2/h$]",
    ]

    dim = "zbp_cluster_number"
    clusters = tgp.common.expand_clusters(zbp_ds.gapped_zbp_cluster, dim=dim)
    is_in_roi2 = clusters.sel(zbp_cluster_number=zbp_cluster_numbers).sum(dim=dim)
    is_in_roi2 = is_in_roi2.sel({"V": selected_plunger}, method="nearest")
    is_in_roi2 = np.array(is_in_roi2) > 0.5

    fig, axs = plt.subplots(1, 4, figsize=(4.2 * 4, 9), sharey=True)
    colors_local = plt.cm.viridis(np.linspace(0.15, 1.1, B.shape[0]))[::-1]
    colors_nonlocal = plt.cm.inferno(np.linspace(0.15, 0.9, B.shape[0]))[::-1]

    for i, (ds, ax, xlabel, clabel) in enumerate(zip(G, axs, xlabels, clabels)):
        side = "left" if "left_bias" in ds.dims else "right"
        other_side = "right" if side == "left" else "left"
        bias_name = f"{side}_bias"
        bias = ds[bias_name].values
        g = ds[f"g_{side[0]}{side[0]}"] if i < 2 else ds[f"g_{other_side[0]}{side[0]}"]

        colors = [colors_local, colors_nonlocal][i // 2]
        colors_w_roi2 = deepcopy(colors)
        colors_w_roi2[is_in_roi2, :] = np.array([0, 0, 0, 1])

        scale_g = [scale_g_local, scale_g_nonlocal][i // 2]
        gbar = [gbar_local, gbar_nonlocal][i // 2]

        for j, _B in enumerate(B):
            g_ = g.where(np.abs(bias) <= bias_max, np.nan)
            g_ = np.array(g_.sel(B=_B, method="nearest"))
            if i >= 2:
                g_ = 0.5 * (g_ - g_[::-1])

            y = _B + scale_g * g_
            ax.fill_between(1e3 * bias, _B, y, color=colors[j], alpha=0.1)
            ax.plot(1e3 * bias, y, c=colors_w_roi2[j], lw=2, marker=3, ms=2)
            y = [_B] * 2
            ax.plot(1e3 * np.array([-0.2, -bias_max * 1.08]), y, c=colors[j], lw=2)
            ax.plot(1e3 * np.array([bias_max * 1.08, 0.2]), y, c=colors[j], lw=2)

        if i < 2:
            ax.plot([0, 0], [0, np.max(B) + 0.05], "--", c="#cccccc", lw=1, zorder=1)
        else:
            gap = ds.gap.where(ds.gap <= ds[bias_name][-2])
            ax.scatter(1e3 * gap, gap["B"], c=colors, s=50, zorder=50)
            ax.scatter(-1e3 * gap, gap["B"], c=colors, s=50, zorder=50)

        if i == 0:
            ax.set_ylabel("$B$ [T]")

        ax.set_xlabel(xlabel)
        ax.set_xlim(-1e3 * bias_max * 1.15, 1e3 * bias_max * 1.15)
        if bias_ticks is not None:
            ax.set_xticks(bias_ticks)
        ax.set_ylim(field_lim)

        add_subfig_label(ax, f"({labels[i]})", width=0.12, height=0.05)

        if field_lim[1] is not None:
            x0 = 0.005
            y0 = field_lim[1] + g_bar_pos
            ax.text(1e3 * x0, y0, clabel, ha="right", va="center", size=14)
            ax.errorbar(
                1e3 * (x0 + 0.015),
                y0,
                yerr=0.5 * scale_g * gbar,
                c=colors[-1],
                ecolor=colors[-1],
                elinewidth=2,
                capsize=4,
            )
            ax.text(
                1e3 * (x0 + 0.02),
                y0,
                r"$%g\,e^2/h$" % gbar,
                ha="left",
                va="center",
                size=14,
            )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.07)

    return fig, axs


def plot_conductance_waterfall_plunger(
    zbp_ds: xr.Dataset,
    ds_left: xr.Dataset,
    ds_right: xr.Dataset,
    selected_cutter: int,
    selected_field: float,
    zbp_cluster_numbers: list[int],
    plunger_lim: tuple[float | None, float | None] = (None, None),
    bias_max: float = 0.1,
    bias_ticks: tuple[float, float] | None = None,
    scale_g_local: float = 0.0007,
    gbar_local: float | tuple[float, float] = 0.5,
    scale_g_nonlocal: float | tuple[float, float] = 0.006,
    gbar_nonlocal: float = 0.05,
    g_bar_pos: float = 0,
    labels: str = "abcd",
    text: str = "",
) -> tuple[mpl.figure.Figure, list[mpl.axes.Axes]]:
    """Plot conductance waterfall plot for a given field and cutter pair.

    Raises
    ------
    ValueError
        If no zbp_cluster_numbers is empty.
    """
    if not zbp_cluster_numbers:
        raise ValueError("No cluster numbers given")
    sel = {"B": selected_field, "cutter_pair_index": selected_cutter}
    if "cutter_pair_index" in zbp_ds.dims:
        zbp_ds = zbp_ds.sel(cutter_pair_index=selected_cutter)
    g_L = ds_left.sel(sel, method="nearest")
    g_R = ds_right.sel(sel, method="nearest")
    G = [g_L, g_R, g_L, g_R]
    V = g_L.g_ll["V"].values
    xlabels = [
        r"Left bias [$\mu$V]",
        r"Right bias [$\mu$V]",
        r"Left bias [$\mu$V]",
        r"Right bias [$\mu$V]",
    ]
    clabels = [
        r"$G_\mathrm{LL}$ [$e^2/h$]",
        r"$G_\mathrm{RR}$ [$e^2/h$]",
        r"$A(G_\mathrm{RL})$ [$e^2/h$]",
        r"$A(G_\mathrm{LR})$ [$e^2/h$]",
    ]

    dim = "zbp_cluster_number"
    clusters = tgp.common.expand_clusters(zbp_ds.gapped_zbp_cluster, dim=dim)
    is_in_roi2 = clusters.sel(zbp_cluster_number=zbp_cluster_numbers).sum(dim=dim)
    is_in_roi2 = is_in_roi2.sel({"B": selected_field}, method="nearest")
    is_in_roi2 = np.array(is_in_roi2) > 0.5

    fig, axs = plt.subplots(1, 4, figsize=(4.2 * 4, 9), sharey=True)
    colors_local = plt.cm.viridis(np.linspace(0.15, 0.8, V.shape[0]))
    colors_nonlocal = plt.cm.inferno(np.linspace(0.4, 0.9, V.shape[0]))

    for i, (ds, ax, xlabel, clabel) in enumerate(zip(G, axs, xlabels, clabels)):
        side = "left" if "left_bias" in ds.dims else "right"
        other_side = "right" if side == "left" else "left"
        bias_name = f"{side}_bias"
        bias = ds[bias_name].values
        g = ds[f"g_{side[0]}{side[0]}"] if i < 2 else ds[f"g_{other_side[0]}{side[0]}"]

        colors = [colors_local, colors_nonlocal][i // 2]
        colors_w_roi2 = deepcopy(colors)
        colors_w_roi2[is_in_roi2, :] = np.array([0, 0, 0, 1])

        scale_g = [scale_g_local, scale_g_nonlocal][i // 2]
        scale_g = scale_g if isinstance(scale_g, float) else scale_g[i % 2]
        gbar = [gbar_local, gbar_nonlocal][i // 2]
        gbar = gbar if isinstance(gbar, float) else gbar[i % 2]

        for j, _V in enumerate(V):
            g_ = g.where(np.abs(bias) <= bias_max, np.nan)
            g_ = np.array(g_.sel(V=_V, method="nearest"))
            if i >= 2:
                g_ = 0.5 * (g_ - g_[::-1])
            y = _V + scale_g * g_
            ax.fill_between(1e3 * bias, _V, y, color=colors[j], alpha=0.1)
            ax.plot(1e3 * bias, y, c=colors_w_roi2[j], lw=2, marker=3, ms=2)
            y = [_V] * 2
            ax.plot(1e3 * np.array([-0.2, -bias_max * 1.08]), y, c=colors[j], lw=2)
            ax.plot(1e3 * np.array([bias_max * 1.08, 0.2]), y, c=colors[j], lw=2)

        if i < 2:
            ax.plot([0, 0], [-5, V.max() + 0.0003], "--", c="#cccccc", lw=1, zorder=1)
        else:
            gap = ds.gap.where(ds.gap <= ds[bias_name][-2])
            ax.scatter(1e3 * gap, gap["V"], c=colors, s=50, zorder=50)
            ax.scatter(-1e3 * gap, gap["V"], c=colors, s=50, zorder=50)

        if i == 0:
            ax.set_ylabel(r"$V_\mathrm{p}$ [V]")

        ax.set_xlabel(xlabel)
        ax.set_xlim(-1e3 * bias_max * 1.15, 1e3 * bias_max * 1.15)
        if bias_ticks is not None:
            ax.set_xticks(bias_ticks)
        ax.set_ylim(plunger_lim)

        add_subfig_label(ax, f"({labels[i]})", width=0.12, height=0.05)

        if plunger_lim[1] is not None:
            x0 = 0.005
            y0 = plunger_lim[1] + g_bar_pos
            ax.text(1e3 * x0, y0, clabel, ha="right", va="center", size=14)
            ax.errorbar(
                1e3 * (x0 + 0.015),
                y0,
                yerr=0.5 * scale_g * gbar,
                c=colors[-1],
                ecolor=colors[-1],
                elinewidth=2,
                capsize=4,
            )
            ax.text(
                1e3 * (x0 + 0.02),
                y0,
                r"$%g\,e^2/h$" % gbar,
                ha="left",
                va="center",
                size=14,
            )

    if text:
        axs[0].text(
            0.08,
            1 - 0.095,
            text,
            ha="left",
            va="center",
            transform=axs[0].transAxes,
            fontsize=14,
        )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.07)

    return fig, axs


def _hex_to_rgb(color_hex: str) -> tuple[int, ...]:
    color_hex = color_hex.lstrip("#")
    lv = len(color_hex)
    return tuple(int(color_hex[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def plot_stage1_clusters(
    L_3w_ta: xr.DataArray,
    R_3w_ta: xr.DataArray,
    L_3w_tat: xr.DataArray,
    R_3w_tat: xr.DataArray,
    thp: float,
    *,
    n: int = 100,
    contour_color: str = "k",
    text: str = "",
    xy_text: tuple[float, float] = (0.04, 0.03),
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot stage 1 clusters."""
    orange_rgb = _hex_to_rgb("#ff7f0e")
    orange = np.ones((n, 4))
    orange[:, 0] = np.linspace(orange_rgb[0] / 256, 1, n)
    orange[:, 1] = np.linspace(orange_rgb[1] / 256, 1, n)
    orange[:, 2] = np.linspace(orange_rgb[2] / 256, 1, n)
    cmap = mpl.colors.ListedColormap(orange[::-1, :])

    B = np.array(L_3w_ta["B"])
    V = np.array(L_3w_ta["V"])
    dB = np.diff(B)[0]
    dV = np.diff(V)[0]

    reps = 20
    B1 = np.linspace(B.min() - dB / 2, B.max() + dB / 2, (B.size + 1) * reps)
    V1 = np.linspace(V.min() - dV / 2, V.max() + dV / 2, (V.size + 1) * reps)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(4.2 * 4, 6))

    pcm = L_3w_ta.plot.pcolormesh(
        x="B",
        y="V",
        ax=ax1,
        shading="nearest",
        infer_intervals=False,
        vmin=0,
        vmax=1,
        cmap=cmap,
        linewidth=0,
        rasterized=True,
        add_colorbar=False,
    )
    pcm.set_edgecolor("face")
    L_3w_tat.interp(B=B1, V=V1, method="nearest").plot.contour(
        x="B",
        y="V",
        ax=ax1,
        levels=[thp],
        colors=contour_color,
        linewidths=1.0,
    )
    ax1.set_xlabel("$B$ [T]")
    ax1.set_ylabel(r"$V_\mathrm{p}$ [V]")
    ax1.set_title("")

    pcm = R_3w_ta.plot.pcolormesh(
        x="B",
        y="V",
        ax=ax2,
        shading="nearest",
        infer_intervals=False,
        vmin=0,
        vmax=1,
        cmap=cmap,
        linewidth=0,
        rasterized=True,
        add_colorbar=False,
    )
    pcm.set_edgecolor("face")

    R_3w_tat.interp(B=B1, V=V1, method="nearest").plot.contour(
        x="B",
        y="V",
        ax=ax2,
        levels=[thp],
        colors=contour_color,
        linewidths=1.0,
    )
    ax2.set_xlabel("$B$ [T]")
    ax2.set_ylabel("")
    ax2.set_title("")

    pcm = (L_3w_ta * R_3w_ta).plot.pcolormesh(
        x="B",
        y="V",
        ax=ax3,
        shading="nearest",
        infer_intervals=False,
        vmin=0,
        vmax=1,
        cmap=cmap,
        linewidth=0,
        rasterized=True,
        add_colorbar=False,
    )
    pcm.set_edgecolor("face")
    cont = (
        (L_3w_tat * R_3w_tat)
        .interp(B=B1, V=V1, method="nearest")
        .plot.contour(
            x="B",
            y="V",
            ax=ax3,
            levels=[thp],
            colors=contour_color,
            linewidths=1.0,
        )
    )
    ax3.set_xlabel("$B$ [T]")
    ax3.set_ylabel("")
    ax3.set_title("")

    axins = inset_axes(
        ax3,
        width="7%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0.0, 1, 1),
        bbox_transform=ax3.transAxes,
        borderpad=0,
    )
    cb = fig.colorbar(pcm, cax=axins)
    cb.add_lines(cont)
    cb.set_label("ZBP cutter fraction")
    if ylim:
        ax3.set_ylim(ylim)

    titles = ["Left", "Right", "Joint ZBP cutter fraction"]
    for i, ax in enumerate([ax1, ax2, ax3]):
        label = "abc"[i]
        add_subfig_label(ax, f"({label})", width=0.1, height=0.08)
        ax.text(
            0.04,
            0.85,
            titles[i],
            c="k",
            ha="left",
            va="bottom",
            size=14,
            transform=ax.transAxes,
        )

    ax1.text(
        *xy_text,
        text,
        c="k",
        ha="left",
        va="bottom",
        size=14,
        transform=ax1.transAxes,
    )

    plt.subplots_adjust(wspace=0.1)

    return fig, (ax1, ax2, ax3)
