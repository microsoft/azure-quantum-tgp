# Copyright (c) Microsoft Corporation. All rights reserved.
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from IPython.display import display
import numpy as np
import pandas as pd
from paper_figures import load_cached_broadened
from scipy.constants import physical_constants
import scipy.stats
from tqdm.auto import tqdm
import xarray as xr

import tgp


def _roi1(ds):
    cluster_infos = tgp.common.cluster_infos(
        ds.clusters,
        "B",
        "V",
        # Minimal distance around box, at least some % or physical distance
        pct_box=10,
        min_margin_box=(0.003, 0.2),
    )
    bboxes = [info.bounding_box for info in cluster_infos]
    x1, y1, x2, y2 = np.array(bboxes).T
    gx1, gy1, gx2, gy2 = (
        ds.coords["V"].values[0],
        ds.coords["B"].values[0],
        ds.coords["V"].values[-1],
        ds.coords["B"].values[-1],
    )
    return dict(
        V_min=max(x1.min(), gx1),
        V_max=min(x2.max(), gx2),
        B_min=max(y1.min(), gy1),
        B_max=min(y2.max(), gy2),
    )


def analyze_1(fn, T_mK, *, return_ds: bool = False):
    thresholds = dict(
        set_gapped={"th_2w_p": 0.5},
        set_3w_th={"th_3w": 1e3},
        set_3w_tat={"th_3w_tat": 0.7},
    )
    ds = xr.load_dataset(fn)
    ds = tgp.prepare.prepare_sim(ds, T_mK=T_mK)
    r = dict(ds.attrs)
    try:
        tgp.one.analyze(ds, thresholds)
    except tgp.one.ThresholdException:
        return r
    else:
        roi1 = _roi1(ds)
        r.update(roi1)
        if return_ds:
            r["ds"] = ds
        return r


@dataclass
class Thresholds:
    volume_threshold: float
    gap_threshold_factor: float
    gap_threshold: float = 10e-3
    noise_threshold: float = 1e-4
    cutter_threshold: float = 0.5
    percentage_boundary_threshold: float = 0.6
    gap_threshold_high: float = 0.067

    @classmethod
    def from_ds_attrs(cls, ds: xr.Dataset) -> Thresholds:
        if ds.sample_name == "simulated_SLG_beta":
            lever_arm = 77.8  # meV/V
            g_factor = 4.11
        elif ds.sample_name == "simulated_DLG_epsilon":
            lever_arm = 85  # meV/V
            g_factor = 5.1
        else:
            raise Exception("Unknown device name.")
        μ_B = physical_constants["Bohr magneton in eV/T"][0]
        E_Z = 0.5 * g_factor * μ_B * 1e3
        # convert 20 ueV^2 to V*T
        volume_threshold = cls.gap_threshold**2 / E_Z / lever_arm

        # Optimal threshold factors for the simulated devices
        gap_thresholds = {0.0: 0.001, 0.1: 0.01, 1.0: 0.05, 2.7: 0.05, 4.0: 0.05}
        gap_threshold_factor = gap_thresholds.get(ds.surface_charge)
        if gap_threshold_factor is None:
            raise Exception(f"Unknown surface_charge ({ds.surface_charge}).")

        return cls(
            volume_threshold=volume_threshold,
            gap_threshold_factor=gap_threshold_factor,
        )


def get_roi2_stats(
    zbp_ds: xr.Dataset, cutter_threshold: float
) -> tuple[list[str], dict[str, int]]:
    results = []
    cnt = {"not passed": 0, "true positive": 0, "false positive": 0}
    if zbp_ds.gapped_zbp_cluster.max() == 0:
        return results, cnt
    roi2s = tgp.common.expand_clusters(zbp_ds["roi2"], dim="roi2")
    for roi2 in roi2s:
        n = zbp_ds.ncutters.where(zbp_ds.cluster_sets == roi2.roi2).max().item()
        if n < cutter_threshold * zbp_ds.dims["cutter_pair_index"]:
            result = "not passed"
        elif (zbp_ds.SI * roi2).any():
            result = "true positive"
        else:
            result = "false positive"
        results.append(result)
        cnt[result] += 1

    return results, cnt


def _soi_stats(zbp_ds: xr.Dataset, cluster: xr.DataArray) -> dict[str, float | int]:
    sel = dict(
        zbp_cluster_number=cluster.zbp_cluster_number.item(),
        cutter_pair_index=cluster.cutter_pair_index.item(),
    )
    ds_sel = zbp_ds.sel(sel)
    keys = {
        "top_quintile_gap",
        "cluster_volume",
        "median_gap",
        "cluster_B_center",
        "cluster_V_center",
        "percentage_boundary",
        "ncutters",
        "cluster_B_size",
        "cluster_V_size",
    }
    r = {k: ds_sel[k].item() for k in keys}
    r["cluster_volume"] *= 1e3
    r.update(sel)
    return r


def get_soi2_stats(zbp_ds: xr.Dataset) -> dict[tuple[int, int], dict[str, float | int]]:
    stats = [
        _soi_stats(zbp_ds, cluster)
        for cl in zbp_ds.gapped_zbp_cluster.transpose("cutter_pair_index", ...)
        for cluster in tgp.common.expand_clusters(cl, dim="zbp_cluster_number")
    ]
    return {
        (stat["cutter_pair_index"], stat["zbp_cluster_number"]): stat for stat in stats
    }


def analyze_2(
    ds_or_fname: xr.Dataset | str | Path,
    T_mK: float,
    B_max: float | None = None,
    force: bool = False,
    return_datasets: bool = True,
) -> dict[str, Any]:
    ds = (
        ds_or_fname
        if isinstance(ds_or_fname, xr.Dataset)
        else load_cached_broadened(ds_or_fname, T_mK, force=force, folder="cached")
    )
    if B_max is not None:
        sel = ds.B <= B_max
        if sel.sum() <= 1:
            return None
        ds = ds.sel(B=sel)

    th = Thresholds.from_ds_attrs(ds)
    ds_left = ds.rename({"bias": "left_bias"})
    ds_right = ds.rename({"bias": "right_bias"})
    ds_left, ds_right = tgp.two.extract_gap(
        ds_left,
        ds_right,
        gap_threshold_factor=th.gap_threshold_factor,
        noise_threshold=th.noise_threshold,
    )

    zbp_ds = tgp.two.zbp_dataset_derivative(
        ds_left, ds_right, average_over_cutter=False
    )
    tgp.two.set_gap_threshold(zbp_ds, threshold_high=th.gap_threshold_high)
    zbp_ds = tgp.two.cluster_and_score(
        zbp_ds,
        cluster_gap_threshold=th.gap_threshold,
        cluster_percentage_boundary_threshold=th.percentage_boundary_threshold,
        cluster_volume_threshold=th.volume_threshold,
    )
    passing_list, roi2_stats = get_roi2_stats(zbp_ds, th.cutter_threshold)
    soi2_stats = get_soi2_stats(zbp_ds)
    overlapping_clusters = tgp.two.cluster_sets_to_cluster_pairs(
        zbp_ds.cluster_sets, mode="dimension"
    )

    result = {
        "roi2_stats": roi2_stats,
        "soi2_stats": soi2_stats,
        "passing_list": passing_list,
        "overlapping_clusters": overlapping_clusters,
        **asdict(th),
        **ds.attrs,
    }
    if return_datasets:
        result["zbp_ds"] = zbp_ds
        result["ds_left"] = ds_left
        result["ds_right"] = ds_right
    return result


def clopper_pearson(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Estimate the confidence interval for a sampled Bernoulli random
    variable.

    Parameters
    ----------
    x
        The number of successes.
    n
        The number trials (x <=n).
    alpha
        The confidence level (i.e., the true probability is
        inside the confidence interval with probability 1-alpha).

    Returns
    -------
    A `(low, high)` pair of numbers indicating the interval
    on the probability.
    """
    assert x <= n
    b = scipy.stats.beta.ppf
    lo = b(alpha / 2, x, n - x + 1)
    hi = b(1 - alpha / 2, x + 1, n - x)
    return 0.0 if np.isnan(lo) else lo, 1.0 if np.isnan(hi) else hi


def parallel_map(f: Callable, seq, max_workers: int | None = None) -> list[Any]:
    with ProcessPoolExecutor(max_workers) as ex:
        return list(tqdm(ex.map(f, seq), total=len(seq)))


def show_region(
    df: pd.DataFrame, caption: str, cols: list[str], groupby: list[str]
) -> None:
    table = df.groupby(groupby)[cols].aggregate(sum)
    total = table["true positive"] + table["false positive"]
    successes = table["false positive"]
    lo, hi = np.vectorize(clopper_pearson)(successes, total, 0.05)
    table["confidence interval_low"] = lo.round(3) * 1e2
    table["confidence interval_high"] = hi.round(3) * 1e2
    table = table.style.set_caption(caption).format(precision=1)
    display(table)


def show_device(
    table_roi2: pd.DataFrame, caption: str, cols: list[str], groupby: list[str]
) -> None:
    device = {}
    for i, gr in table_roi2.groupby(groupby):
        mask = gr[["true positive", "false positive"]] > 0
        mask["not passed"] = False  # if no positives, it is not passed
        mask.loc[mask.sum(axis=1) == 0, "not passed"] = True
        device[i] = mask.sum().to_dict()
        device[i]["total"] = len(mask)
    table = pd.DataFrame(device).T
    table = table.style.set_caption(caption).format(precision=2)
    display(table)


def show_roi2_tables(
    df_stats: pd.DataFrame, groupby: list[str] = ["sample_name", "surface_charge"]
) -> None:
    cols = ["true positive", "false positive", "not passed"]
    table_roi2 = df_stats.join(df_stats.roi2_stats.apply(pd.Series))
    show_region(table_roi2, "Statistics per ROI2.", cols, groupby)
    show_device(table_roi2, "Statistics per ROI2 per device.", cols, groupby)


def show_soi2_tables(
    df_stats: pd.DataFrame, groupby: list[str] = ["sample_name", "surface_charge"]
) -> None:
    rows = []
    for _, row in df_stats.iterrows():
        # Only take the clusters that are in a passed ROI2
        for passed, overlaps in zip(row.passing_list, row.overlapping_clusters):
            if "positive" in passed:  # take both True and False positives
                rows.extend(
                    dict(row.to_dict(), **row.soi2_stats[pair]) for pair in overlaps
                )
    cluster_stats = pd.DataFrame(rows)

    cols = [
        "top_quintile_gap",
        "cluster_volume",
        "median_gap",
        "percentage_boundary",
        "ncutters",
        "cluster_B_center",
        "cluster_V_center",
    ]

    for name, func in [
        ("mean", np.mean),
        ("median", np.median),
        ("standard deviation", np.std),
    ]:
        table = (
            cluster_stats.groupby([*groupby, "gap_threshold_factor"])[cols]
            .aggregate(func)
            .round(3)
        )
        table = table.style.set_caption(
            f"Statistics per ZBP cluster that pass: {name}"
        ).format(precision=3)
        display(table)
