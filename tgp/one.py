# Copyright (c) Microsoft Corporation. All rights reserved.

from __future__ import annotations

import itertools
import math
from typing import Any, Literal, NamedTuple

from IPython.display import display
import joblib
import numpy as np
import scipy.ndimage
import xarray as xr

from tgp.common import (
    CLUSTER_INFO_VARIABLES,
    ThresholdException,
    cluster_info,
    ds_requires,
    expand_clusters,
    groupby_map,
    set_clusters_of,
)

memory = joblib.Memory("cachedir")


def print_lim(da: xr.DataArray, lim: float) -> None:
    """Print the mean and max of the array divided by lim."""
    x = np.mean(np.abs(da)).item() / lim
    print(f"mean(|{da.name}|) / lim = {x}")
    y = np.max(np.abs(da)).item() / lim
    print(f"max(|{da.name}|)  / lim = {y}")


def _noise_level_threshold(da: xr.DataArray) -> xr.DataArray:
    μ = da.mean(dim=("B", "V"))
    σ = da.std(dim=("B", "V"))
    th = μ + 2 * σ
    th.name = "thresholds"
    return th


def _threshold_2w(
    da: xr.DataArray,
    # Based on: pre-selected region in B, V
    B_max: float | None,
    V_max: float | None,
    # Based on: automatically determine noise in several regions
    n_tiles: int = 100,
    percentile: int = 33,
    # Based on: directly setting the threshold
    th_2w: float | xr.DataArray | None = None,
    verbose: bool = False,
) -> xr.DataArray:
    if B_max is not None and V_max is not None:
        sel = da.where((da.B < B_max) & (da.V < V_max))
        th_2w = _noise_level_threshold(sel)
    if th_2w is None:
        th_2w = auto_extract_threshold_2w(da, n_tiles, percentile)
    if verbose:
        display(th_2w)
    return (da > th_2w).astype(float)


@ds_requires(
    variables=("L_2w_nl", "R_2w_nl"),
    dims=("B", "V", "cutter_pair_index"),
    sets_variables=("L_2w_nl_t", "R_2w_nl_t", "LR_2w_nl_t", "L_2w_nl_ta", "R_2w_nl_ta"),
)
def set_2w_th(
    ds: xr.Dataset,
    B_max: float | None = None,
    V_max: float | None = None,
    n_tiles: int = 100,
    percentile: int = 33,
    th_2w: float
    | xr.DataArray
    | tuple[float, float]
    | tuple[xr.DataArray, xr.DataArray]
    | None = None,
    verbose: bool = False,
) -> None:
    """Set 2w threshold.

    Either B_max and V_max or th_2w or n_tiles and percentile must be specified.

    Parameters
    ----------
    ds
        Full Stage 1 ``xarray.Dataset``.
    B_max
        Maximum value in B, used to determine the threshold.
    V_max
        Maximum value in V, used to determine the threshold.
    n_tiles
        Number of tiles that the data will be binned into.
    percentile
        Percentile of the histogram of sorted noise of
        the tiles, which determines the threshold.
    th_2w
        The value of the threshold that can be passed directly.
        Might be a tuple of values for the left and right side.
    verbose
        Show the thresholds before applying them.
    """
    prefix = "set_2w_th"
    ds.attrs.update(
        {
            f"{prefix}.B_max": B_max,
            f"{prefix}.V_max": V_max,
            f"{prefix}.n_tiles": n_tiles,
            f"{prefix}.percentile": percentile,
            f"{prefix}.th_2w": th_2w,
        }
    )
    # By providing th_2w as a tuple, it allows to manually set
    # thresholds for left and right data set.
    if not isinstance(th_2w, tuple):
        th_2w = (th_2w, th_2w)

    ds["L_2w_nl_t"], ds["R_2w_nl_t"] = (
        _threshold_2w(da, B_max, V_max, n_tiles, percentile, th, verbose)
        for da, th in zip((ds.L_2w_nl, ds.R_2w_nl), th_2w)
    )
    ds["LR_2w_nl_t"] = ds.L_2w_nl_t * ds.R_2w_nl_t
    ds["L_2w_nl_ta"] = ds.L_2w_nl_t.mean(dim="cutter_pair_index")
    ds["R_2w_nl_ta"] = ds.R_2w_nl_t.mean(dim="cutter_pair_index")


@ds_requires(
    variables=("L_2w_nl_ta", "R_2w_nl_ta"),
    sets_variables=("L_gapped", "R_gapped", "gapped"),
)
def set_gapped(
    ds: xr.Dataset, th_2w_p: float, method: Literal["structure", "fair"] = "structure"
) -> None:
    """Set the "gapped" array in the dataset.

    Parameters
    ----------
    ds
        Full Stage 1 ``xarray.Dataset``.
    th_2w_p
        The threshold for the ``L_2w_nl_ta`` and ``R_2w_nl_ta`` arrays.
    method
        "structure" or "fair".

    Raises
    ------
    NotImplementedError
        When using an invalid method.
    """
    ds["L_gapped"] = ds.L_2w_nl_ta > th_2w_p
    ds["R_gapped"] = ds.R_2w_nl_ta > th_2w_p
    prefix = "set_gapped"
    ds.attrs.update({f"{prefix}.th_2w_p": th_2w_p, f"{prefix}.method": method})

    if method == "structure":  # more structure is visible
        ds["gapped"] = 1.0 - 1.0 * ds.L_gapped * ds.R_gapped
    elif method == "fair":  # more fair way
        ds["gapped"] = 1.0 * (~ds.L_gapped) * (~ds.R_gapped)
    else:
        raise NotImplementedError(f"{method} is not an option.")


@ds_requires(
    variables=("L_3w", "R_3w"),
    sets_variables=("L_3w_t", "R_3w_t", "LR_3w_t", "L_3w_ta", "R_3w_ta"),
)
def set_3w_th(
    ds: xr.Dataset,
    th_3w: float
    | xr.DataArray
    | tuple[float, float]
    | tuple[xr.DataArray, xr.DataArray]
    | None = None,
) -> None:
    """Set 3w threshold.

    Parameters
    ----------
    ds
        Full Stage 1 ``xarray.Dataset``.
    th_3w
        The value of the 3w threshold which can be a float or array or a tuple
        of them when applying different threshold to the left and right side.

    Raises
    ------
    ThresholdException
        When the combined thresholded left and right data does
        not yield any True values.
    """
    ds.attrs["set_3w_th.th_3w"] = th_3w
    # By providing th_3w as a tuple, it allows to manually set
    # thresholds for left and right data set.
    if not isinstance(th_3w, tuple):
        th_3w = (th_3w, th_3w)

    ds["L_3w_t"] = (ds.L_3w < -th_3w[0]).astype(float)
    ds["R_3w_t"] = (ds.R_3w < -th_3w[1]).astype(float)
    ds["LR_3w_t"] = ds.L_3w_t * ds.R_3w_t
    if ds.LR_3w_t.sum() == 0:
        raise ThresholdException("'th_3w' too high, no overlapping clusters.")
    ds["L_3w_ta"] = ds.L_3w_t.mean(dim="cutter_pair_index")
    ds["R_3w_ta"] = ds.R_3w_t.mean(dim="cutter_pair_index")


@ds_requires(
    variables=("L_3w_ta", "R_3w_ta"),
    sets_variables=("L_3w_tat", "R_3w_tat", "LR_3w_tat"),
)
def set_3w_tat(ds: xr.Dataset, th_3w_tat: float) -> None:
    """Set the 3w threshold on the left and right thresholded and averaged
    3w data (``L_3w_ta`` and ``R_3w_ta``).

    Parameters
    ----------
    ds
        Full Stage 1 ``xarray.Dataset``.
    th_3w_tat
        The threshold value applied to the left and right thresholded and averaged data.

    Raises
    ------
    ThresholdException
        When the combined thresholded left and right data does
        not yield any True values.
    """
    ds["L_3w_tat"] = (ds.L_3w_ta > th_3w_tat).astype(float)
    ds["R_3w_tat"] = (ds.R_3w_ta > th_3w_tat).astype(float)
    ds["LR_3w_tat"] = ds.L_3w_tat * ds.R_3w_tat
    if ds.LR_3w_tat.sum() == 0:
        raise ThresholdException("'th_3w_tat' too high, no overlapping clusters.")
    ds.attrs["set_3w_tat.th_3w_tat"] = th_3w_tat


@ds_requires(variables=("LR_3w_tat",), dims=("B", "V"), sets_variables=("clusters",))
def set_clusters(
    ds: xr.Dataset,
    *,
    min_samples: int = 3,
    xi: float = 0.1,
    min_cluster_size: float = 0.01,
    max_eps: float = 10.0,
) -> None:
    """Set the ``clusters`` array in the dataset.

    Parameters
    ----------
    ds
        Full Stage 1 ``xarray.Dataset``.
    min_samples
        Argument passed to `sklearn.cluster.OPTICS`.
    xi
        Argument passed to `sklearn.cluster.OPTICS`.
    min_cluster_size
        Argument passed to `sklearn.cluster.OPTICS`.
    max_eps
        Argument passed to `sklearn.cluster.OPTICS`.
    """
    set_clusters_of(
        ds,
        "LR_3w_tat",
        "clusters",
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size,
        max_eps=max_eps,
    )
    prefix = "set_clusters"
    ds.attrs.update(
        {
            f"{prefix}.min_samples": min_samples,
            f"{prefix}.xi": xi,
            f"{prefix}.min_cluster_size": min_cluster_size,
            f"{prefix}.max_eps": max_eps,
        }
    )


@ds_requires(
    variables=("clusters",), dims=("B", "V"), sets_variables=CLUSTER_INFO_VARIABLES
)
def set_cluster_info(ds: xr.Dataset) -> None:
    """Set the ``cluster_info`` array in the dataset.

    Sets the following arrays with dimension "zbp_cluster_number":
    - ``cluster_B_min``
    - ``cluster_B_max``
    - ``cluster_V_min``
    - ``cluster_V_max``
    - ``cluster_B_center``
    - ``cluster_V_center``
    - ``cluster_B_size``
    - ``cluster_V_size``
    """
    # The dimension might already exist, so we remove it first.
    existing_dims = [
        k for k, v in ds.data_vars.items() if "zbp_cluster_number" in v.dims
    ]
    if existing_dims:
        # We cannot do this because we first need to remove the data variables
        # that depend on the dimension but we cannot do that because we rely on
        # inplace operations.
        raise ValueError("Cluster infos have already been set once, cannot redo it.")
    clusters = expand_clusters(ds.clusters, dim="zbp_cluster_number")
    info = groupby_map(cluster_info, clusters, ("B", "V"), as_xarray=True, pct_box=0)
    for k, v in info.data_vars.items():
        ds[k] = v


def get_thresholds(attrs: dict[str, Any]) -> dict[str, Any]:
    """Get the thresholds used in the dataset."""
    keys = [
        "set_2w_th",
        "set_gapped",
        "set_3w_th",
        "set_3w_tat",
        "set_clusters",
    ]
    sel_attrs = {
        k: v for k, v in attrs.items() if any(k.startswith(key) for key in keys)
    }
    thresholds = {}
    for k, v in sel_attrs.items():
        func, name = k.split(".", 1)
        thresholds.setdefault(func, {})[name] = v
    return thresholds


@ds_requires(
    variables=("L_2w_nl", "R_2w_nl", "L_3w", "R_3w"),
    dims=("B", "V", "cutter_pair_index"),
    sets_variables=(
        set_2w_th.__sets_variables__
        + set_gapped.__sets_variables__
        + set_3w_th.__sets_variables__
        + set_3w_tat.__sets_variables__
        + set_clusters.__sets_variables__
    ),
)
def analyze(
    ds: xr.Dataset,
    thresholds: dict[str, Any],
    force: bool = False,
) -> xr.Dataset:
    """Perform the entire stage 1 analysis on a prepared dataset.

    Parameters
    ----------
    ds
        Full Stage 1 ``xarray.Dataset``.
    thresholds
        Dictionary of thresholds to use.
        Use :func:`get_thresholds` to get the thresholds from a dataset.
        Or pass for example:

        >>> {'set_2w_th': {'B_max': None,
        ...  'V_max': None,
        ...  'n_tiles': 100,
        ...  'percentile': 33,
        ...  'th_2w': None},
        ...  'set_gapped': {'th_2w_p': 0.5, 'method': 'structure'},
        ...  'set_3w_th': {'th_3w': 10000000.0},
        ...  'set_3w_tat': {'th_3w_tat': 0.5},
        ...  'set_clusters': {'min_samples': 3,
        ...  'xi': 0.1,
        ...  'min_cluster_size': 0.01,
        ...  'max_eps': 10.0}}

    force : bool, optional
        Redo the analysis even if the analysis has already
        been performed with the same thresholds.

    Returns
    -------
    xr.Dataset
        Dataset with new ``xarray.DataArray``s set.
    """
    th = thresholds
    th_old = get_thresholds(ds.attrs)

    def needs_update(key):
        return True if key not in th else th_old.get(key) != th[key]

    if force or needs_update("set_2w_th"):
        set_2w_th(ds, **th.get("set_2w_th", {}))
        force = True

    if force or needs_update("set_gapped"):
        set_gapped(ds, **th.get("set_gapped", {}))
        force = True

    if force or needs_update("set_3w_th"):
        set_3w_th(ds, **th.get("set_3w_th", {}))
        force = True

    if force or needs_update("set_3w_tat"):
        set_3w_tat(ds, **th.get("set_3w_tat", {}))
        force = True

    if force or needs_update("set_clusters"):
        set_clusters(ds, **th.get("set_clusters", {}))
    return ds


def set_defaults(ds: xr.Dataset, defaults: dict[str, float]):
    """Set default values for thresholds in the ``ds.attrs`` dictionary."""
    defaults = {
        k: v for k, v in defaults.items() if v is not None and not math.isnan(v)
    }
    fmt = {
        "set_gapped": ["th_2w_p"],
        "set_3w_th": ["th_3w"],
        "set_3w_tat": ["th_3w_tat"],
        "set_2w_th": ["B_max", "V_max", "th_2w"],
    }
    for func, names in fmt.items():
        for name in names:
            if name in defaults:
                ds.attrs[f"{func}.{name}"] = defaults[name]


def _calc_columns_rows(n: int) -> tuple[int, int]:
    num_columns = int(math.ceil(math.sqrt(n)))
    num_rows = int(math.ceil(n / float(num_columns)))
    return (num_columns, num_rows)


class _Tile(NamedTuple):
    array: np.ndarray
    number: int
    position: tuple[int, int]
    coords: tuple[float, float]


def _slice_to_tiles(ar: np.ndarray, n_tiles: int) -> list[_Tile]:
    w, h, *_ = ar.shape
    columns, rows = _calc_columns_rows(n_tiles)
    tile_w, tile_h = int(math.floor(w / columns)), int(math.floor(h / rows))
    tiles = []
    for number, (pos_y, pos_x) in enumerate(
        itertools.product(range(0, h - rows, tile_h), range(0, w - columns, tile_w)),
        start=1,
    ):
        x0, y0, x1, y1 = (pos_x, pos_y, pos_x + tile_w, pos_y + tile_h)
        _slice = ar[x0:x1, y0:y1]
        position = (
            int(math.floor(pos_x / tile_w)) + 1,
            int(math.floor(pos_y / tile_h)) + 1,
        )
        coords = (pos_x, pos_y)
        tile = _Tile(_slice, number, position, coords)
        tiles.append(tile)
    return tiles


def auto_extract_threshold_2w(
    da: xr.DataArray, n_tiles: int = 100, percentile: int = 10
):
    """Split up (B vs V) images in n tiles and calculate its threshold.

    Then order and take the value at the $m$th percentile.
    """
    tiles = _slice_to_tiles(da.transpose(..., "cutter_pair_index"), n_tiles)
    arrays = [_noise_level_threshold(t.array) for t in tiles]
    lst = sorted(arrays, key=np.sum)
    th = xr.concat(lst, "slices")
    return th.isel(slices=(len(arrays) * percentile) // 100, drop=True)


@ds_requires(variables=("clusters",), dims=("B", "V"))
def get_zoomin_ranges(
    ds: xr.Dataset, n_clusters: int, zoomin_V_height: float
) -> list[dict[str, tuple[float, float]]]:
    """Get the zoomin ranges for each cluster.

    Use with ``tgp.plot.one.plot_zoomin_ranges``.
    """
    zoomin_ranges = []
    dV = zoomin_V_height * 0.5
    data_clusters = expand_clusters(ds.clusters)
    n_clusters = min(n_clusters, len(data_clusters))
    for i in range(n_clusters):
        com = scipy.ndimage.center_of_mass(data_clusters[i].values)
        bc, vc = np.round(com).astype(int)
        B_range = [ds.B.min(), ds.B.max()]
        V_mid = ds.V[vc].item()
        V_range = [
            max(ds.V.min(), V_mid - dV),
            min(V_mid + dV, ds.V.max()),
        ]
        zoomin_ranges.append({"B": B_range, "V": V_range})
    return zoomin_ranges
