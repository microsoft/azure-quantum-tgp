# Copyright (c) Microsoft Corporation. All rights reserved.

from __future__ import annotations

import itertools
from typing import Any, Literal, NamedTuple, Tuple
import warnings

import numpy as np
import scipy.ndimage
import scipy.signal
from scipy.signal import savgol_filter
import xarray as xr

import tgp
from tgp.common import (
    CLUSTER_INFO_VARIABLES,
    ds_requires,
    expand_clusters,
    flatten_clusters,
    get_metadata,
    groupby_map,
    xr_map,
)

try:
    from typing import TypeAlias
except ImportError:

    from typing_extensions import TypeAlias


def _smooth_deriv_per_side(
    ds: xr.Dataset,
    side: str = "left",
    average_over_cutter: bool = True,
    bias_window: float = 3e-2,
    deriv: int = 2,
    polyorder: int = 2,
) -> xr.DataArray:
    other_side = {"left": "right", "right": "left"}[side]
    bias = f"{side}_bias"
    other_bias = f"{other_side}_bias"
    g_da = ds[f"g_{side[0]}{side[0]}"].load()

    # Calculate window_length using the bias axis
    dims = list(g_da.dims)
    axis = dims.index(bias)
    delta = np.mean(np.diff(ds[bias]))
    if np.max(np.abs(ds[bias])) > bias_window:
        window_length = int(2 * bias_window / delta)
    else:
        raise ValueError(
            "Bias window must be smaller or equal than the bias range in the data"
        )
    if window_length % 2 == 0:
        window_length += 1

    g_smooth = xr.apply_ufunc(
        savgol_filter,
        g_da,
        window_length,
        polyorder,
        deriv,
        delta,
        axis,
        input_core_dims=[dims, [], [], [], [], []],
        output_core_dims=[dims],
    )
    g_2nd_deriv = g_smooth.sel({bias: 0.0}, method="nearest", drop=True)
    if other_bias in g_2nd_deriv.dims:
        g_2nd_deriv = g_2nd_deriv.squeeze(other_bias, drop=True)
    return g_2nd_deriv.mean("cutter_pair_index") if average_over_cutter else g_2nd_deriv


@ds_requires(sets_variables=("left", "right", "zbp"))
def zbp_dataset_derivative(
    ds_left: xr.Dataset,
    ds_right: xr.Dataset,
    derivative_threshold: float = 2 * 0.5 / 0.05**2 * 1 / 4,
    zbp_probability_threshold: float = 0.6,
    bias_window: float = 10e-3,
    polyorder: int = 2,
    average_over_cutter: bool = True,
) -> xr.Dataset:
    """
    Create new `xarray.Dataset` containing ZBP maps.

    Has ZBP on left and right sides, and a joint map indicating
    where ZBPs are present with a probability higher than a certain threshold.

    Using the `scipy.signal.savgol_filter` to approximate the 2nd derivative.

    Parameters
    ----------
    ds_left
        Dataset of the left side.
    ds_right
        Dataset of the right side.
    derivative_threshold : float
        Minimum derivative per side.
    zbp_probability_threshold:
        Minimum probability to pass the joint ZBP test.
    bias_window, polyorder
        See `_smooth_deriv_per_side`.
    average_over_cutter:
        Decides whether to average over cutter or not

    Returns
    -------
    xarray.Dataset
        A dataset containing information about the postions of ZBPs.
    """
    kwargs = dict(
        bias_window=bias_window, polyorder=polyorder, average_over_cutter=False
    )
    left = _smooth_deriv_per_side(ds_left, side="left", **kwargs)
    right = _smooth_deriv_per_side(ds_right, side="right", **kwargs)
    P_left = (left <= -derivative_threshold).squeeze()
    P_right = (right <= -derivative_threshold).squeeze()
    if average_over_cutter and ("cutter_pair_index" in P_left.indexes):
        P_left = P_left.mean("cutter_pair_index")
        P_right = P_right.mean("cutter_pair_index")
    zbp_ds = xr.Dataset(
        {
            "left": P_left,
            "right": P_right,
            "zbp": (P_right >= zbp_probability_threshold)
            * (P_left >= zbp_probability_threshold),
        }
    )
    zbp_ds = drop_dimensionless_variables(zbp_ds)
    prefix = "zbp_dataset_derivative"
    zbp_ds.attrs.update(
        {
            f"{prefix}.derivative_threshold": derivative_threshold,
            f"{prefix}.zbp_probability_threshold": zbp_probability_threshold,
            f"{prefix}.bias_window": bias_window,
            f"{prefix}.polyorder": polyorder,
            f"{prefix}.average_over_cutter": average_over_cutter,
            **get_metadata(),
        }
    )
    if guid := ds_left.attrs.get("guid"):
        zbp_ds.attrs["guid.left"] = guid
    if guid := ds_right.attrs.get("guid"):
        zbp_ds.attrs["guid.right"] = guid

    # Set gap if possible
    if "gap" in ds_left.variables.keys():
        set_zbp_gap(zbp_ds, ds_left, ds_right)
    else:
        warnings.warn("No gap variable found in the data, call `set_zbp_gap` first.")

    if "L_SI" in ds_left.variables.keys():
        L_SI = (ds_left.L_SI < 0).astype(bool)
        R_SI = (ds_right.R_SI < 0).astype(bool)
        L_SI = L_SI.sum(dim="cutter_pair_index").astype(bool)
        R_SI = R_SI.sum(dim="cutter_pair_index").astype(bool)
        if "left_bias" in L_SI.dims:
            L_SI = L_SI.sel(left_bias=0.0, method="nearest")
        if "right_bias" in R_SI.dims:
            R_SI = R_SI.sel(right_bias=0.0, method="nearest")
        zbp_ds["SI"] = (L_SI.astype(int) + R_SI.astype(int)).astype(bool)

    return zbp_ds


def drop_dimensionless_variables(ds, skip: set | None = None):
    """Drop dimensionless variables from a dataset."""
    skip = skip or set()
    to_drop = [
        coord.name
        for coord in ds.coords.values()
        if coord.dims == () and coord.name not in skip
    ]
    return ds.drop_vars(to_drop)


def extract_gap_from_trace(
    bool_array: np.ndarray,
    bias_array: np.ndarray,
    max_gap_mode: Literal["nan", "max_bias"] = "max_bias",
) -> float:
    """Extract the gap values from a 1D binary array in units of the given bias array.

    Returns NaN if no gap is found.

    Parameters
    ----------
    bool_array
        Boolean 1D array.
    bias_array
        1D array of bias values.
    max_gap_mode
        If "nan", return NaN if no gap is found, otherwise return the maximum bias value.

    Returns
    -------
    float
        gap
    """
    i_zero = np.argmin(np.abs(bias_array))  # Index of the bias zero.
    bool_array = bool_array[i_zero:]
    bias_array = bias_array[i_zero:]
    gap = bias_array[bool_array]
    if gap.size == 0:
        return bias_array[-1] if max_gap_mode == "max_bias" else np.nan

    found_gap = gap[0]
    i = np.argmin(np.abs(bias_array - found_gap))
    if i == len(bias_array) - 1:
        return found_gap if max_gap_mode == "max_bias" else np.nan
    correction = (bias_array[i + 1] - bias_array[i]) / 2
    return found_gap + correction


def determine_gap(
    conductance: xr.Dataset,
    bias_name: str,
    field_name: str | None = "B",
    median_size: float = 2,
    gauss_sigma: tuple[float, float] = (0.0, 0.0),
    gap_threshold_factor: float = 0.05,
    upper_conductance_threshold: float = float("inf"),
    noise_threshold: float = 0.0,
    filtered_antisym_g: bool = False,
    max_gap_mode: Literal["nan", "max_bias"] = "max_bias",
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """
    Extract the gap from a 2D array (non-local conductance vs bias and field)
    using thresholding of filtered data.

    Parameters
    ----------
    conductance
        (non-local conductance vs bias, an optionally field)
    bias_name
        Name of bias axis.
    field_name
        Name of field axis, is allowed to be None.
    median_size
        Size of the median filter, ``size`` argument to `scipy.ndimage.median_filter`.
    gauss_sigma
        Size of the gaussian filter, ``sigma`` argument to `scipy.ndimage.gaussian_filter`.
    gap_threshold_factor
        Threshold in ``gap_threshold_factor * min(max(x), upper_conductance_threshold)``.
    upper_conductance_threshold
        See `gap_threshold_factor`.
    noise_threshold
        Whenever the max conductance at each bias value is smaller
        than this, the gap is set to the max bias value.
    filtered_antisym_g
        Include the filtered_antisym_g from which the gap
        is extracted. If True, a tuple of ``gap, filtered`` is returned.
    max_gap_mode
        If "nan", return NaN if no gap is found, otherwise return the maximum bias value.

    Returns
    -------
    xarray.DataArray or Tuple[xarray.DataArray, xarray.DataArray]
        Estimates of the gap.
    """

    if field_name is None:
        vectorize = False
        dims = [bias_name]
    else:
        vectorize = True
        dims = [bias_name, field_name]

    conductance = conductance.load()
    defaults = dict(
        input_core_dims=[dims],
        output_core_dims=[dims],
        vectorize=vectorize,
    )
    conductance = xr.apply_ufunc(
        scipy.ndimage.median_filter,
        conductance,
        kwargs=dict(size=median_size),
        **defaults,
    )
    antisymmetric_conductance = antisymmetric_conductance_part(
        conductance, bias_name, field_name
    )
    filtered = xr.apply_ufunc(
        scipy.ndimage.gaussian_filter,
        antisymmetric_conductance,
        kwargs=dict(sigma=gauss_sigma),
        **defaults,
    )

    def gap_thresholding(G):
        f = gap_threshold_factor * min(np.max(G), upper_conductance_threshold)
        return np.abs(G) > max(f, noise_threshold)

    binary = xr.apply_ufunc(gap_thresholding, filtered, **defaults)
    gap = xr.apply_ufunc(
        extract_gap_from_trace,
        binary,
        binary[bias_name],
        input_core_dims=[[bias_name], [bias_name]],
        output_core_dims=[[]],
        kwargs=dict(max_gap_mode=max_gap_mode),
        vectorize=True,
    )
    gap.attrs.update(
        median_size=median_size,
        gauss_sigma=gauss_sigma,
        gap_threshold_factor=gap_threshold_factor,
        filtered_antisym_g=filtered_antisym_g,
        max_gap_mode=max_gap_mode,
    )
    gap = gap.rename("gap")
    return (gap, filtered) if filtered_antisym_g else gap


def extract_gap(
    ds_left: xr.Dataset,
    ds_right: xr.Dataset,
    median_size: float | tuple[float, float] = 2,
    gauss_sigma: float | tuple[float, float] = (0.0, 0.0),
    gap_threshold_factor: float = 0.05,
    upper_conductance_threshold: float = float("inf"),
    noise_threshold: float = 0.0,
    max_gap_mode: Literal["nan", "max_bias"] = "max_bias",
) -> tuple[xr.Dataset, xr.Dataset]:
    """Add the gap to all regions and sides for a stage 2 datasets.

    Parameters
    ----------
    median_size
        Size of the median filter, ``size`` argument to `scipy.ndimage.median_filter`.
    gauss_sigma
        Size of the gaussian filter, ``sigma`` argument to `scipy.ndimage.gaussian_filter`.
    gap_threshold_factor
        Threshold in ``gap_threshold_factor * min(max(x), upper_conductance_threshold)``.
    upper_conductance_threshold
        See `gap_threshold_factor`.
    noise_threshold
        Whenever the max conductance at each bias value is smaller
        than this, the gap is set to the max bias value.
    max_gap_mode
        If "nan", return NaN if no gap is found, otherwise return the maximum bias value.
    """
    # merge gap data with xarray of left and right side
    for side, ds in (("left", ds_left), ("right", ds_right)):
        other_side = {"left": "r", "right": "l"}[side]
        key = f"g_{other_side}{side[0]}"
        (ds["gap"], ds["filtered_antisym_g"]) = determine_gap(
            ds[key],
            bias_name=f"{side}_bias",
            field_name="B" if "B" in ds[key].dims else None,
            median_size=median_size,
            gauss_sigma=gauss_sigma,
            gap_threshold_factor=gap_threshold_factor,
            filtered_antisym_g=True,
            max_gap_mode=max_gap_mode,
            upper_conductance_threshold=upper_conductance_threshold,
            noise_threshold=noise_threshold,
        )
        prefix = "extract_gap"
        ds.attrs.update(
            {
                f"{prefix}.median_size": median_size,
                f"{prefix}.gauss_sigma": gauss_sigma,
                f"{prefix}.gap_threshold_factor": gap_threshold_factor,
                f"{prefix}.upper_conductance_threshold": upper_conductance_threshold,
                f"{prefix}.noise_threshold": noise_threshold,
                f"{prefix}.max_gap_mode": max_gap_mode,
            }
        )
    return ds_left, ds_right


@ds_requires(variables=("gap",), sets_variables=("gap_boolean", "gapped_zbp"))
def set_gap_threshold(
    zbp_ds: xr.Dataset,
    threshold_low: float = 10e-3,
    threshold_high: float | None = None,
) -> None:
    """Set the gap threshold for a stage 2 dataset.

    Adds ``zbp_ds["gap_boolean"]`` and ``zbp_ds["gapped_zbp"]``.

    Parameters
    ----------
    zbp_ds
        ZBP dataset, as returned by `tgp.two.zbp_dataset_derivative`.
    threshold_low
        Lower gap threshold for the ``gap_boolean`` array.
    threshold_high
        Upper gap threshold, only used for the ``gapped_zbp`` array.
        If None, no upper threshold is used.
    """
    zbp_ds["gap_boolean"] = zbp_ds.gap > threshold_low
    zbp_ds["gapped_zbp"] = zbp_ds.gap_boolean * zbp_ds.zbp > 0
    if threshold_high is not None:
        zbp_ds["gapped_zbp"] *= zbp_ds.gap < threshold_high
    prefix = "set_gap_threshold"
    zbp_ds.attrs.update(
        {
            f"{prefix}.threshold_low": threshold_low,
            f"{prefix}.threshold_high": threshold_high,
        }
    )


@ds_requires(sets_variables=("gap",))
def set_zbp_gap(
    zbp_ds: xr.Dataset,
    ds_left: xr.Dataset,
    ds_right: xr.Dataset,
):
    """Add ``zbp_ds["gap"]`` based on the minimal gap of the left and right dataset."""
    # Take the minimum of both gaps, but when one gap is NaN and the other is not, take the value.
    left = ds_left.gap.squeeze().copy()
    right = ds_right.gap.squeeze().copy()
    left = left.where(~np.isnan(left), np.inf)
    right = right.where(~np.isnan(right), np.inf)
    gap = np.minimum(left, right)
    gap = gap.where(~np.isinf(gap), np.nan)
    zbp_ds["gap"] = gap


def _gapless_boundary(
    boundary_indices: set[tuple[int, int]],
    gap_closed_array: np.ndarray,
    variance: int,
) -> tuple[float, dict[tuple[int, int], bool]]:
    """Return the percentage of the boundary of the 2D cluster that is gapless.

    Parameters
    ----------
    boundary_indices
        Set of indices of the points at the boundary of a cluster.
    gap_closed_array
        Boolean array variables with True signifying closed gap.
    variance
        Position tolerance for distance between the ZBP array end and the gap
        closing.

    Returns
    -------
    Tuple[float, Dict[Tuple[int, int], bool]]]
        - Percentage of the boundary that is gapless.
        - Dictionary with boundary indices -> bool
    """

    ###### For pixels inside the cluster ######

    s = max(1, variance // 2)
    n, m = gap_closed_array.shape
    topo_boundary = {}
    for i, j in boundary_indices:
        gap_is_close = gap_closed_array[
            max(0, i - s) : min(i + s + 1, n),
            max(0, j - s) : min(j + s + 1, m),
        ].any()
        topo_boundary[i, j] = gap_is_close
    percentage = (
        sum(topo_boundary.values()) / len(topo_boundary) if topo_boundary else 0.0
    )

    return percentage, topo_boundary


class _GapStat(NamedTuple):
    minimal_gap: float = np.nan
    maximal_gap: float = np.nan
    average_gap: float = np.nan  # (with sign)
    median_gap: float = np.nan  # (with sign)
    top_quintile_gap: float = np.nan  # (with sign)
    average_absolute_gap: float = np.nan  # (without sign)
    median_absolute_gap: float = np.nan  # (without sign)


def _gap_statistics(cluster_array: np.ndarray, gap_array: np.ndarray) -> _GapStat:
    """Given a cluster array and array of gap values (computed or deduced),
    this function returns average, maximal, and minimal (sign-resolved) gap.

    Parameters
    ----------
    cluster_array
        A boolean array of a cluster.
    gap_array
        Array of gap values for all data points.

    Returns
    -------
    _GapStat namedtuple
    """
    gaps = gap_array[np.nonzero(cluster_array)]
    if gaps.size == 0:
        return _GapStat()
    abs_gap = np.abs(gaps)
    return _GapStat(
        minimal_gap=gaps.min(),
        maximal_gap=gaps.max(),
        average_gap=gaps.mean(),
        median_gap=np.median(gaps),
        top_quintile_gap=np.percentile(gaps, 80.0),
        average_absolute_gap=abs_gap.mean(),
        median_absolute_gap=np.median(abs_gap),
    )


SCORE_VARIABLES = (
    "percentage_boundary",
    "top_quintile_gap",
    "median_gap",
    "average_gap",
    "nonlocal_score_top_quintile",
    "nonlocal_score_median",
    "nonlocal_score_average",
    "nonlocal_score_max",
    "cluster_volume",
    "cluster_npixels",
    *CLUSTER_INFO_VARIABLES,
)
SCORE_VARIABLES_OPTIONAL = ("cluster_sets", "nclusters", "ncutters", "roi2")


@ds_requires(
    variables=("gapped_zbp_cluster", "gap_boolean", "gap"),
    sets_variables=SCORE_VARIABLES,
)
def set_score(
    zbp_ds: xr.Dataset,
    variance: int = 3,
    with_overlaps: bool = False,
    *,
    add_attrs: bool = True,
) -> None:
    """Set the score for the given Stage 2 ZBP dataset.

    Adds "percentage_boundary", "top_quintile_gap", "median_gap", "average_gap",
    "nonlocal_score_top_quintile", "nonlocal_score_median", "nonlocal_score_average",
    "nonlocal_score_max", "cluster_volume", "cluster_npixels", "cluster_B_center",
    "cluster_V_center", "cluster_B_min", "cluster_V_min", "cluster_B_max",
    "cluster_V_max", arrays.

    Parameters
    ----------
    zbp_ds
        Full Stage 2 dataset that has been clustered.
    variance
        Position tolerance for distance between the ZBP array end and the gap
        closing.
    with_overlaps
        If True, "cluster_sets", "nclusters", "ncutters", "roi2" DataArrays
        are added, but only if the "cutter_pair_index" dimension is included.
    add_attrs
        Whether to add the input parameters as attributes to the dataset.
    """
    dim = "zbp_cluster_number"
    clusters = expand_clusters(zbp_ds["gapped_zbp_cluster"], dim=dim)
    if clusters.coords[dim].shape == (0,):
        empty = clusters.copy().sum(["B", "V"])
        keys = list(SCORE_VARIABLES)
        if with_overlaps and "cutter_pair_index" in zbp_ds.gapped_zbp_cluster.dims:
            keys.extend(list(SCORE_VARIABLES_OPTIONAL))
        for k in keys:
            empty.name = k
            zbp_ds[k] = empty.copy()
        return

    boundary_indices_set = xr.apply_ufunc(
        _get_boundary_indices,
        clusters,
        input_core_dims=[["V", "B"]],
        output_core_dims=[[]],
        vectorize=True,
    )

    percentage_boundary, _ = xr.apply_ufunc(
        _gapless_boundary,
        boundary_indices_set,
        ~zbp_ds["gap_boolean"].astype(bool),
        input_core_dims=[[], ["V", "B"]],
        output_core_dims=[[], []],
        vectorize=True,
        kwargs=dict(variance=variance),
    )

    cluster_volume = clusters.integrate("B").integrate("V")
    cluster_npixels = clusters.sum(["B", "V"])

    info = groupby_map(
        tgp.common.cluster_info, clusters, ("B", "V"), as_xarray=True, pct_box=0
    )
    for k, v in info.data_vars.items():
        zbp_ds[k] = v

    _, max_gap, average_gap, median_gap, top_quintile_gap, *_ = xr.apply_ufunc(
        _gap_statistics,
        clusters,
        zbp_ds["gap"],
        input_core_dims=[
            ["V", "B"],
            ["V", "B"],
        ],
        output_core_dims=len(_GapStat._fields) * [[]],
        vectorize=True,
    )
    zbp_ds["percentage_boundary"] = percentage_boundary
    zbp_ds["top_quintile_gap"] = top_quintile_gap
    zbp_ds["median_gap"] = median_gap
    zbp_ds["average_gap"] = average_gap
    zbp_ds["nonlocal_score_top_quintile"] = top_quintile_gap * percentage_boundary
    zbp_ds["nonlocal_score_median"] = median_gap * percentage_boundary
    zbp_ds["nonlocal_score_average"] = average_gap * percentage_boundary
    zbp_ds["nonlocal_score_max"] = max_gap
    zbp_ds["cluster_volume"] = cluster_volume
    zbp_ds["cluster_npixels"] = cluster_npixels

    if with_overlaps and "cutter_pair_index" in zbp_ds.gapped_zbp_cluster.dims:
        zbp_ds["cluster_sets"] = get_overlapping_clusters(zbp_ds.gapped_zbp_cluster)
        zbp_ds["nclusters"] = get_nclusters(zbp_ds.cluster_sets)
        zbp_ds["ncutters"] = get_ncutters(zbp_ds.gapped_zbp_cluster)
        roi2 = construct_roi2(zbp_ds.gapped_zbp_cluster, zbp_ds.cluster_sets)
        zbp_ds["roi2"] = flatten_clusters(roi2.astype(bool), dim="roi2")

    if add_attrs:
        prefix = "set_score"
        zbp_ds.attrs.update(
            {
                f"{prefix}.variance": variance,
                f"{prefix}.with_overlaps": with_overlaps,
            }
        )


def cluster_sets_to_cluster_pairs(
    cluster_sets: xr.DataArray, mode: Literal["index", "dimension"] = "index"
) -> list[set[ClusterPair]]:
    """Convert `cluster_sets` DataArray to a list of sets with tuples of `ClusterPair`s."""
    cluster_sets = cluster_sets.transpose("cutter_pair_index", "zbp_cluster_number")
    overlaps = []
    for i in np.sort(np.unique(cluster_sets)):
        if np.isnan(i):
            continue
        i, j = np.where(cluster_sets == i)
        sets = {(a, b) for a, b in zip(i, j)}
        overlaps.append(sets)
    if mode == "dimension":
        dim_a = cluster_sets["cutter_pair_index"]
        dim_b = cluster_sets["zbp_cluster_number"]
        overlaps = [
            {(dim_a.values[a], dim_b.values[b]) for a, b in sets} for sets in overlaps
        ]
    return overlaps


def set_boundary_array(
    ds,
    field_name: str = "B",
    plunger_gate_name: str = "V",
    pixel_size: int = 3,
    name_cluster: str = "gapped_zbp_cluster",
    name_boundary_indices: str = "boundary_indices",
    name_boundary_array: str = "boundary_array",
    all_boundaries: bool = False,
) -> None:
    """Set the boundary array (array of booleans with edge pixels) for the given dataset.

    Parameters
    ----------
    ds
        Full Stage 2 dataset.
    field_name
        Magnetic field name.
    plunger_gate_name
        Plunger gate name.
    pixel_size
        Size of boundary pixels.
    name_cluster
        Name of the existing cluster array.
    name_boundary_indices
        Name of the existing boundary indices array.
    name_boundary_array
        Name of the new boundary array.
    all_boundaries
        Whether to add a boundary array for each cluster or only topological clusters.
    """

    def _add_boundary_array(cluster_array, boundary, pixel_size):

        boundary_arr = np.zeros_like(cluster_array) * np.nan
        for (i, j), is_topo in boundary.items():
            if is_topo or all_boundaries:
                px = pixel_size - 1
                boundary_arr[i - px : i + px + 1, j - px : j + px + 1] = 1
        return boundary_arr

    ds[name_boundary_array] = xr.apply_ufunc(
        _add_boundary_array,
        ds[name_cluster],
        ds[name_boundary_indices],
        input_core_dims=[[plunger_gate_name, field_name], []],
        output_core_dims=[[plunger_gate_name, field_name]],
        vectorize=True,
        kwargs=dict(pixel_size=pixel_size),
    )


@ds_requires(variables=("gapped_zbp",), sets_variables=("gapped_zbp_cluster",))
def set_clusters(
    zbp_ds: xr.Dataset,
    min_cluster_size: int = 7,
    xi: float = 0.05,
    max_eps: float = 2,
    min_samples: int = 3,
    *,
    add_attrs: bool = True,
) -> None:
    """Set the clusters for the given Stage 2 ZBP dataset.

    Can have dimensions ``(B, V)`` or ``(B, V, cutter_pair_index)``.

    If the data contains ``"cutter_pair_index"``, the clustering will be performed
    per ``cutter_pair_value``.

    Parameters
    ----------
    zbp_ds
        ZBP dataset, as returned by `tgp.two.zbp_dataset_derivative`.
    min_samples
        Argument passed to `sklearn.cluster.OPTICS`.
    xi
        Argument passed to `sklearn.cluster.OPTICS`.
    min_cluster_size
        Argument passed to `sklearn.cluster.OPTICS`.
    max_eps
        Argument passed to `sklearn.cluster.OPTICS`.
    add_attrs
        Whether to add the input parameters as attributes to the dataset.
    """
    tgp.common.set_clusters_of(
        zbp_ds,
        "gapped_zbp",
        "gapped_zbp_cluster",
        min_cluster_size=min_cluster_size,
        xi=xi,
        max_eps=max_eps,
        min_samples=min_samples,
        force=True,
    )
    if add_attrs:
        prefix = "set_clusters"
        zbp_ds.attrs.update(
            {
                f"{prefix}.min_cluster_size": min_cluster_size,
                f"{prefix}.xi": xi,
                f"{prefix}.max_eps": max_eps,
                f"{prefix}.min_samples": min_samples,
            }
        )


@ds_requires(
    variables=set_clusters.__required_variables__,
    sets_variables=set_clusters.__sets_variables__ + set_score.__sets_variables__,
)
def cluster_and_score(
    zbp_ds: xr.Dataset,
    min_cluster_size: int = 7,
    xi: float = 0.05,
    max_eps: float = 2,
    min_samples: int = 3,
    variance: int = 3,
    cluster_gap_threshold: float | None = None,
    cluster_volume_threshold: float | None = None,
    cluster_percentage_boundary_threshold: float | None = None,
) -> xr.Dataset:
    """Run clustering and score on the given Stage 2 ZBP dataset.

    Function that calls `set_clusters` and `set_score` and allows to ignore clusters
    based on thresholds.

    Parameters
    ----------
    zbp_ds
        ZBP dataset, as returned by `tgp.two.zbp_dataset_derivative`.
    min_samples
        Argument passed to `sklearn.cluster.OPTICS`.
    xi
        Argument passed to `sklearn.cluster.OPTICS`.
    min_cluster_size
        Argument passed to `sklearn.cluster.OPTICS`.
    max_eps
        Argument passed to `sklearn.cluster.OPTICS`.
    variance
        Position tolerance for distance between the ZBP array end and the gap
        closing.
    cluster_gap_threshold
        Minimum median gap of the clusters.
    cluster_volume_threshold
        Minimum volume of the clusters.
    cluster_percentage_boundary_threshold
        Minimum percentage of boundary of the clusters.

    Returns
    -------
    xr.Dataset
        Dataset with new arrays added.
    """
    no_thresholds = (
        cluster_gap_threshold is None
        and cluster_percentage_boundary_threshold is None
        and cluster_volume_threshold is None
    )
    ds = zbp_ds if no_thresholds else zbp_ds.copy()
    set_clusters(
        ds,
        min_cluster_size=min_cluster_size,
        xi=xi,
        max_eps=max_eps,
        min_samples=min_samples,
        add_attrs=False,
    )
    set_score(ds, variance=variance, with_overlaps=no_thresholds, add_attrs=False)

    prefix = "cluster_and_score"
    zbp_ds.attrs.update(
        {
            f"{prefix}.min_samples": min_samples,
            f"{prefix}.xi": xi,
            f"{prefix}.min_cluster_size": min_cluster_size,
            f"{prefix}.max_eps": max_eps,
            f"{prefix}.variance": variance,
            f"{prefix}.cluster_gap_threshold": cluster_gap_threshold,
            f"{prefix}.cluster_volume_threshold": cluster_volume_threshold,
            f"{prefix}.cluster_percentage_boundary_threshold": cluster_percentage_boundary_threshold,
        }
    )

    if no_thresholds:
        return ds

    # Remove clusters that do not meet condition
    condition = 1
    if cluster_gap_threshold is not None:
        gap_condition = ds.median_gap > cluster_gap_threshold * 1e-3
        condition &= gap_condition
    if cluster_percentage_boundary_threshold is not None:
        boundary_condition = (
            ds.percentage_boundary > cluster_percentage_boundary_threshold
        )
        condition &= boundary_condition
    if cluster_volume_threshold is not None:
        volume_condition = ds.cluster_volume > cluster_volume_threshold
        condition &= volume_condition
    # Set clusters that do not satisfy conditions to False and
    # assign flattened 'gapped_zbp_cluster' to unclustered dataset
    zbp_ds["gapped_zbp_cluster"] = remove_with_condition(
        ds.gapped_zbp_cluster, condition
    )
    zbp_ds["cluster_condition"] = condition
    # Redo score for clusters
    set_score(zbp_ds, with_overlaps=True)

    return zbp_ds


def _get_boundary_indices(cluster_array: np.ndarray) -> set[tuple[int, int]]:
    """Return indices of the points at the boundary of a cluster.

    Parameters
    ----------
    cluster_array
        A boolean array of a cluster.

    Returns
    -------
    Set[Tuple[int, int]]
    """
    # Add a single pixel padding to the cluster array to avoid issues in the corners
    cluster_array = np.pad(cluster_array, 1)

    nonzero = cluster_array != 0
    boundary = set()  # set with (i, j) tuples of the boundary

    s = 1  # set as a parameter?
    n, m = nonzero.shape
    for i in range(n):
        for j in range(m):
            is_boundary = (
                nonzero[i, j]
                and (
                    ~nonzero[
                        max(0, i - s) : min(i + s + 1, n),
                        max(0, j - s) : min(j + s + 1, m),
                    ]
                ).any()
            )
            if is_boundary:
                boundary.add((i, j))

    # Remove padding
    boundary = {(i - 1, j - 1) for i, j in boundary}

    return boundary


def antisymmetric_conductance_part(
    conductance: xr.DataArray, bias_name: str, field_name: str | None = "B"
) -> xr.DataArray:
    """Return the anti-symmetric part of the conductance."""
    dims = [bias_name, field_name] if field_name else [bias_name]
    return xr.apply_ufunc(
        lambda x: (x - x[::-1]) / 2,
        conductance,
        input_core_dims=[dims],
        output_core_dims=[dims],
        vectorize=True,
    )


ClusterPair: TypeAlias = Tuple[int, int]


def _all_overlapping_combinations(clusters: xr.DataArray) -> list[set[ClusterPair]]:
    dim_cut = "cutter_pair_index"
    dim_clu = "zbp_cluster_number"
    pairs = itertools.product(
        range(len(clusters[dim_cut])), range(len(clusters[dim_clu]))
    )
    pairs = np.array(list(pairs))

    # Take array in known shape
    x = clusters.transpose(dim_cut, dim_clu, "B", "V").values

    # flatten dim_cut, dim_clu
    x = x.reshape(-1, len(clusters["B"]), len(clusters["V"]))
    prod = x[None, ...] * x[:, None, ...]  # calculate overlap for all combinations
    mat = prod.sum(axis=(-1, -2))  # sum over B and V
    mat = np.triu(mat, k=-1)  # set lower triang elements to 0 to avoid double counting
    # get indices of pairs that overlap
    overlaps = [pairs[row > 0].tolist() for row in mat]
    # to list[set[tuple[int, int]]]
    return [{tuple(x) for x in ov} for ov in overlaps if ov]


def _join_overlapping_sets(sets):
    if len(sets) <= 1:
        return sets
    i, j = zip(*itertools.combinations(range(len(sets)), 2))
    data = [not sets[a].isdisjoint(sets[b]) for a, b in zip(i, j)]
    # construct a graph of which sets have intersections
    graph = scipy.sparse.coo_matrix((data, (i, j)), shape=(len(sets), len(sets)))
    # each set intersects with itself (csgraph.connected_components needs these to function)
    graph += scipy.sparse.identity(len(sets))
    # find which sets "transitively" intersect
    ncomponents, labels = scipy.sparse.csgraph.connected_components(
        graph, directed=False
    )
    # Merge together sets that transitively intersect
    ret = [set() for _ in range(ncomponents)]
    for s, j in zip(sets, labels):
        ret[j] |= s
    return ret


def _mapping_to_dataarray(
    mapping: dict[Any, Any],
    dims: list[str, list[Any]],
    fill: float = np.nan,
    dtype: str = "float",
) -> xr.DataArray:
    arr = np.ones(tuple(len(v) for _, v in dims), dtype=dtype) * fill
    for i, label in mapping.items():
        arr[i] = label
    return xr.DataArray(
        arr,
        coords={k: v for k, v in dims},
        dims=[k for k, _ in dims],
    )


def _cluster_sets_to_da(
    clusters: xr.DataArray, cluster_sets: list[set[ClusterPair]]
) -> xr.DataArray:
    mapping = {
        pair: i for i, pairs in enumerate(cluster_sets, start=1) for pair in pairs
    }
    dim_cut = "cutter_pair_index"
    dim_clu = "zbp_cluster_number"
    dims = [(dim_cut, clusters[dim_cut]), (dim_clu, clusters[dim_clu])]
    return _mapping_to_dataarray(mapping, dims)


def get_overlapping_clusters(
    gapped_zbp_cluster: xr.DataArray,
) -> list[set[ClusterPair]]:
    """Find overlapping clusters.

    Parameters
    ----------
    zbp_ds
        Dataset containing ``gapped_zbp_cluster``.

    Returns
    -------
    cluster_sets
        List of sets of cluster pairs that overlap.
    """
    clusters = expand_clusters(gapped_zbp_cluster, dim="zbp_cluster_number")
    overlaps = _all_overlapping_combinations(clusters)
    cluster_sets = _join_overlapping_sets(overlaps)
    return _cluster_sets_to_da(clusters, cluster_sets)


def get_nclusters(cluster_sets: xr.DataArray):
    """Find the number of clusters for each ROI2."""
    nclusters = xr.zeros_like(cluster_sets, dtype=int)
    for i in np.unique(cluster_sets):
        if np.isnan(i):
            continue
        n = (cluster_sets == i).astype(int).values.sum()
        nclusters.values[cluster_sets.values == i] = n
    return nclusters


def get_ncutters(gapped_zbp_cluster: xr.DataArray) -> xr.DataArray:
    """Get the number of overlapping clusters at different cutter values."""
    dim_cut = "cutter_pair_index"
    dim_clu = "zbp_cluster_number"
    clusters = expand_clusters(gapped_zbp_cluster, dim=dim_clu)
    union = clusters.astype(int).sum([dim_cut, dim_clu])
    mapping = {
        (i, j): (cluster.astype(int) * union).max().item()
        for i, _clusters in enumerate(clusters.transpose(dim_cut, dim_clu, ...))
        for j, cluster in enumerate(_clusters)
    }
    dims = [(dim_cut, clusters[dim_cut]), (dim_clu, clusters[dim_clu])]
    return _mapping_to_dataarray(mapping, dims, fill=0, dtype=int)


def construct_roi2(
    gapped_zbp_cluster: xr.DataArray, cluster_sets: xr.DataArray
) -> xr.Dataset:
    """Construct the ROI2 dataset where the numbers indicate the number of cutter values.

    The 'roi2' dimension that is added corresponds with the labels in ``zbp_ds.cluster_sets``.
    """
    clusters = expand_clusters(gapped_zbp_cluster, dim="zbp_cluster_number")
    if gapped_zbp_cluster.max() == 0:  # no clusters so no roi2
        return clusters.sum(dim=["B", "V"]).rename(zbp_cluster_number="roi2")
    roi2s = []
    labels = [int(i) for i in np.unique(cluster_sets) if not np.isnan(i)]
    for i in labels:
        i_x, i_y = np.where(cluster_sets.transpose("cutter_pair_index", ...) == i)
        roi2 = sum(
            drop_dimensionless_variables(
                clusters.isel(cutter_pair_index=a, zbp_cluster_number=b)
            )
            for a, b in zip(i_x, i_y)
        )
        roi2s.append(roi2)
    roi2s = xr.concat(roi2s, "roi2")
    roi2s = roi2s.assign_coords({"roi2": labels})
    return roi2s


def remove_with_condition(
    gapped_zbp_cluster: xr.DataArray, condition: xr.DataArray
) -> xr.DataArray:
    """Remove clusters that do not meet condition."""
    cl = expand_clusters(gapped_zbp_cluster, dim="zbp_cluster_number")
    gapped_zbp_cluster = cl.where(condition, other=False)
    # Reflatten the masked out gapped_zbp_cluster array
    gapped_zbp_cluster = xr_map(
        gapped_zbp_cluster,
        f=flatten_clusters,
        dim="cutter_pair_index",
        kwargs={"dim": "zbp_cluster_number"},
    )
    return gapped_zbp_cluster
