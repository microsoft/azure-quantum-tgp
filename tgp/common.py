# Copyright (c) Microsoft Corporation. All rights reserved.

from __future__ import annotations

import datetime
import functools
import getpass
from numbers import Number
import platform
from typing import Any, Callable, Iterable, Literal, NamedTuple, TypeVar

import joblib
import numpy
import numpy as np
import scipy
from scipy import ndimage
import sklearn
import sklearn.cluster
import xarray as xr

import tgp

memory = joblib.Memory("cachedir", verbose=0)

PassingOptions = Literal["passed", "failed", "inconclusive"]


T = TypeVar("T")


def ds_requires(
    variables: tuple[str | set[str], ...] = (),
    dims: tuple[str | set[str], ...] = (),
    sets_variables: tuple[str | set[str], ...] = (),
) -> Callable[[Callable[..., xr.Dataset]], Callable[..., xr.Dataset]]:
    """Decorate function to assert that a dataset has the required variables and dimensions.

    When variables or dimensions are a set of strings, the dataset must have any of them.

    Parameters
    ----------
    variables
        Tuple of strings or sets of strings.
        The dataset must have all the specified variables.
    dims
        Tuple of strings or sets of strings.
        The dataset must have all the specified dimensions.
    sets_variables
        Tuple of strings or sets of strings.
        The dataset must have all the specified variables,
        and they must be data variables.
    """

    def is_in(x: Iterable[str], k: str | set[str]) -> bool:
        return any(_k in x for _k in k) if isinstance(k, set) else k in x

    def decorator(f: Callable[..., xr.Dataset]) -> Callable[..., xr.Dataset]:
        @functools.wraps(f)
        def _wrapped(*args, **kwargs) -> xr.Dataset:
            ds = args[0]
            for v in variables + tuple(kwargs.get("keys", ())):
                if not is_in(ds.variables, v):
                    raise KeyError(f"Data variable '{v}' is missing from `ds`.")
            for d in dims:
                if not is_in(ds.dims, d):
                    raise KeyError(f"Dimension '{d}' is missing from `ds`.")
            result = f(*args, **kwargs)
            if result is None:
                # if the function returns None, don't check the output
                # but the arrays added the the input dataset
                check = ds
            else:
                check = result
            for v in sets_variables:
                if not is_in(check.variables, v):
                    raise KeyError(f"Data variable '{v}' is missing from `ds`.")
                if v not in check.data_vars:
                    raise KeyError(f"Data variable '{v}' is not a data variable.")
            return result

        _wrapped.__required_variables__ = variables
        _wrapped.__required_dims__ = dims
        _wrapped.__sets_variables__ = sets_variables
        return _wrapped

    return decorator


class ThresholdException(Exception):
    """Threshold too low or high."""


def _data_clusters(
    da: xr.DataArray,
    *,
    min_samples: int = 3,
    xi: float = 0.1,
    min_cluster_size: float | int = 0.01,
    max_eps: float = 10.0,
    flatten: bool = False,
) -> np.ndarray:
    X = np.transpose(np.nonzero(da.data))
    if X.shape[0] > min_cluster_size:
        model = sklearn.cluster.OPTICS(
            min_samples=min_samples,
            xi=xi,
            min_cluster_size=min_cluster_size,
            max_eps=max_eps,
        )
        labels = model.fit_predict(X)
    else:
        labels = [-1] * X.shape[0]

    data_clusters = []
    for label in np.unique(labels):
        if label < 0:
            continue
        data_cluster = np.zeros_like(da)
        for ii, jj in X[labels == label]:
            data_cluster[ii, jj] = 1
        data_clusters.append(data_cluster)
    data_clusters = sorted(data_clusters, key=np.sum, reverse=True)
    data_clusters = np.array(data_clusters, dtype=bool)
    if data_clusters.size == 0:
        if not flatten:
            # if no clusters, return an empty array of the correct shape
            return np.empty((0, *da.shape), dtype=bool)
        else:
            return np.zeros(da.shape, dtype=int)
    return data_clusters if not flatten else _np_flatten(data_clusters)


def _np_flatten(arr: np.ndarray) -> np.ndarray:
    overlap = np.zeros(arr.shape[1:], dtype=int)
    if arr.shape[0] == 0:
        return overlap
    for x in arr:
        overlap += (1 + overlap.max()) * x.astype(int)
    return overlap


def flatten_clusters(da: xr.DataArray, *, dim: str = "cluster") -> xr.DataArray:
    """Flatten a dimension of a booleans array to integers."""
    overlap = xr.zeros_like(da, dtype=int).sum(dim=dim)
    if len(da.coords[dim]) == 0:
        return overlap
    for i, _ in enumerate(da[dim]):
        data_cluster = da.isel({dim: i}).astype(int)
        overlap += (1 + overlap.max().item()) * data_cluster
    return overlap.drop_vars(dim)


def expand_clusters(
    da: xr.DataArray,
    *,
    dim: str = "cluster",
    zeros_if_empty: bool = False,
) -> xr.DataArray:
    """Expand clusters array of shape (N, M) with numbers from 0 to K to shape (N, M, K-1)."""
    max_cluster = da.values.max() if da.size > 0 else 0
    clusters = np.arange(1, max_cluster + 1)
    if clusters.size == 0:
        if zeros_if_empty:
            return xr.zeros_like(da, dtype=bool)
        coords = {dim: np.array(()), **da.coords}
        return xr.DataArray(
            np.empty((0, *da.shape)),
            coords=coords,
            dims=(dim, *da.dims),
        )
    da_clusters = xr.concat(
        [da.where(da == i, other=0).astype(bool) for i in clusters],
        dim=dim,
    )
    return da_clusters.assign_coords({dim: clusters})


def _xr_data_clusters(
    da: xr.DataArray,
    *,
    min_samples: int = 3,
    xi: float = 0.1,
    min_cluster_size: float | int = 0.01,
    max_eps: float = 10.0,
) -> xr.DataArray:
    clusters = xr.apply_ufunc(
        _data_clusters,
        da,
        kwargs={
            "min_samples": min_samples,
            "xi": xi,
            "min_cluster_size": min_cluster_size,
            "max_eps": max_eps,
            "flatten": True,
        },
        input_core_dims=[("B", "V")],
        output_core_dims=[("B", "V")],
        vectorize=True,
    )
    clusters.name = "clusters"
    return clusters


def set_clusters_of(
    ds: xr.Dataset,
    to_cluster_name: str,
    cluster_name: str | None,
    *,
    min_samples: int = 3,
    xi: float = 0.1,
    min_cluster_size: float | int = 0.01,
    max_eps: float = 10.0,
    force: bool = False,
) -> xr.DataArray | None:
    """Set the clusters of a dataset.

    Parameters
    ----------
    ds
        Dataset to set the clusters of.
    to_cluster_name
        The name of the variable to cluster.
    cluster_name
        The name of the resulting cluster array.
        If None, the array is returned.
    min_samples
        Argument passed to `sklearn.cluster.OPTICS`.
    xi
        Argument passed to `sklearn.cluster.OPTICS`.
    min_cluster_size
        Argument passed to `sklearn.cluster.OPTICS`.
    max_eps
        Argument passed to `sklearn.cluster.OPTICS`.
    force
        Perform the clustering even if the ``to_cluster`` array
        contains only False values.
    """
    if not force and ds[to_cluster_name].sum() <= min_samples:
        raise ThresholdException(
            f"Number of samples in `ds.{to_cluster_name}` is less"
            f" than `min_samples={min_samples}`. Either lower the threshold"
            " or reduce `min_samples`.",
        )
    data_clusters = _xr_data_clusters(
        ds[to_cluster_name],
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size,
        max_eps=max_eps,
    )
    if cluster_name is None:
        return data_clusters
    else:
        ds[cluster_name] = data_clusters


CLUSTER_INFO_VARIABLES = (
    "cluster_B_min",
    "cluster_B_max",
    "cluster_V_min",
    "cluster_V_max",
    "cluster_B_center",
    "cluster_V_center",
    "cluster_B_size",
    "cluster_V_size",
    "cluster_area",
)


class ClusterInfo(NamedTuple):
    """Information about a cluster.

    Contains bounding box, center coordinate, and relative area.
    """

    bounding_box: tuple[float, float, float, float] = 4 * (np.nan,)
    center: tuple[float, float] = 2 * (np.nan,)
    size: tuple[float, float] = 2 * (np.nan,)
    area: float = np.nan
    npixels: int = np.nan

    def as_xarray(self: ClusterInfo) -> xr.Dataset:
        """Convert namedtuple to `xarray.Dataset`."""
        (x_min, y_min, x_max, y_max) = self.bounding_box
        (x_center, y_center) = self.center
        (x_size, y_size) = self.size
        info = {
            "cluster_B_min": {"data": x_min, "dims": []},
            "cluster_B_max": {"data": x_max, "dims": []},
            "cluster_V_min": {"data": y_min, "dims": []},
            "cluster_V_max": {"data": y_max, "dims": []},
            "cluster_B_center": {"data": x_center, "dims": []},
            "cluster_V_center": {"data": y_center, "dims": []},
            "cluster_B_size": {"data": x_size, "dims": []},
            "cluster_V_size": {"data": y_size, "dims": []},
            "cluster_area": {"data": self.area, "dims": []},
            "cluster_npixels": {"data": self.npixels, "dims": []},
        }
        return xr.Dataset.from_dict(info)


def cluster_info(
    cluster: xr.DataArray,
    plunger_gate_name: str = "V",
    field_name: str = "B",
    pct_box: Number = 5,
    min_margin_box: tuple[Number, Number] = (0, 0),
    *,
    as_xarray: bool = False,
) -> ClusterInfo | xr.Dataset:
    """Get the bounding box and center of a cluster.

    The size of the bounding box is determined by the size of the cluster
    with a margin around it. This margin is at least ``pct_box``% larger, or
    ``min_margin_box`` (in physical units), whichever is larger.

    Parameters
    ----------
    cluster
        Data array of the cluster.
    plunger_gate_name
        Plunger gate name.
    field_name
        Magnetic field name.
    pct_box
        Percentage of the cluster size to add to the bounding box.
    min_margin_box
        The minimum margin to add to the bounding box in physical units.
    as_xarray
        Return the data as `xarray.Dataset` instead of `ClusterInfo` namedtuple.

    Returns
    -------
    ClusterInfo namedtuple.
    """
    if not cluster.any():
        info = ClusterInfo()
        return info.as_xarray() if as_xarray else info
    cluster = cluster.squeeze().transpose(field_name, plunger_gate_name)
    p = cluster.mean(field_name)
    p = p[p > 0].coords[plunger_gate_name]
    p_left, p_right = p.min(), p.max()

    f = cluster.mean(plunger_gate_name)
    f = f[f > 0].coords[field_name]
    f_left, f_right = f.min(), f.max()

    dy = np.abs(p_right - p_left) * pct_box / 100
    dx = np.abs(f_right - f_left) * pct_box / 100
    dx = max(dx, min_margin_box[0])
    dy = max(dy, min_margin_box[1])
    bounding_box = (f_left - dx, p_left - dy, f_right + dx, p_right + dy)

    center = ndimage.center_of_mass(cluster.data)

    mid_inds = dict(zip(cluster.dims, center))
    mid_inds = {k: round(v) for k, v in mid_inds.items()}

    mid = cluster.isel(**mid_inds)
    center = (mid.coords[field_name].data, mid.coords[plunger_gate_name].data)

    dB = cluster.B.values[1] - cluster.B.values[0]
    dV = cluster.V.values[1] - cluster.V.values[0]
    np.testing.assert_allclose(dB, np.diff(cluster.B.values))
    np.testing.assert_allclose(dV, np.diff(cluster.V.values))
    npixels = cluster.sum(["B", "V"])
    area = npixels * dB * dV

    bounding_box = tuple(map(float, bounding_box))
    center = tuple(map(float, center))
    size = (float(f_right - f_left), float(p_right - p_left))
    info = ClusterInfo(bounding_box, center, size, float(area))
    return info.as_xarray() if as_xarray else info


def cluster_infos(
    clusters: xr.DataArray,
    plunger_gate_name: str = "V",
    field_name: str = "B",
    pct_box: Number = 5,
    min_margin_box: tuple[Number, Number] = (0, 0),
    dim: str = "cluster",
) -> list[ClusterInfo]:
    """Get the bounding box and center of an array of clusters.

    Parameters
    ----------
    clusters
        Data array of the clusters along dimension ``dim``.
    plunger_gate_name
        Plunger gate name.
    field_name
        Magnetic field name.
    pct_box
        Percentage of the cluster size to add to the bounding box.
    min_margin_box
        The minimum margin to add to the bounding box in physical units.

    Returns
    -------
    list[ClusterInfo]
        List of ClusterInfo namedtuples.
    """
    clusters = expand_clusters(clusters).transpose(dim, ...)
    return [
        cluster_info(cl, plunger_gate_name, field_name, pct_box, min_margin_box)
        for cl in clusters
    ]


def xr_map(
    da: xr.DataArray,
    f: Callable,
    dim: str,
    kwargs: dict[str, Any] | None = None,
) -> xr.DataArray:
    """Map function over dimension and concatenate the results."""
    if kwargs is None:
        kwargs = {}
    _f = functools.partial(f, **kwargs)
    return xr.concat([_f(da.sel({dim: i})) for i in da[dim]], dim)


def groupby_map(
    func: Callable[..., Any],
    da: xr.DataArray,
    exclude_dims: tuple[str, ...] = (),
    **kwargs,
) -> xr.DataArray | xr.Dataset:
    """Map a function over a groupby object."""
    other_dims = [d for d in da.dims if d not in exclude_dims]
    if other_dims:
        da = da.stack(new=other_dims).groupby("new")
        return da.map(func, **kwargs).unstack("new")
    else:
        return func(da, **kwargs)


def code_metadata() -> dict[str, str]:
    """Get metadata for the current analysis."""
    return {
        "metadata.version.python": platform.python_version(),
        "metadata.version.tgp": tgp.__version__,
        "metadata.version.scipy": scipy.__version__,
        "metadata.version.numpy": numpy.__version__,
        "metadata.version.sklearn": sklearn.__version__,
        "metadata.version.xarray": xr.__version__,
        "metadata.user": getpass.getuser(),
        "metadata.date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metadata.platform": platform.platform(),
    }
