# Copyright (c) Microsoft Corporation. All rights reserved.

from __future__ import annotations

from copy import deepcopy
import fnmatch
from pathlib import Path
from typing import Any, Literal
import warnings

import numpy as np
from ruamel.yaml import YAML
from scipy.constants import physical_constants
from scipy.interpolate import RegularGridInterpolator, splev, splrep
import scipy.signal
from scipy.signal import fftconvolve
import xarray as xr

from tgp.common import code_metadata


def _set_attrs(ds: xr.Dataset) -> None:
    ds.B.attrs = {"long_name": "Magnetic field", "unit": "T"}
    ds.V.attrs = {"long_name": "Plunger gate voltage", "unit": "V"}


def prepare(
    ds: xr.Dataset,
    fillna: bool = True,
    add_cutter_pair_index: bool = True,
) -> xr.Dataset:
    """Prepare datasets for analysis.

    Rename the data variables and dimensions occurring to a
    naming scheme defined in ``tgp/data_naming.yaml``.

    Parameters
    ----------
    ds
        Raw dataset.
    fillna
        Replace NaNs with zeros if True.
    add_cutter_pair_index
        Whether to add a dimension "cutter_pair_index" in case it is missing.

    Returns
    -------
    xr.Dataset
        Renamed dataset.
    """
    ds = rename_and_transform(ds)
    if add_cutter_pair_index and "cutter_pair_index" not in ds.dims:
        print(
            "The dimension 'cutter_pair_index' is not present, it's being added automatically.",
        )
        ds = ds.expand_dims({"cutter_pair_index": [0]})
    args = [k for k in ("cutter_pair_index", "B", "V") if k in ds]
    ds = ds.transpose(*args, ...)
    _set_attrs(ds)

    if "I_1w_L" in ds:
        ds["L_1w"] = np.real(ds["I_1w_L"])
        ds["R_1w"] = np.real(ds["I_1w_R"])
    if "I_2w_L" in ds:
        ds["L_2w"] = np.real(ds["I_2w_L"])
        ds["R_2w"] = np.real(ds["I_2w_R"])
    if "I_2w_LR" in ds:
        ds["R_2w_nl"] = np.real(ds["I_2w_LR"])
        ds["L_2w_nl"] = np.real(ds["I_2w_RL"])
    if "I_3w_L" in ds:
        ds["L_3w"] = np.real(ds["I_3w_L"])
        ds["R_3w"] = np.real(ds["I_3w_R"])

    ds.attrs.update(code_metadata())

    return ds.fillna(0) if fillna else ds


def _temp_kernel(bias: np.ndarray, T: float) -> np.ndarray:
    beta = 1.0 / T
    K = beta * np.exp(-beta * np.abs(bias)) / (1.0 + np.exp(-beta * np.abs(bias))) ** 2
    # this is important - kernel needs to be normalized to one but because of a finite range in bias
    # the norm is not always respected at low temperatures
    K = K / np.trapz(x=bias, y=K)
    return K


def broaden_with_temperature(
    ds: xr.Dataset,
    T_mK: float,
    bias_name: str = "bias",
    keys: tuple[str, ...] = ("g_ll", "g_lr", "g_rl", "g_rr"),
) -> xr.Dataset:
    """Broaden the data with a temperature dependent kernel.

    Manipulate the data arrays in ``ds``.

    Parameters
    ----------
    ds
        Dataset where ``keys`` are the data variables to broaden.
    T_mK
        Temperature in mK.
    bias_name
        Name of the bias dimension.
    keys
        Data array keys to broaden.

    Returns
    -------
    xarray.Dataset
        Dataset with broadened data.
    """
    k_B = physical_constants["Boltzmann constant in eV/K"][0]
    T_meV = k_B * (T_mK * 1e-3) * 1e3

    for k in keys:
        if np.isnan(ds[k]).sum() > 0:
            ds[k] = ds[k].interpolate_na(dim="V")

    if T_meV == 0.0:
        return ds

    bias = ds[bias_name].values
    bias_max = max(2.5 * bias[-1], 10 * T_meV)
    bias_step = np.min(np.unique(np.diff(bias)))
    npts_bias = int(round((2 * bias_max) / bias_step)) // 2 * 2 + 1
    bias_reg = np.linspace(-bias_max, bias_max, npts_bias)

    assert np.allclose(np.amin(np.abs(bias_reg)), 0.0), "Grid must go through zero"

    ds_out = ds.copy()

    dims_old = {k: ds[k].shape[0] for k in ds[keys[0]].dims}
    for k in dims_old:
        dims_old[k] = ds[k].values
    dims_new = dims_old.copy()
    dims_new["bias"] = bias_reg

    def _pts_shape(
        dims: dict[str, np.ndarray],
    ) -> tuple[RegularGridInterpolator, np.ndarray, tuple[int, ...]]:
        vals = tuple(dims.values())
        pts = np.reshape(
            np.meshgrid(*vals, indexing="ij"),
            (len(dims), -1),
            order="C",
        ).T
        shape = tuple(v.shape[0] for v in dims.values())
        ip = RegularGridInterpolator(
            vals,
            np.zeros(shape),
            bounds_error=False,
            fill_value=None,
        )
        return ip, pts, shape

    interp, pts_old, shape_old = _pts_shape(dims_old)
    interp_back, pts_new, shape_new = _pts_shape(dims_new)

    size = 1
    bias_index = None
    for i, (k, v) in enumerate(dims_new.items()):
        if k != "bias":
            size *= v.shape[0]
        else:
            bias_index = i

    K = _temp_kernel(bias_reg, T_meV)
    K_conv = np.outer(np.ones(size), K).reshape(shape_new)

    for x in keys:
        interp.values = ds[x].values
        g_new = interp(pts_new).reshape(shape_new)
        g_conv = (
            fftconvolve(K_conv, g_new, axes=bias_index, mode="same")
            * np.diff(bias_reg)[0]
        )
        interp_back.values = g_conv
        g_conv = interp_back(pts_old).reshape(shape_old)
        ds_out[x] = xr.DataArray(
            g_conv,
            coords=dims_old,
            dims=list(dims_old.keys()),
        )
    ds_out.attrs["broaden_with_temperature.T_mK"] = T_mK
    return ds_out


def _derivative(
    da: xr.DataArray,
    order: int = 1,
    bias_name: str = "bias",
) -> xr.DataArray:
    bias = da[bias_name].values
    return xr.apply_ufunc(
        lambda x: splev(0.0, splrep(bias, x, k=2), der=order),
        da,
        input_core_dims=[[bias_name]],
        output_core_dims=[[]],
        vectorize=True,
    )


def add_2w_3w(ds: xr.Dataset) -> None:
    """Add 2w and 3w data to the dataset based on derivates of the conductance."""
    ds["I_2w_LR"] = _derivative(ds.g_lr, order=1)
    ds["I_2w_RL"] = _derivative(ds.g_rl, order=1)
    ds["I_3w_L"] = _derivative(ds.g_ll, order=2)
    ds["I_3w_R"] = _derivative(ds.g_rr, order=2)


def prepare_sim(ds_raw: xr.Dataset, T_mK: float) -> xr.Dataset:
    """Prepare the data for a simulation and temperature broaden the data."""
    ds = rename(ds_raw)
    ds = broaden_with_temperature(ds, T_mK)
    ds.attrs["T_mK"] = T_mK
    add_2w_3w(ds)
    return prepare(ds)


def _match_names(x: str, mapping: dict[str, list[str]]) -> str | None:
    matches = []
    for new, olds in mapping.items():
        for old in [*olds, new]:
            if fnmatch.fnmatchcase(x, old):
                matches.append(new)
    if not matches:
        return None
    if len(set(matches)) > 1:
        raise RuntimeError(f"matches for {x} are {matches}")
    return matches[0]


def _renames(ds: xr.Dataset, mapping: dict[str, list[str]]) -> dict[str, str]:
    renames = {}
    for k in ds.dims.keys() | ds.variables.keys():
        m = _match_names(k, mapping)
        if m is not None and m != k:
            renames[k] = m
    new_names = list(renames.values())
    assert len(new_names) == len(set(new_names))
    return renames


def _transform(ds: xr.Dataset) -> xr.Dataset:
    has_LC = "V_leftcutter" in ds.coords
    has_RC = "V_rightcutter" in ds.coords
    has_RB = "left_bias" in ds.coords
    has_LB = "right_bias" in ds.coords

    if "cutter_pair_index" in ds.coords or "cutter_pair_index" in ds.dims:
        # Nothing to do
        pass
    elif has_LC and has_RC:
        ds = ds.stack(cutter_pair_index=["V_leftcutter", "V_rightcutter"])
        ds = ds.reset_index("cutter_pair_index")
    elif "cutter_pair_left" in ds.coords and "cutter_pair_right" in ds.coords:
        ds = ds.stack(cutter_pair_index=["cutter_pair_left", "cutter_pair_right"])
        ds = ds.reset_index("cutter_pair_index")
    elif (has_LC or has_RC) and len(ds.dims) == 3:
        # Only one of them is defined
        return
    elif (has_RB or has_LB) and "V" in ds.coords and len(ds.dims) == 3:
        # Dataset where the cutter gate has been averaged out
        # or measured at a single cutter value. Some phase 2 data is like this.
        pass
    elif len(ds.dims) == 2:
        # Skip because data is incomplete
        return
    else:
        raise ValueError(f"Unexpected naming convention for {ds}")
    return ds


def _load_yaml(fname: Path | str) -> dict[str, Any]:
    fname = Path(fname)
    with fname.open() as f:
        return YAML(typ="safe").load(f)


def rename(
    ds: xr.Dataset,
    mapping: dict[str, list[str]] | str | Path | None = None,
) -> xr.Dataset:
    """Rename the variables in the dataset to a standardized convention.

    Based on the naming convention defined in ``tgp/data_naming.yaml``.
    """
    if mapping is None:
        fname = Path(__file__).parent / "data_naming.yaml"
        mapping = _load_yaml(fname)
    elif isinstance(mapping, (Path, str)):
        mapping = _load_yaml(mapping)
    renames = _renames(ds, mapping)
    return ds.rename(renames)


def rename_and_transform(ds: xr.Dataset) -> xr.Dataset:
    """Rename the variables in the dataset to a standardized convention and transforms the data.

    Parameters
    ----------
    ds
        Raw `xarray.Dataset`.

    Returns
    -------
    xarray.Dataset
        Transformed and renamed `xarray.Dataset`.

    Raises
    ------
    ValueError
        If the dataset is not in a recognized format.
    """
    ds = rename(ds)
    ds = _transform(ds)
    if ds is None:
        raise ValueError("Cannot transform dataset.")
    return ds


def _find_peaks_or_dips_in_trace(
    x: np.ndarray,
    y: np.ndarray,
    relative_prominence: float,
    relative_height: float,
    peak_params: dict[str, Any],
    filter_params: dict[str, int] = None,
    dips: bool = False,
    *,
    full_output: bool = False,
) -> dict[str, Any]:
    """Find peaks in a 1D trace.

    Parameters
    ----------
    x
        Array of values on the x-axis.
    y
        Array of values on the y-axis.
    relative_prominence
        Relative threshold for the prominence of peaks, number between 0 and 1.
    relative_height
        Relative threshold for peak heights, number between 0 and 1.
    peak_params
        Dictionary of parameters to be passed to find_peaks.
    filter_params
        Dictionary of parameters to be passed to savgol_filter:
        ``dict(window_length=..., polyorder=...)``.
    dips
        If True, looks for dips instead of peaks.
    full_output
        Return the x and y (+filtered) values, by default False.

    Returns
    -------
    dict
        Dictionary with peak information.
    """
    if np.isnan(y).any():
        # If the 1D trace contains NaN, we ignore it and go on,
        # scipy.signal.find_peaks is ill behaved with NaNs.
        return np.nan

    out = {}
    if filter_params is not None:
        yf = scipy.signal.savgol_filter(y, **filter_params)  # smoothen raw data
    else:
        yf = y
    prominence = np.ptp(yf) * relative_prominence
    y_dip = -yf if dips else yf
    height = np.ptp(yf) * relative_height + np.min(y_dip)
    out["i_peaks"], out["properties"] = scipy.signal.find_peaks(
        y_dip,
        height=height,
        prominence=prominence,
        **peak_params,
    )
    out["x_peaks"] = x[out["i_peaks"]]
    out["y_peaks"] = yf[out["i_peaks"]]
    out["n_peaks"] = len(out["i_peaks"])

    if full_output:
        out["x"] = x
        out["y"] = y
        if filter_params is not None:
            out["y_filtered"] = yf
    return out


def find_symmetry_axis_2D(
    da: xr.DataArray,
    along_dim: str,
    return_spectrum: bool = False,
) -> dict[str, np.ndarray] | tuple[float, float]:
    """Find the most symmetric axis along a dimension.

    Takes inner products of all rows (with the data transposed such
    that [along_dim, ...]) and sums over all diagonals.
    This results in a spectrum of overlaps which is then fed to a
    peak finding algorithm. The position of the most prominent peak
    is returned.

    Parameters
    ----------
    da
        2D data array.
    along_dim
        Dimension along which the symmetry line is searched.
    return_spectrum
        If True, a dict with dim-coords and a spectrum is returned.
        If False the position of the most prominent peak is returned.

    Returns
    -------
    dict or float
        See `return_spectrum` argument.
    """
    da_norm = da - da.mean(dim=along_dim)
    z_norm = da_norm.transpose(along_dim, ...).data
    weights = z_norm @ z_norm.T

    def _sum_of_diags(X: np.ndarray) -> np.ndarray:
        N = X.shape[1]
        rot = np.rot90(X)
        return np.array([np.diag(rot, i).sum() for i in range(1 - N, N)])

    sums = _sum_of_diags(weights)
    x = da[along_dim].values
    x_new = np.linspace(x[0], x[-1], len(sums))
    if return_spectrum:
        return {"x": x_new, "y": sums}
    peak_info = _find_peaks_or_dips_in_trace(
        x_new,
        sums,
        relative_prominence=0.04,
        relative_height=0.04,
        filter_params={"window_length": 21, "polyorder": 2},
        peak_params={},
    )
    _, peak_pos = max(zip(peak_info["properties"]["prominences"], peak_info["x_peaks"]))
    return peak_pos


def _closest(x: np.ndarray, val: float) -> tuple[int, float]:
    i = np.argmin(np.abs(x - val))
    return i, x[i]


def _symmetrize_bias(
    ds: xr.Dataset,
    cond: str,
    bias: str,
    B: str = "B",
    method: Literal["by_spectrum", "by_offset"] = "by_spectrum",
    offset: float | None = None,
    return_correction: bool = False,
) -> xr.Dataset | tuple[xr.Dataset, float]:
    """Symmetrize bias in a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        dataset to analyze
    cond : str
        name of conductance variable to analyze
    bias : str
        name of bias variable
    B : str, optional
        name of magnetic field variable
    method,  Literal["by_spectrum", "by_offset"]
        Use "by_spectrum" for automatic bias correction using spectral analysis method,
        use "by_offset" to correct for fixed value.
    offset, float, optional
        Fixed offset in volts to correct, ignored if method is not "by_offset"
    return_correction : bool, optional
        If False, returns only symmetrized dataset, if True returns the dataset and correction offset, by default False

    Returns
    -------
    xr.Dataset or (xr.Dataset, float)
        symmetrized dataset and (if return_correction is True) bias correction value
    """
    OFFSET_PERCENTAGE_TO_WARN = 0.25

    g = ds[[var for var in ds.data_vars if var.startswith("g_")]].mean(
        [dim for dim in ds[cond].dims if dim not in {bias, B}],
    )
    i_zero, _ = _closest(ds[bias].values, 0)
    if method == "by_spectrum":
        offset_in_V = find_symmetry_axis_2D(g[cond], bias)
    elif method == "by_offset":
        offset_in_V = offset

    bias_magnitude = np.ptp(ds[bias].values)
    bias_mean = ds[bias].mean()
    if offset_in_V - bias_mean > bias_magnitude * OFFSET_PERCENTAGE_TO_WARN:
        warnings.warn(
            f"Offset in {bias} is too large, consider 'manual' or 'by_offset' correction methods. "
            f"Found offset is {offset_in_V} V, "
            f"while dataset has bias magnitute {bias_magnitude} V and bias mean {bias_mean} V"
        )

    i_mid, correction = _closest(ds[bias].values, offset_in_V)
    if i_mid > i_zero:
        ds = ds.isel({bias: slice(2 * i_mid - ds.dims[bias], None)})
    elif i_mid < i_zero:
        ds = ds.isel({bias: slice(0, i_mid * 2 + 1)})
    ds[bias] = ds[bias].values - correction
    return ds if not return_correction else ds, correction


def correct_bias_auto(
    ds_left: xr.Dataset,
    ds_right: xr.Dataset,
    norm: float = 1.0,
    *,
    offset: float | tuple[float, float] | None = None,
    method: Literal["by_spectrum", "by_offset"] = "by_spectrum",
    add_attrs: bool = True,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Symmetrize the conductance wrt the bias.

    Based on an algorithm that finds the most symmetric axis.

    Parameters
    ----------
    ds_left
        Dataset taken on the left.
    ds_right
        Dataset taken on the right.
    norm
        Multiplicative factor for the bias.
    offset, float or tuple[float, float], optional
        Fixed offset in volts to correct. Can be tuple for (left, right) offsets.
        Ignored if method is not "by_offset"
    method
        Use "by_spectrum" for automatic bias correction using spectral analysis method,
        or "manual" for manual bias correction.
    add_attrs
        Whether to add the input parameters as attributes to the dataset.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        Left and right datasets with corrected bias.
    """
    if method == "auto":
        warnings.warn(
            DeprecationWarning(
                "method='auto' is deprecated and renamed to 'by_spectrum', use that instead"
            ),
        )
        method = "by_spectrum"
    if method == "by_offset":
        assert (
            offset is not None
        ), "Please provide 'offset' parameter for 'by_offset' method"
        if not isinstance(offset, tuple):
            offset_left, offset_right = offset, offset
        else:
            offset_left, offset_right = offset
    else:
        if offset is not None:
            warnings.warn(f"Offset value is ignored with {method=}", Warning)
        offset_left, offset_right = None, None
    ds_left, corr_left = _symmetrize_bias(
        ds_left,
        "g_ll",
        "left_bias",
        method=method,
        offset=offset_left,
        return_correction=True,
    )
    ds_right, corr_right = _symmetrize_bias(
        ds_right,
        "g_rr",
        "right_bias",
        method=method,
        offset=offset_right,
        return_correction=True,
    )
    ds_left["left_bias"] = norm * ds_left["left_bias"]
    ds_right["right_bias"] = norm * ds_right["right_bias"]
    prefix = "correct_bias_auto"
    ds_left.attrs[f"{prefix}.out.bias_correction"] = norm * corr_left
    ds_right.attrs[f"{prefix}.out.bias_correction"] = norm * corr_right
    if add_attrs:
        ds_left.attrs[f"{prefix}.norm"] = norm
        ds_right.attrs[f"{prefix}.norm"] = norm
        ds_left.attrs[f"{prefix}.method"] = method
        ds_right.attrs[f"{prefix}.method"] = method
        ds_left.attrs[f"{prefix}.offset"] = offset_left
        ds_right.attrs[f"{prefix}.offset"] = offset_right
    return ds_left, ds_right


def correct_bias_manual(
    ds_left: xr.Dataset,
    ds_right: xr.Dataset,
    drop_indices: list[int] | tuple[list[int], list[int]],
    max_bias_index: int | tuple[int, int],
    norm: float = 1.0,
    *,
    add_attrs: bool = True,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Correct the bias manually.

    Parameters
    ----------
    ds_left
        Dataset taken on the left.
    ds_right
        Dataset taken on the right.
    drop_indices
        Drop indices from the bias. Can be a tuple of
        two lists of indices, one for each dataset (left and right).
    max_bias_index
        Index of the maximum bias. Can be a tuple of
        two indices, one for each dataset (left and right).
    norm
        Multiplicative factor for the bias.
    add_attrs
        Whether to add the input parameters as attributes to the dataset.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        Left and right datasets with corrected bias.
    """
    datasets = []

    if not isinstance(drop_indices, tuple):
        assert isinstance(drop_indices, list)
        drop_indices = (drop_indices, drop_indices)
    if not isinstance(max_bias_index, tuple):
        max_bias_index = (max_bias_index, max_bias_index)

    for ds, bias, _drop_ind, _i in zip(
        [ds_left, ds_right],
        ["left_bias", "right_bias"],
        drop_indices,
        max_bias_index,
    ):
        ds = deepcopy(ds)
        if _drop_ind:
            ds = ds.drop_isel(**{bias: _drop_ind})
        length = len(ds[bias])
        max_bias = np.round(ds[bias][_i] * norm, 5)
        ds[bias] = np.linspace(-max_bias, max_bias, num=length)
        datasets.append(ds)
    _ds_left, _ds_right = datasets

    if add_attrs:
        prefix = "correct_bias_manual"
        attrs = {
            f"{prefix}.drop_indices": drop_indices,
            f"{prefix}.max_bias_index": "max_bias_index",
            f"{prefix}.norm": norm,
        }
        _ds_left.attrs.update(attrs)
        _ds_right.attrs.update(attrs)
    return _ds_left, _ds_right


def correct_bias(
    ds_left: xr.Dataset,
    ds_right: xr.Dataset,
    drop_indices: list[int] | tuple[list[int], list[int]] | None = None,
    max_bias_index: int | tuple[int, int] | None = None,
    norm: float = 1.0,
    offset: float | tuple[float, float] | None = None,
    method: Literal["by_spectrum", "by_offset", "manual"] = "manual",
) -> tuple[xr.Dataset, xr.Dataset]:
    """Symmetrize the conductance wrt the bias automatically or manually.

    Parameters
    ----------
    ds_left
        Dataset taken on the left.
    ds_right
        Dataset taken on the right.
    drop_indices
        Drop indices from the bias. Can be a tuple of
        two lists of indices, one for each dataset (left and right).
    max_bias_index
        Index of the maximum bias. Can be a tuple of
        two indices, one for each dataset (left and right).
    offset, float or tuple[float, float], optional
        Fixed offset in volts to correct. Can be tuple for (left, right) offsets.
        Ignored if method is not "by_offset"
    norm
        Multiplicative factor for the bias.
    method
        Use "by_spectrum" for automatic bias correction using spectral analysis method,
        or "manual" for manual bias correction.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        Left and right datasets with corrected bias.
    """
    prefix = "correct_bias"
    attrs = {
        f"{prefix}.drop_indices": drop_indices,
        f"{prefix}.max_bias_index": max_bias_index,
        f"{prefix}.norm": norm,
        f"{prefix}.method": method,
        f"{prefix}.offset": offset,
    }
    ds_left.attrs.update(attrs)
    ds_right.attrs.update(attrs)
    if method in ("by_spectrum", "by_offset", "auto"):
        return correct_bias_auto(
            ds_left,
            ds_right,
            norm,
            method=method,
            offset=offset,
            add_attrs=False,
        )
    elif method == "manual":
        assert drop_indices is not None and max_bias_index is not None
        return correct_bias_manual(
            ds_left,
            ds_right,
            drop_indices,
            max_bias_index,
            norm,
            add_attrs=False,
        )
    else:
        raise ValueError(f"method {method} not understood")
