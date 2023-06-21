# Copyright (c) Microsoft Corporation. All rights reserved.
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from ruamel.yaml import YAML
import xarray as xr

from tgp.frequency_correction import convert_conductances_array, generate_mapping


def correct_high_frequency_three_terminal(
    dataset: xr.Dataset,
    frequency_left: float | None = None,
    frequency_left_label: str = "frequency_left",
    frequency_right: float | None = None,
    frequency_right_label: str = "frequency_right",
    fridge_parameters: dict | None = None,
    fridge_parameters_file: str | Path | None = None,
    g_ll_name: str = "g_ll",
    g_lr_name: str = "g_lr",
    g_rl_name: str = "g_rl",
    g_rr_name: str = "g_rr",
    left_bias_name: str = "left_bias",
    right_bias_name: str = "right_bias",
    left_dc_current_name: str = "left_dc_current",
    right_dc_current_name: str = "right_dc_current",
    dc_sign: int = 1,
    ac_sign: int = 1,
) -> xr.Dataset:
    """Correct voltage divider effects for high-frequency conductance measurements
    on three-terminal devices.

    .. note::
       Some parameters are abbreviated see call signature for full list.

    Parameters
    ----------
    dataset
        Raw dataset with conductance (e^2/h) and DC current (Amperes) measurements.
    frequency_k
        Measurement frequency of side k AC bias.
    frequency_k_label
        If the frequency_k is a dimension of the dataset specify the label here and do
        not pass frequency_k to use the value in the dataset.
    fridge_parameters
        Fridge parameters as dictionary.
    fridge_parameters_file
        Path of yaml file specifying fridge parameters.
    g_ij_name
        Variable name for dI_i/dV_j.
    bias_k_name
        Variable name for bias_i value applied before line resistances.
    i_k_name
        Variable name for DC current on side k.
    dc_sign
        The sign of the measured DC current. If the current is
        measured as flowing *into* the device, this sign is positive.
        If the current measured is flowing *out* of the device, this
        sign is negative.
    ac_sign
        The sign of the measured AC current. If the current is
        measured as flowing *into* the device, this sign is positive.
        If the current measured is flowing *out* of the device, this
        sign is negative.
    """
    if fridge_parameters is None:
        if fridge_parameters_file is None:
            raise ValueError(
                "Either fridge_parameters or fridge_parameters_file"
                "have to be provided, none have been provided",
            )
        with open(fridge_parameters_file) as f:
            fridge_parameters = dict(YAML(typ="safe").load(f))

    if frequency_left is None:
        if frequency_left_label not in dataset.dims:
            raise ValueError(
                f"Frequency_left not provided and unable to find"
                f"{frequency_left_label} in dataset dimensions,"
                "one of these must be provided",
            )
        frequency_left = float(dataset[frequency_left_label])

    if frequency_right is None:
        if frequency_right_label not in dataset.dims:
            raise ValueError(
                f"Frequency_right not provided and unable to find"
                f"{frequency_left_label} in dataset dimensions,"
                "one of these must be provided",
            )
        frequency_right = float(dataset[frequency_right_label])

    g_names = [[g_ll_name, g_lr_name], [g_rl_name, g_rr_name]]
    v_names = [left_bias_name, right_bias_name]
    i_names = [left_dc_current_name, right_dc_current_name]

    # Pad bias coord with zeros if needed:
    for v_name in v_names:
        if v_name not in dataset.dims:
            dataset = dataset.expand_dims(v_name)
            dataset = dataset.assign_coords({v_name: [0.0]})
            dataset.coords[v_name].attrs["long_name"] = v_name
            dataset.coords[v_name].attrs["units"] = "V"

    # make the low frequency correction
    def get_line_r(fridge_parameters: dict, side: str) -> float:
        r_vals = [
            stage[f"Resistance_{side}"]
            for stage in fridge_parameters["line_properties"].values()
        ]
        return float(np.sum(r_vals))

    r_left, r_right, r_gnd = (
        get_line_r(fridge_parameters, side) for side in ["L", "R", "D"]
    )

    circuit = ThreeTerminalCircuit(
        r_1=r_left,
        r_2=r_right,
        r_gnd=r_gnd,
        dc_sign=dc_sign,
        ac_sign=ac_sign,
    )
    transformer = GDataset(g_names, v_names, i_names, circuit=circuit)
    corrected_dataset = transformer.correct(dataset)
    corrected_dataset.attrs = dataset.attrs

    # high-freq correction:
    map_left = generate_mapping(frequency_left, fridge_parameters)
    map_right = generate_mapping(frequency_right, fridge_parameters)
    (
        corrected_dataset[g_ll_name],
        corrected_dataset[g_rl_name],
        corrected_dataset[g_lr_name],
        corrected_dataset[g_rr_name],
    ) = convert_conductances_array(
        dataset[g_ll_name],
        dataset[g_rl_name],
        dataset[g_lr_name],
        dataset[g_rr_name],
        map_left,
        map_right,
    )

    # discard imaginary part of lock-in signal
    corrected_dataset = np.real(corrected_dataset)

    return corrected_dataset


class Circuit:
    """Rescale a bias parameter with the measured DC current.

    Parameters
    ----------
    line_r : sequence
        A (T - 1, T - 1) array with the line resistances used to
        model the circuit, where T is the number of terminals. This
        array should be built by a subclass.
    dc_sign : int
        The sign of the measured DC current. If the current is
        measured as flowing *into* the device, this sign is positive.
        If the current measured is flowing *out* of the device, this
        sign is negative.
    ac_sign : int
        The sign of the measured Ac current. If the current is
        measured as flowing *into* the device, this sign is positive.
        If the current measured is flowing *out* of the device, this
        sign is negative.
    """

    def __init__(
        self,
        line_r: Iterable[Iterable[float]],
        dc_sign: int = 1,
        ac_sign: int = 1,
    ) -> None:
        self.line_r = np.array(line_r)
        self.dc_sign = dc_sign
        self.ac_sign = ac_sign

    def rescale(self, v: np.ndarray, i: np.ndarray) -> np.ndarray:
        """Rescale a bias parameter with the measured DC current.

        Parameters
        ----------
        v : numpy.ndarray
            Bias voltage in V as a (N, T - 1) array, where N is the number
            of data points and T is the number of terminals.
        i : numpy.ndarray
            DC current in A as a (N, T - 1) array, where N is the number
            of data points and T is the number of terminals.
        """
        i *= self.dc_sign
        v_p = v - np.einsum("ij,...j->...i", self.line_r, i)
        return v_p

    def transform(self, g: np.ndarray) -> np.ndarray:
        """Transform a conductance matrix for voltage divider effects.

        Parameters
        ----------
        g : numpy.ndarray
            Conductance matrix in SI units (*not* natural units) as a
            (N, T - 1, T - 1) array, where N is the number of data points
            and T is the number of terminals.
        """
        dvp_dv = np.eye(2) - self.ac_sign * np.einsum("ij,...jd->...id", self.line_r, g)
        dv_dvp = np.linalg.inv(dvp_dv)
        di_dvp = np.einsum("...ij,...jk->...ik", g, dv_dvp)
        return di_dvp


class ThreeTerminalCircuit(Circuit):
    """A circuit model of a three-terminal setup including line resistances.

    The model represents such a circuit:

    .. code-block:: none

        GND --- v_1+ --- r_1 --- sample --- r_2 --- +v_2 --- GND
                                   |
                                   | --- r_gnd --- GND
        where + denotes the positive bias sides.

    Parameters
    ----------
    r_1 : float
        Resistance of line 1 in ohm.
    r_2 : float
        Resistance of line 2 in ohm.
    dc_sign : int
        The sign of the measured DC current. If the current is
        measured as flowing *into* the device, this sign is positive.
        If the current measured is flowing *out* of the device, this
        sign is negative.
    ac_sign : int
        The sign of the measured Ac current. If the current is
        measured as flowing *into* the device, this sign is positive.
        If the current measured is flowing *out* of the device, this
        sign is negative.
    """

    def __init__(
        self,
        r_1: float = 0.0,
        r_2: float = 0.0,
        r_gnd: float = 0.0,
        dc_sign: int = 1,
        ac_sign: int = 1,
    ) -> None:
        line_r = np.array([[r_1 + r_gnd, r_gnd], [r_gnd, r_2 + r_gnd]])
        super().__init__(line_r, dc_sign, ac_sign)


class GDataset:
    """A three-terminal conductance matrix dataset wrapper.

    This class transforms conductance matrix datasets to correct for voltage
    divider and bias axis rescaling effects.

    Parameters
    ----------
    g_names : list of list of str
        Conductance matrix names, as a matrix.
    v_names : list of str
        Bias voltage names, as a list.
    i_names : list of str
        DC current names, as a list.
    natural_units : bool
        If True, the conductance is assumed to be in units
        of e^2/h.
    """

    def __init__(
        self,
        g_names: Iterable[Iterable[str]],
        v_names: Iterable[str],
        i_names: Iterable[str],
        circuit: Circuit,
        natural_units: bool = True,
    ) -> None:
        self.g_names = np.array(g_names, dtype=str)
        self.v_names = np.array(v_names, dtype=str)
        self.i_names = np.array(i_names, dtype=str)
        self.circuit = circuit
        self.natural_units = natural_units

    def correct_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Correct a conductance pandas.DataFrame for circuit effects.

        The names of the original indices are kept, but their original values
        are overwritten with the corrected value. This is so that the corrected
        dataframe can be used just like the original dataframe.
        """
        index_names = data.index.names
        data = data.reset_index()
        data = self.pad_data(data)
        data = self.transform(data)
        data = self.rescale(data)
        data = data.set_index(index_names)
        return data

    def copy_metadata(self, from_dataset: xr.Dataset, to_dataset: xr.Dataset) -> None:
        """Copy long names and units from from_dataset to to_dataset.

        This method modifies the datasets in place.
        """
        to_dataset.attrs = from_dataset.attrs
        for variable in to_dataset:
            if variable in from_dataset:
                to_dataset[variable].attrs = from_dataset[variable].attrs
        for coord in to_dataset.coords:
            coord = str(coord)
            if coord in from_dataset.coords:
                to_dataset.coords[coord].attrs = from_dataset.coords[coord].attrs
                to_dataset.coords[coord + "_sample"].attrs = from_dataset.coords[
                    coord
                ].attrs

    def correct_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        """Correct a conductance xr.Dataset for circuit effects.

        The names and original values of the original dimensions are kept,
        and new logical coordinates are added with the corrected values, and
        _sample appended to the names of the old dimensions. This is so that
        the corrected dataframe can be indexed like the original dataframe.
        """
        dataframe = dataset.to_dataframe()
        index_names = dataframe.index.names
        corrected_dataframe = self.correct_dataframe(dataframe).reset_index()
        dataframe = dataframe.reset_index()
        for column in corrected_dataframe:
            new_name = column + "_sample" if column in index_names else column
            dataframe[new_name] = corrected_dataframe[column]
        dataframe = dataframe.set_index(index_names)
        new_dataset = dataframe.to_xarray()
        new_dataset = self.restore_variable_coordinates(dataset, new_dataset)
        new_index_names = [name + "_sample" for name in index_names]
        new_dataset = new_dataset.set_coords(new_index_names)
        self.copy_metadata(dataset, new_dataset)
        return new_dataset

    def correct(self, data: xr.Dataset | pd.DataFrame) -> xr.Dataset | pd.DataFrame:
        """Correct either an xarray.Dataset or a pandas.DataFrame."""
        if isinstance(data, xr.Dataset):
            return self.correct_dataset(data)
        elif isinstance(data, pd.DataFrame):
            return self.correct_dataframe(data)
        else:
            raise NotImplementedError(
                "Only Xarrays and Pandas Dataframes are supported.",
            )

    def pad_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Pad missing variables in the data with zeros."""
        all_names = (
            np.ravel(self.g_names).tolist()
            + self.v_names.tolist()
            + self.i_names.tolist()
        )
        for name in all_names:
            if name not in data:
                data[name] = 0.0
        return data

    def rescale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Rescale the bias coordinates of the dataframe."""
        v = data[self.v_names.tolist()].to_numpy(copy=True)
        i = data[self.i_names.tolist()].to_numpy(copy=True)
        v_sample = self.circuit.rescale(v, i)
        data[self.v_names.tolist()] = v_sample
        return data

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply a conductance transformation to a dataframe."""
        from scipy.constants import e, h

        shape = (len(data), 2, 2)
        flat_g_names = np.ravel(self.g_names)
        g = data[flat_g_names].to_numpy(copy=True).reshape(shape)
        if self.natural_units:
            g *= e**2 / h  # transform to SI units for Circuit
        g_sample = self.circuit.transform(g).reshape((len(data), 4))
        if self.natural_units:
            g_sample /= e**2 / h  # transform to natural units from Circuit
        data[flat_g_names] = g_sample
        return data

    def restore_variable_coordinates(
        self,
        original_dataset: xr.Dataset,
        new_dataset: xr.Dataset,
    ) -> xr.Dataset:
        """Restore the correct coordinates for all variables.

        The correction procedure
        involves conversion of an xarray dataset to a pandas dataframe and back to
        an xarray dataset. In the conversion to a dataframe, all variables obtain
        equal coordinates (MultiIndices), which is incorrect if there are variables
        with different coordinates in the dataset.

        Parameters
        ----------
        original_dataset
            xarray dataset before conversion.
        new_dataset
            xarray dataset after conversion.

        Returns
        -------
        restored_dataset
        """
        var_names = [
            var
            for var in list(original_dataset.variables)
            if var not in list(original_dataset.coords)
        ]

        for var_name in var_names:
            for coord_name in list(new_dataset[var_name].coords):
                # remove coordinates that did not exist in the original dataset and
                # are not zero-dimensional setpoints that the correction adds when
                # parameters are missing in the dataset
                if (coord_name not in list(original_dataset[var_name].coords)) and (
                    new_dataset[var_name][coord_name].size > 1
                ):
                    new_dataset[var_name] = new_dataset[var_name].isel(
                        {coord_name: 0},
                        drop=True,
                    )
        return new_dataset


def correct_three_terminal(
    dataset: xr.Dataset,
    r_left: float,
    r_right: float,
    r_gnd: float,
    g_ll_name: str = "g_ll",
    g_lr_name: str = "g_lr",
    g_rl_name: str = "g_rl",
    g_rr_name: str = "g_rr",
    left_bias_name: str = "left_bias",
    right_bias_name: str = "right_bias",
    left_dc_current_name: str = "left_dc_current",
    right_dc_current_name: str = "right_dc_current",
    dc_sign: int = 1,
    ac_sign: int = 1,
) -> xr.Dataset:
    """Correct voltage divider effects on three-terminal devices.

    Parameters
    ----------
    dataset : raw dataset with conductance and DC current measurements.
    r_k : line resistance on terminal k
    g_ij_name : variable name for dI_i/dV_j
    bias_k_name : variable name for bias_i value applied before line resistances
    i_k_name : variable name for DC current on side k
    dc_sign : int
        The sign of the measured DC current. If the current is
        measured as flowing *into* the device, this sign is positive.
        If the current measured is flowing *out* of the device, this
        sign is negative.
    ac_sign : int
        The sign of the measured AC current. If the current is
        measured as flowing *into* the device, this sign is positive.
        If the current measured is flowing *out* of the device, this
        sign is negative.
    """
    dataset = np.real(dataset)  # discard imaginary part of lock-in signal
    g_names = [[g_ll_name, g_lr_name], [g_rl_name, g_rr_name]]
    v_names = [left_bias_name, right_bias_name]
    i_names = [left_dc_current_name, right_dc_current_name]

    # Pad bias coord with zeros if needed:
    for v_name in v_names:
        if v_name not in dataset.dims:
            dataset = dataset.expand_dims(v_name)
            dataset = dataset.assign_coords({v_name: [0.0]})
            dataset.coords[v_name].attrs["long_name"] = v_name
            dataset.coords[v_name].attrs["units"] = "V"

    circuit = ThreeTerminalCircuit(
        r_1=r_left,
        r_2=r_right,
        r_gnd=r_gnd,
        dc_sign=dc_sign,
        ac_sign=ac_sign,
    )
    transformer = GDataset(g_names, v_names, i_names, circuit=circuit)
    corrected_dataset = transformer.correct(dataset)
    corrected_dataset.attrs = dataset.attrs
    return corrected_dataset
