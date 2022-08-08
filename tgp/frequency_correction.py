# Copyright (c) Microsoft Corporation. All rights reserved.

"""
Finite frequency voltage divider correction for the conductance matrix in three
terminal devices.

"""
from __future__ import annotations

import itertools
import json

import numpy as np
from scipy.constants import e, h
from toolz.dicttoolz import get_in
import xarray
import xarray as xr
import yaml


def generate_filter_stage_mapping_generic(
    frequency: float, R: np.ndarray, C: np.ndarray
) -> np.ndarray:
    """Generate a matrix representation of a filter stage with N input and output
    lines mapping the voltages and currents at the input lines to voltages and
    currents at the output lines.

    In the circuit, each line n is assumed to comprise of
    1. parallel capacitance C[n][n] from input n to ground
    2. parallel capacitance C[n][m] from input n to the input of line m
    3. resistance R[n] between the input and the output


    Parameters
    ----------
    frequency
        used voltage excitation freq
    R
        a array of resistances
    C
        a ndarray of capacitances

    Returns
    -------
    mapping of type numpy.ndarray
    """

    ports = len(R)
    M = np.zeros([2 * ports, 2 * ports], dtype=complex)
    frequency_multiplier = 2j * np.pi * frequency
    for k in range(ports):
        M[k, k] = 1 + frequency_multiplier * R[k] * C[k][k]
        M[k, k + ports] = -R[k]
        M[k + ports, k] = -frequency_multiplier * C[k][k]
        M[k + ports, k + ports] = 1

        for i in range(ports):
            if k != i:
                M[k, i] = -frequency_multiplier * R[k] * C[k][i]
                M[k + ports, i] = frequency_multiplier * C[k][i]

    return M


def generate_amplifier_circuit_3T(
    frequency: float,
    left_amp_input_imp_0: float,
    left_gain: float,
    left_gwbp: float,
    right_amp_input_imp_0: float,
    right_gain: float,
    right_gwbp: float,
    drain_imp: float = 50.0,
) -> np.ndarray:
    """
    Generate a 'filter stage' only comprising of the current preamplifier
    input impedances in a 3T setup. Capacitances are assumed to be
    negligible at this stage.

    Parameters
    ----------
    frequency
        AC excitation frequency (Hz).
    left_amp_input_imp_0
        Left amplifier input impedance in the zero-
        frequency limit.
    left_gain
        The gain setting of the left amplifier.
    left_gwbp
        The gain-bandwidth product of the left amplifier.
    right_amp_input_imp_0
        Right amplifier input impedance in the zero-
        frequency limit.
    right_gain
        The gain setting of the right amplifier.
    right_gwbp
        The gain-bandwidth product of the right amplifier.
    drain_imp
        The impedance from drain to ground in Ohm (default: 50).

    Returns
    -------
        a mapping of type numpy.ndarray
    """

    Zin_L = left_amp_input_imp_0 + frequency * left_gain / left_gwbp
    Zin_R = right_amp_input_imp_0 + frequency * right_gain / right_gwbp

    R = np.array([Zin_L, Zin_R, drain_imp])
    C = np.zeros([3, 3])

    return generate_filter_stage_mapping_generic(frequency, R, C)


def generate_filter_stage_3T(
    frequency: float,
    res_L: float,
    res_R: float,
    res_D: float,
    cap_L: float,
    cap_R: float,
    cap_D: float,
    cap_LR: float,
    cap_LD: float,
    cap_RD: float,
) -> np.ndarray:
    """Generate a mapping for a singlefilter stage in a 3T setup consisting
    of resistances and capacitances.

    Parameters
    ----------
    frequency
        Frequency in Hz
    res_L
        Resistance left side of the device
    res_R
        Resistance right side of the device
    res_D
        Resistance drain
    cap_L
        Capacitance left side of the device
    cap_R
        Capacitance right side of the device
    cap_D
        Capacitance drain
    cap_LR
        Cross-capacitance between left and right side
    cap_LD
        Cross-capacitance between left side and drain
    cap_RD
        Cross-capacitance between right side and drain

    Returns
    -------
    a mapping of type numpy.ndarray
    """

    R = np.array([res_L, res_R, res_D])
    C = np.array(
        [
            [cap_L + cap_LR + cap_LD, cap_LR, cap_LD],
            [cap_LR, cap_R + cap_LR + cap_RD, cap_RD],
            [cap_LD, cap_RD, cap_D + cap_LD + cap_RD],
        ]
    )

    return generate_filter_stage_mapping_generic(frequency, R, C)


def reduce_mapping(M: np.ndarray) -> np.ndarray:
    """Reduces the 6 x 6 mapping (that includes the mappings for drain voltage
    and current) of a 3T setup and returns a 4 x 4 one (that does not) by
    imposing current conservation and voltage reference to drain at the
    sample.

    Parameters
    ----------
    M
        a 6 x 6 mapping to be reduced

    Returns
    -------
    a reduced 4 x 4 mapping
    """

    M_1 = np.zeros([5, 5], dtype=complex)

    for k, i in itertools.product(range(5), range(5)):
        M_1[k, i] = M[k, i] - M[k, 5] * sum(M[3:6, i]) / sum(M[3:6, 5])

    M_2 = np.zeros([4, 4], dtype=complex)
    for k, i in itertools.product(range(4), range(4)):
        l1ind = i + 1 if i >= 2 else i
        M_2[k, i] = M_1[k + 1, l1ind] if k >= 2 else M_1[k, l1ind] - M_1[2, l1ind]
    return M_2


def voltage_scaling_output_only(
    frequency_left: float,
    frequency_right: float,
    filters_left: dict[str, float],
    filters_right: dict[str, float],
) -> tuple[complex, complex]:
    """
    Determine the multipliers for the voltage amplitude to apply the specified
    voltages at the cryostat input corrected for the voltage output filters.

    For example, to apply 10 uV AC excitation at the cryostat input to left
    contact, the voltage at the instrument should be
    abs(V_output_scaling_left) x 10 uV.

    Parameters
    ----------
    frequency_left
        ac frequency of the left voltage excitation
    frequency_right
        ac frequency of the right voltage excitation
    filters_left
        Dictionary specifying the filters of the left voltage
        source, where the values are the -3 dB cutoff frequencies.
    filters_right
        Dictionary specifying the filters of the right voltage
        source, where the values are the -3 dB cutoff frequencies.

    Returns
    -------
    Tuple of complex values v_output_scaling_left and v_output_scaling_right
    """

    v_output_scaling_left = complex(1)
    for value in filters_left.values():
        v_output_scaling_left /= 1 + 1j * frequency_left / value

    v_output_scaling_right = complex(1)
    for value_ in filters_right.values():
        v_output_scaling_right /= 1 + 1j * frequency_right / value_

    return v_output_scaling_left, v_output_scaling_right


def include_filter_contributions(
    M: np.ndarray,
    frequency: float,
    voltage_source_properties: dict[str, dict[str, float]],
    fcut_left: float,
    gain_left_fcut: float,
    left_rise_time: float,
    fcut_right: float,
    gain_right_fcut: float,
    right_rise_time: float,
) -> np.ndarray:
    """Incorporate the scalings and phase shifts to the mapping M
    imposed by the filters of the voltage outputs or current
    preamplifiers. Format of voltage_source_properties dictionary should be
    as folllows.

    {
        'filters_left': {
            'cutoff_1': val1,
            'cutoff_2': val2,
            .
            .
            'cutoff_n': valn
            }
        'filters_right: {
            'cutoff_1': val1,
            'cutoff_2': val2,
            .
            .
            'cutoff_n': valn
            }
    }

    Parameters
    ----------
    M
        Matrix (numpy array) defining the mapping. It can be a 6x6 or a
        4x4 mapping
    frequency
        AC frequency to evaluate at.
    voltage_source_properties
        contains filter cutoff frequencies for
        left and right
    fcut_left
        -3 dB cutoff frequency of the lowpass filter on the left.
    gain_left_fcut
        Gain-bandwidth product of the preamplifier on the left.
    left_rise_time
        Rise time of the preamplifier on the left.
    fcut_right
        -3 dB cutoff frequency of the lowpass filter on the right.
    gain_right_fcut
        Gain-bandwidth product of the preamplifier on the right
    right_rise_time
        Rise time of the preamplifier on the right.

    Returns
    -------
    a mapping of type numpy.ndarray
    """

    M_out = M.copy()

    V_output_scaling_left, V_output_scaling_right = voltage_scaling_output_only(
        frequency,
        frequency,
        voltage_source_properties["filters_left"],
        voltage_source_properties["filters_right"],
    )

    I_output_scaling_left = (
        1 / (1 + 1j * frequency / fcut_left) / (1 + 1j * frequency / gain_left_fcut)
    )
    I_output_scaling_left *= np.exp(-(1j * frequency * left_rise_time))

    I_output_scaling_right = (
        1 / (1 + 1j * frequency / fcut_right) / (1 + 1j * frequency / gain_right_fcut)
    )
    I_output_scaling_right *= np.exp(-(1j * frequency * right_rise_time))

    if len(M) == 6:
        M_out[:, 0] = V_output_scaling_left * M_out[:, 0]
        M_out[:, 1] = V_output_scaling_right * M_out[:, 1]
        M_out[:, 3] = M_out[:, 3] / I_output_scaling_left
        M_out[:, 4] = M_out[:, 4] / I_output_scaling_right
    elif len(M) == 4:
        M_out[:, 0] = V_output_scaling_left * M_out[:, 0]
        M_out[:, 1] = V_output_scaling_right * M_out[:, 1]
        M_out[:, 2] = M_out[:, 2] / I_output_scaling_left
        M_out[:, 3] = M_out[:, 3] / I_output_scaling_right
    else:
        raise RuntimeError("Input matrix M must be either 6x6 or 4x4")

    return M_out


def generate_mapping(
    frequency: float, fridge_parameters: dict, reduced: bool = True
) -> np.ndarray:
    """Generate a matrix that maps the voltages applied and currents measured
    to voltages and currents at the 3T device. The mapping is generated for
    the given frequency using the parameters specified in fridge_parameters.

    By default, the matrix will have a dimension of 4 x 4, mapping voltages
    and currents at left and right source contacts. With reduced = False,
    the matrix will have a dimension of 6 x 6 and includes the voltages and
    currents at the drain contact.

    This function takes in a fridge_parameters dictionary. The structure of
    it should be as below. An example of the implementation in a yaml file can
    be found in data/fridge/deviceA1.yaml.

    {
    'voltage_source_properties': {
        'filters_left': {
            'cutoff_1': val1,
            'cutoff_2': val2,
            .
            .
            'cutoff_n': valn
            }
        'filters_right: {
            'cutoff_1': val1,
            'cutoff_2': val2,
            .
            .
            'cutoff_n': valn
            }
        },
    'line_properties': {
        'stage_1': {
            'Resistance_L': value,
            'Resistance_R': value,
            'Resistance_D': value,
            'Capacitance_L': value,
            'Capacitance_R': value,
            'Capacitance_D': value,
            'Capacitance_LR': value,
            'Capacitance_LD': value,
            'Capacitance_RD': value,
            },
        'stage_2': { ... },
        .,
        .,
        'stage_n': { ... }
        }
     'amplifier_properties': {
        'amplifier_left': {
            'gain': value,
            'GWBP': value,
            'input_impedance_0': value,
            'fcut': value,
            'setting_table': {
                'gain': {
                            gain_value: {'fcut': value}
                        },
                'fcut': {
                            fcut_value: {'rise_time': value}
                        }
                }
            },
        'amplifier_right': {...}
    }

    Parameters
    ----------
    frequency
        Frequency in Hz
    fridge_parameters
        A dictionary of fridge parameter information, for
        formatting, see above.
    reduced
        Boolean to reduce the matrix from 6x6 to 4x4 (default: True).

    Returns
    -------
    mapping of type numpy.ndarray
    """

    line_properties = fridge_parameters["line_properties"]
    amplifier_properties = fridge_parameters["amplifier_properties"]
    voltage_source_properties = fridge_parameters["voltage_source_properties"]

    left_amp_input_imp_0 = amplifier_properties["amplifier_left"]["input_impedance_0"]
    left_gain = amplifier_properties["amplifier_left"]["gain"]
    left_gwbp = amplifier_properties["amplifier_left"]["GWBP"]
    right_amp_input_imp_0 = amplifier_properties["amplifier_right"]["input_impedance_0"]
    right_gain = amplifier_properties["amplifier_right"]["gain"]
    right_gwbp = amplifier_properties["amplifier_right"]["GWBP"]

    M = generate_amplifier_circuit_3T(
        frequency,
        left_amp_input_imp_0,
        left_gain,
        left_gwbp,
        right_amp_input_imp_0,
        right_gain,
        right_gwbp,
    )

    for stage in line_properties.keys():
        res_L = line_properties[stage]["Resistance_L"]
        res_R = line_properties[stage]["Resistance_R"]
        res_D = line_properties[stage]["Resistance_D"]

        cap_L = line_properties[stage]["Capacitance_L"]
        cap_R = line_properties[stage]["Capacitance_R"]
        cap_D = line_properties[stage]["Capacitance_D"]

        cap_LR = line_properties[stage]["Capacitance_LR"]
        cap_LD = line_properties[stage]["Capacitance_LD"]
        cap_RD = line_properties[stage]["Capacitance_RD"]

        M1 = generate_filter_stage_3T(
            frequency, res_L, res_R, res_D, cap_L, cap_R, cap_D, cap_LR, cap_LD, cap_RD
        )
        M = np.matmul(M1, M)

    fcut_left = amplifier_properties["amplifier_left"]["fcut"]
    gain_left_fcut = amplifier_properties["setting_table"]["gain"][left_gain]["fcut"]
    left_rise_time = amplifier_properties["setting_table"]["fcut"][fcut_left][
        "rise_time"
    ]

    fcut_right = amplifier_properties["amplifier_left"]["fcut"]
    gain_right_fcut = amplifier_properties["setting_table"]["gain"][left_gain]["fcut"]
    right_rise_time = amplifier_properties["setting_table"]["fcut"][fcut_left][
        "rise_time"
    ]

    M = include_filter_contributions(
        M,
        frequency,
        voltage_source_properties,
        fcut_left,
        gain_left_fcut,
        left_rise_time,
        fcut_right,
        gain_right_fcut,
        right_rise_time,
    )

    return reduce_mapping(M) if reduced else M


def convert_conductances_array(
    g_ll: xarray.Dataset,
    g_rl: xarray.Dataset,
    g_lr: xarray.Dataset,
    g_rr: xarray.Dataset,
    map_left: np.ndarray,
    map_right: np.ndarray,
):
    """Convert the conductances.

    Parameters
    ----------
    g_ll
        xarray Dataset of the left local conductances as extracted by
    g_rl
        xarray Dataset of the nonlocal conductances from left to right as extracted by
    g_lr
        xarray Dataset of the left local conductances as extracted by
    g_rr
        xarray Dataset of the nonlocal conductances from left to right as extracted by
    map_left
        the mapping for the excitation at the left source contact, generated by generate_mapping()
    map_right
        the mapping for the excitation at the right source contact, generated by generate_mapping()
    fridge_parameters
        dictionary used by generate_mapping(). Only required if at least one of the sweep
        parameters has been frequency.
    force_real
        boolean that, when true, causes the return values to be real. (default: False)

    Returns
    -------
    A tuple of converted g_ll, g_rl, g_lr, g_rr.
    """

    g0 = e**2 / h

    i_ll = g_ll * g0
    i_rl = g_rl * g0
    i_lr = g_lr * g0
    i_rr = g_rr * g0

    V_L1 = map_left[0, 0] + map_left[0, 2] * i_ll + map_left[0, 3] * i_rl
    V_R1 = map_left[1, 0] + map_left[1, 2] * i_ll + map_left[1, 3] * i_rl
    I_L1 = map_left[2, 0] + map_left[2, 2] * i_ll + map_left[2, 3] * i_rl
    I_R1 = map_left[3, 0] + map_left[3, 2] * i_ll + map_left[3, 3] * i_rl

    V_L2 = map_right[0, 1] + map_right[0, 2] * i_lr + map_right[0, 3] * i_rr
    V_R2 = map_right[1, 1] + map_right[1, 2] * i_lr + map_right[1, 3] * i_rr
    I_L2 = map_right[2, 1] + map_right[2, 2] * i_lr + map_right[2, 3] * i_rr
    I_R2 = map_right[3, 1] + map_right[3, 2] * i_lr + map_right[3, 3] * i_rr

    divider = g0 * (V_L1 * V_R2 - V_R1 * V_L2)

    g_ll_out = (I_L1 * V_R2 - I_L2 * V_R1) / divider
    g_lr_out = (-I_L1 * V_L2 + I_L2 * V_L1) / divider
    g_rl_out = (I_R1 * V_R2 - I_R2 * V_R1) / divider
    g_rr_out = (-I_R1 * V_L2 + I_R2 * V_L1) / divider

    return g_ll_out, g_rl_out, g_lr_out, g_rr_out


def _freq(ds: xr.Dataset, lockin: str, fridge_parameters: dict) -> np.ndarray:
    snapshot = json.loads(ds.snapshot)
    keys = [
        "station",
        "instruments",
        lockin,
        "submodules",
        "oscs",
        "channels",
        f"{lockin}_oscs0",
        "parameters",
        "freq",
        "value",
    ]
    try:
        freq = get_in(keys, snapshot, no_default=True)
    except KeyError:
        keys = tuple(snapshot["station"]["instruments"].keys())
        raise KeyError(
            f"The key lockin={lockin} is not found in `ds.snapshot`, pick one of {keys}."
        ) from None
    return generate_mapping(freq, fridge_parameters)


def apply_phase_shift(
    ds_left: xr.Dataset,
    ds_right: xr.Dataset,
    phase_shift_left: float,
    phase_shift_right: float,
):
    """Apply a phase shift to the conductances."""
    phase_factor_left = np.exp(1.0j * phase_shift_left * np.pi / 180.0)
    phase_factor_right = np.exp(1.0j * phase_shift_right * np.pi / 180.0)
    for ds in (ds_left, ds_right):
        for key in ("g_ll", "g_rl"):
            ds[key] = ds[key] * phase_factor_left
        for key in ("g_rr", "g_lr"):
            ds[key] = ds[key] * phase_factor_right


def correct_frequencies(
    ds_left: xr.Dataset,
    ds_right: xr.Dataset,
    lockin_left: str,
    lockin_right: str,
    fridge_parameters: dict | str,
    phase_shift_left: float = 0.0,
    phase_shift_right: float = 0.0,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Correct the conductance data.

    Parameters
    ----------
    ds_left
        Dataset of the data taken at the left.
    ds_right
        Dataset of the data taken at the right.
    lockin_left
        Lockin at the left, as defined in the snapshot.
    lockin_right
        Lockin at the right, as defined in the snapshot.
    fridge_parameters
        Fridge parameters dictionary. If a string is given, it is
        assumed to be the path to the YAML file.
    phase_shift_left
        Phase shift of the conductance data on the left in degrees.
    phase_shift_right
        Phase shift of the conductance data on the right in degrees.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        Left and right datasets with corrected conductances.

    Raises
    ------
    RuntimeError
        If the data has already been corrected.
    """
    if ds_left.attrs.get("correct_frequencies.lockin_left"):
        raise RuntimeError("Dataset has already been frequency corrected.")

    apply_phase_shift(ds_left, ds_right, phase_shift_left, phase_shift_right)

    if not isinstance(fridge_parameters, dict):
        with open(fridge_parameters) as f:
            fridge_parameters = yaml.safe_load(f)
    freq_left, freq_right = (
        _freq(ds, lockin, fridge_parameters)
        for ds, lockin in [(ds_left, lockin_left), (ds_right, lockin_right)]
    )
    (g_ll, g_rl, _, _), (_, _, g_lr, g_rr) = (
        convert_conductances_array(
            ds["g_ll"], ds["g_rl"], ds["g_lr"], ds["g_rr"], freq_left, freq_right
        )
        for ds in (ds_left, ds_right)
    )
    left = xr.Dataset({"g_ll": g_ll.real, "g_rl": g_rl.real})
    right = xr.Dataset({"g_lr": g_lr.real, "g_rr": g_rr.real})

    attrs = {
        "correct_frequencies.lockin_left": lockin_left,
        "correct_frequencies.lockin_right": lockin_right,
        "correct_frequencies.phase_shift_left": phase_shift_left,
        "correct_frequencies.phase_shift_right": phase_shift_right,
    }
    left.attrs.update(ds_left.attrs)
    left.attrs.update(attrs)
    right.attrs.update(ds_right.attrs)
    right.attrs.update(attrs)
    return left, right


def output_current_closed(Map: np.ndarray, V: list[float]) -> np.ndarray:
    """
    Evaluate the predicted measured current when all conductances are zero
    for a given mapping and excitation voltage.

    Parameters
    ----------
    Map
        The mapping for the applied excitation
    V
        A list in the form [V_left, V_right] with the AC voltage applied to
        left or right.

    Returns
    -------
    Predicted measured current as numpy array.
    """

    # Translate the voltage(s) applied into a vector
    V_vec = np.array([V[0], V[1]], dtype=complex)

    # Evaluate the measured current
    I_vec = np.matmul(np.matmul(-np.linalg.inv(Map[2:4, 2:4]), Map[2:4, 0:2]), V_vec)

    return I_vec


def output_current_open(Map: np.ndarray, V: list[float]) -> np.ndarray:
    """
    Evaluate the predicted measured current when all conductances are infinite
    for a given mapping and excitation voltage.

    Parameters
    ----------
    Map
        The mapping for the applied excitation.
    V
        A list in the form [V_left, V_right] with the AC voltage applied to
        left or right.

    Returns
    -------
    Predicted measured current as numpy array.
    """

    # Translate the voltage(s) applied into a vector
    V_vec = np.array([V[0], V[1]], dtype=complex)

    # Evaluate the measured current
    I_vec = np.matmul(np.matmul(-np.linalg.inv(Map[0:2, 2:4]), Map[0:2, 0:2]), V_vec)

    return I_vec
