#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:33:33 2023

@author: peterson
"""
import numpy as np

import src.physics.thermodynamics as thermo


def replace_out_of_bounds_values(array, threshold_value=1e-6, is_lower_bound=True, array_name="Array", eps=1e-12):
    """
    Replace values in an array based on a threshold and check for NaNs/Infs.

    Parameters:
    - array: numpy.ndarray, the input array to modify.
    - threshold_value: float, the value to replace out-of-bound elements with.
    - is_lower_bound: bool, True to replace values below the threshold,
                           False to replace values above the threshold.
    - array_name: str, name of the array for logging purposes.
    - eps: float, tolerance for significant changes.

    Returns:
    - updated_array: numpy.ndarray with modified values.
    """
    if is_lower_bound:
        # Identify values to replace
        out_of_bounds_mask = array < threshold_value
    else:
        # Identify values to replace
        out_of_bounds_mask = array > threshold_value

    # Create a copy to track changes
    updated_array = array.copy()
    updated_array[out_of_bounds_mask] = threshold_value

    # Identify indices of changes
    changed_indices = np.argwhere(out_of_bounds_mask)

    for idx in changed_indices:
        idx = tuple(idx)  # Convert to tuple for indexing
        original_value = array[idx]
        new_value = updated_array[idx]

        # Check for significant changes
        sign_changed = np.sign(original_value) != np.sign(new_value)
        magnitude_change = abs(original_value - new_value) > 2*eps

        if sign_changed or magnitude_change:
            tempppp = 0
            #import pdb; pdb.set_trace()
            #print(f"{array_name}[{idx}] changed from {original_value} to {new_value}")

    # Check for NaNs and Infs
    if np.any(np.isnan(updated_array)):
        tempppp = 0
        #print(f"Warning: {array_name} contains NaN values!")
        #import pdb; pdb.set_trace()
    if np.any(np.isinf(updated_array)):
        tempppp = 0
        #print(f"Warning: {array_name} contains infinite values!")
        #import pdb; pdb.set_trace()

    return updated_array


def format_time(seconds):
    """
    Convert seconds to a string format of hours, minutes, and seconds.

    Parameters:
    seconds (float): Time in seconds.

    Returns:
    str: Formatted time string in hours, minutes, and seconds.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    return f"{hours}h {minutes}m {seconds}s"


def get_center_temperature(X_CO2, T, data):
    """Get temperature at the reactor center based on average temperature."""
    T_cool = data["T_cool"]
    v_gas_in = data["v_gas_in"]
    X_IV = X_CO2[0]
    T_IV = T[0]
    X_CO2 = X_CO2[1:]
    T = T[1:]

    ###########################################################################
    # INITIAL CONDITIONS
    ###########################################################################
    # pressure drop in Pa
    p_loss = 0
    p_R_loss = data["p_R"]-p_loss*data["zeta"]*data["L_R"]

    ###########################################################################
    # CONVERSIONS
    ###########################################################################
    # get part of fractions
    n_in_1 = data["n_in"][1]
    n_i = data["n_in"][:, None, None] + data["nue"][:, None, None] * X_CO2[None, :] * n_in_1
    n = np.sum(n_i, axis=0)
    x_i = n_i / n
    # M_gas
    # M_gas = np.sum(x_i.T*data["Molar_Mass"], axis=2).T
    M_gas = np.einsum('ijk,i->jk', x_i, data["Molar_Mass"])
    # Gas densitity by ideal gas law in kg/m^3 (validiert)
    density_fluid = (p_R_loss[:, None] * M_gas) / (data["R"] * T)
    # Mass flow (axial mass flow remains constant) - quasistationary
    v_gas = v_gas_in * data["density_fluid_in"] / density_fluid

    ###########################################################################
    # THERMODYNAMICS
    ###########################################################################
    expanded_X_prop = np.tile(data["X_CO2_prop"][:, np.newaxis],
                              (1, X_CO2.shape[1]))
    n_i_prop = data["n_in"][:, None, None] \
        + data["nue"][:, None, None] * expanded_X_prop[None, :] * n_in_1
    n_prop = np.sum(n_i_prop, axis=0)
    x_i_prop = n_i_prop / n_prop

    # Initialize empty lists to store the results
    U_list = []
    U_slash_list = []
    # List comprehension to compute U and U_slash
    for i in range(X_CO2.shape[1]):
        U_i, U_slash_i, _ = thermo.get_heat_transfer_coeff(
            T[:, i], x_i_prop[:, :, i], v_gas[:, i], density_fluid[:, i])
        U_list.append(U_i.flatten())
        U_slash_list.append(U_slash_i.flatten())
    # Convert lists to numpy arrays and concatenate along the appropriate axis
    U = np.stack(U_list, axis=1)
    U_slash = np.stack(U_slash_list, axis=1)

    temperature_difference = T - T_cool
    T_center = U/U_slash*(temperature_difference) + T_cool
    T_center = np.concatenate([T_IV[np.newaxis, :], T_center], axis=0)
    return T_center
