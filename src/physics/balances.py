"""
Solving Partial-Differential Equation 1D-PFTR for Methanantion
Author: Luisa Peterson
"""

###########################################################################
# Import packages
###########################################################################
import numpy as np

import src.physics.thermodynamics as thermo
import src.physics.pressure_drop as pd
from src.physics.utils import replace_out_of_bounds_values
from src.physics.reaction_rate import reaction_rate_LHHW, reaction_rate_PL
from src.physics.effective_diffusion import effective_diffusion

import ipdb

eps = np.finfo(float).eps

###########################################################################
# location-discretized partial differential equations
###########################################################################
def PDE_sys(t, x, data):
    """Build up mass and energy balances for the methanation reaction."""
    X_CO2 = x[:,0]          #[0*data["Nz"]:1*data["Nz"]]
    tau = x[:,1]            #[1*data["Nz"]:2*data["Nz"]]
    
    T = tau#*data["T_scale"]+data["T_scale_add"]
    X_CO2 = replace_out_of_bounds_values(X_CO2, eps, array_name='X_CO2_'+str(t))
    X_CO2 = replace_out_of_bounds_values(X_CO2,
                                         threshold_value=1,
                                         is_lower_bound=False,
                                         array_name='X_CO2_'+str(t))
    T = replace_out_of_bounds_values(T, data["T_min"], array_name="T_"+str(t))
    ###########################################################################
    # TIME-CHANGING PARAMATERS
    ###########################################################################
    T_cool = np.interp(t, data["t_points"], data["T_cool"])
    v_gas_in = np.interp(t, data["t_points"], data["v_gas_in"])

    ###########################################################################
    # INITIAL CONDITIONS
    ###########################################################################
    # pressure drop in Pa
    p_loss = 0 # pd.pressure_drop(data["T_gas_in"], data["x_in"], v_gas_in)
    p_R_loss = data["p_R"]-p_loss*data["zeta"]*data["L_R"]

    ###########################################################################
    # CONVERSIONS
    ###########################################################################
    # get part of fractions
    n_i = data["n_in"][:, None] \
        + data["nue"][:, None]*X_CO2[None, :]*data["n_in"][1]
    # Set negative values to eps
    n_i = replace_out_of_bounds_values(n_i, eps, array_name="n_i_"+str(t))
    # mole fraction
    n = np.sum(n_i, axis=0)
    x_i = (n_i/n)
    # M_gas
    M_gas = np.sum(x_i.T*data["Molar_Mass"], axis=1)
    # Gas densitity by ideal gas law in kg/m^3 (validiert)
    density_fluid = (p_R_loss*M_gas/data["R"]/(T)).T
    # Mass flow (axial mass flow remains constant) - quasistationary
    v_gas = v_gas_in*data["density_fluid_in"]/density_fluid

    ###########################################################################
    # THERMODYNAMICS
    ###########################################################################
    # get thermodynamics
    cp_fluid = thermo.get_cp_fluid(data["T_prop"], data["x_i_prop"])
    H_r = thermo.get_reaction_enthalpy(T)
    U, _, lambda_ax = thermo.get_heat_transfer_coeff(
        T, data["x_i_prop"], v_gas, density_fluid)

    ###########################################################################
    # REACTION RATE
    ###########################################################################
    # mass based reactino rate in  mol/(s*kg_cat)
    if np.min(x_i[0, :]) < 1e-4:
        r = np.zeros_like(T)
    else:
        r = reaction_rate_PL(T, x_i, p_R_loss)
    # molar Reaction Rate in mol/(s*mcat^3)
    if np.all(r == 0):
        r_eff = r
    else:
        r_int = (1-data["epsilon_core"])*data["density_cat_core"]*r
        eta = effective_diffusion(T, p_R_loss, x_i, v_gas, r_int)
        r_eff = data["cat_frac"]*(1-data["epsilon"])*eta*r_int
        
    ###########################################################################
    # BALANCES
    ###########################################################################
    # Initialize composition matrix
    dxdt = np.zeros((2, data["Nz"]))
    dz = data["D_zeta"] * data["L_R"]
    dz_2 = data["d_zeta"]*data["L_R"]

    # Mass Balance
    dxdt_mass_0 = -v_gas[0] * (X_CO2[0]-data["X_in"])/data["epsilon"]/dz[0] \
        + data["Molar_Mass"][1]*r_eff[0]/(data["w_in"][1]+eps)/density_fluid[0]/data["epsilon"]
    dxdt[0, 0] = dxdt_mass_0
    # inner control volumes
    dxdt_mass = -v_gas[1:data["Nz"]]*(X_CO2[1:data["Nz"]]-X_CO2[0:data["Nz"]-1]) \
        / data["epsilon"] / dz[1:data["Nz"]] \
        + data["Molar_Mass"][1]*r_eff[1:data["Nz"]] \
        / ((data["w_in"][1]+eps)*density_fluid[1:data["Nz"]]*data["epsilon"]+eps)
    dxdt[0, 1:data["Nz"]] = dxdt_mass

    # Pre-compute constants and repeated expressions
    lambda_ax_mean = 0.5 * (lambda_ax[:-1] + lambda_ax[1:])  # arithmetic mean
    lambda_ax_mean[lambda_ax_mean < 0] = 0
    # Expressions for Axial Dispersion
    Ax_Disp = np.zeros(data["Nz"])
    # Vectorized computation for the inner elements
    i = np.arange(1, data["Nz"]-1)
    Ax_Disp[i] = lambda_ax_mean[i]*(T[i+1]-T[i])/(dz_2[i]*dz[i]) \
        - lambda_ax_mean[i-1]*(T[i]-T[i-1])/(dz_2[i-1]*dz[i])
    # left BC
    Ax_Disp[0] = lambda_ax_mean[0] * (T[1] - T[0]) / (dz_2[0]*dz[0]) \
        -lambda_ax[0]*(T[-1]-T[-2])/(dz_2[0]*(dz[0]))
    # right BC
    Ax_Disp[-1] = 0-lambda_ax_mean[-2]*(T[-1]-T[-2])/(dz_2[-2]*(dz[-1]))


    # Kinetics for heat transfer (Wärmeübergang Gas-Wand)
    g_qw = 4 * U * (T - T_cool) / data["D_R"]

    # Calculating effective heat capacity
    rhocp_cat = ((data["r_core"]**3 / data["r_shell"]**3) * (1 - data["epsilon_core"]) * data["density_cat_core"] * data["cp_core"]
                 + (data["r_shell"]**3 - data["r_core"]**3) / data["r_shell"]**3 * (1 - data["epsilon_shell"]) * data["density_cat_shell"] * data["cp_shell"])
    rhocp_g = density_fluid*cp_fluid*data["epsilon"] \
        + rhocp_cat*(1-data["epsilon"])
    dxdt_T_0 = (-v_gas_in*data["density_fluid_in"]*cp_fluid[0]
                   * ((T[0]-data["T_gas_in"])/dz[0]))/rhocp_g[0] \
        - r_eff[0].T*H_r[0]/rhocp_g[0] \
        - g_qw[0]/rhocp_g[0] \
        + Ax_Disp[0]/rhocp_g[0]
    dxdt_tau_0 = dxdt_T_0/data["T_scale"]
    dxdt[1, 0] = dxdt_tau_0
    # ODEs inner control volumes
    dxdt_T = (-v_gas_in*data["density_fluid_in"]*cp_fluid[1:data["Nz"]]
                 * ((T[1:data["Nz"]]-T[0:data["Nz"]-1])/dz[1:data["Nz"]])) \
        / rhocp_g[1:data["Nz"]] \
        - r_eff[1:data["Nz"]]*H_r[1:data["Nz"]]/rhocp_g[1:data["Nz"]] \
        - g_qw[1:data["Nz"]]/rhocp_g[1:data["Nz"]]\
        + Ax_Disp[1:data["Nz"]]/rhocp_g[1:data["Nz"]]
    dxdt_tau = dxdt_T#/data["T_scale"]
    dxdt[1, 1:data["Nz"]] = dxdt_tau
    # Flatten to create a long row vector with entries for each component in each control volume
    #dxdt = dxdt.flatten()
    return dxdt
